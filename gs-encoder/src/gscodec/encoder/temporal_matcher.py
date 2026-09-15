"""
Temporal matcher for unstructured Gaussian Splatting sequences.

Establishes frame-to-frame correspondences via iterative FAISS nearest-neighbor
matching, enabling compression of sequences where Gaussian counts and ordering
differ between frames. Conforms all frames in a chunk to a fixed target count
(global maximum) with consistent slot ordering. Frames with fewer Gaussians
are padded with disabled Gaussian slots.
"""

import logging

import faiss
import torch
from gsply import GSTensor

logger = logging.getLogger(__name__)

# Per-Gaussian attributes carried through slot assignment when conforming a chunk.
# shN is included so higher-order SH is matched/propagated rather than dropped.
_SLOT_ATTRS = ("means", "scales", "quats", "opacities", "sh0", "shN", "masks")


class TemporalMatcher:
    """Matches Gaussians across frames for temporal coherence.

    Uses iterative NN matching to assign each frame's Gaussians to consistent
    slots relative to the previous frame. Frames are padded to target_n;
    padded slots have their presence mask cleared.
    """

    def __init__(self, k_passes: int = 10, device: str = "cpu"):
        self.k_passes = k_passes
        self.device = device

    def conform_chunk(
        self, frames: list[GSTensor], target_n: int
    ) -> list[GSTensor]:
        """Conform all frames to target_n Gaussians with consistent ordering.

        Assumes frames are already Morton-sorted externally.

        1. First frame: pad to target_n (invisible padding)
        2. Each subsequent frame: NN match to previous, pad remainder

        Args:
            frames: List of GSTensor frames (already Morton-sorted).
            target_n: Target Gaussian count (global maximum, 4-aligned).

        Returns:
            List of conformed GSTensor frames, each with exactly target_n Gaussians.
        """
        if not frames:
            raise ValueError("Cannot conform empty chunk")

        # Unstructured inputs have no persistent row identity. Only active
        # candidates participate in matching; inactive rows are free capacity.
        frames = [f[f.masks.reshape(-1).bool()] for f in frames]

        conformed = [self._pad_frame(frames[0], target_n)]

        for i in range(1, len(frames)):
            conformed_frame = self._conform_frame(
                frames[i], conformed[i - 1], target_n
            )
            conformed.append(conformed_frame)

        return conformed

    def _pad_frame(self, frame: GSTensor, target_n: int) -> GSTensor:
        """Pad a frame to target_n with disabled slots.

        Args:
            frame: Input GSTensor with N <= target_n Gaussians.
            target_n: Target Gaussian count.

        Returns:
            GSTensor with exactly target_n Gaussians.
        """
        n = frame.means.shape[0]
        if n >= target_n:
            return frame[:target_n]

        result = self._create_empty_like(frame, target_n)
        src_indices = torch.arange(n, device=self.device)
        self._assign_slots(result, frame, src_indices, src_indices)

        # The binary mask is authoritative for padded slots.
        self._retire_slots(result, torch.arange(n, target_n, device=self.device))

        return result

    def _conform_frame(
        self,
        current: GSTensor,
        previous: GSTensor,
        target_n: int,
    ) -> GSTensor:
        """Match current frame to previous via NN, produce target_n output.

        Phase 1: Fill slots with matched Gaussians from current frame.
        Phase 2: Fill remaining empty slots with unmatched current Gaussians.
        Phase 3: Disable remaining empty slots, retaining their attributes.

        Args:
            current: Current frame GSTensor (may differ from target_n).
            previous: Previous conformed frame (exactly target_n).
            target_n: Target Gaussian count.

        Returns:
            Conformed GSTensor with exactly target_n Gaussians.
        """
        prev_means = previous.means  # [target_n, 3]
        curr_means = current.means  # [M, 3]
        n_curr = curr_means.shape[0]

        active_slots = torch.where(previous.masks.reshape(-1).bool())[0]
        curr_indices, active_indices = self._perform_matching(prev_means[active_slots], curr_means)
        slot_indices = active_slots[active_indices]

        # Retain previous attributes for slots that become inactive.
        result = self._create_empty_like(previous, target_n)
        self._copy_slots(result, previous, torch.arange(target_n, device=self.device))

        filled_slots = torch.zeros(target_n, dtype=torch.bool, device=self.device)
        used_curr = torch.zeros(n_curr, dtype=torch.bool, device=self.device)

        # Phase 1: fill matched slots with current frame data
        if curr_indices.numel() > 0:
            self._assign_slots(result, current, slot_indices, curr_indices)
            filled_slots[slot_indices] = True
            used_curr[curr_indices] = True

        # Phase 2: fill remaining empty slots with unmatched current Gaussians
        empty_slots = torch.where(~filled_slots)[0]
        if len(empty_slots) > 0:
            unmatched_curr = torch.where(~used_curr)[0]
            # Reuse vacant slots by proximity before the deterministic fallback.
            vacant = empty_slots[~previous.masks.reshape(-1).bool()[empty_slots]]
            if len(vacant) and len(unmatched_curr):
                candidate, local_slot = self._match_vacant_slots(previous, current, vacant, unmatched_curr)
                self._assign_slots(result, current, vacant[local_slot], unmatched_curr[candidate])
                filled_slots[vacant[local_slot]] = True
                used_curr[unmatched_curr[candidate]] = True
                empty_slots = torch.where(~filled_slots)[0]
                unmatched_curr = torch.where(~used_curr)[0]
            n_fill = min(len(empty_slots), len(unmatched_curr))
            if n_fill > 0:
                self._assign_slots(
                    result, current,
                    empty_slots[:n_fill], unmatched_curr[:n_fill],
                )
                empty_slots = empty_slots[n_fill:]

        # Retired slots retain attributes for temporal compression.
        if len(empty_slots) > 0:
            self._retire_slots(result, empty_slots)

        return result

    def _perform_matching(
        self,
        prev_means: torch.Tensor,
        curr_means: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-pass nearest-neighbor matching with conflict resolution.

        Each pass runs a fresh FAISS NN search over the still-unmatched current
        points against the still-unclaimed previous slots, so every pass finds
        the true nearest available slot. When several current Gaussians claim the
        same slot the closest wins; the losers carry over to the next pass.

        This full-re-search scheme was benchmarked against a single cached k-NN
        search (gaussian_tree style) and a GPU brute-force ``cdist``; both were
        slower and gave worse temporal coherence, so the multi-pass design is
        retained.

        Args:
            prev_means: Previous frame means [N_prev, 3].
            curr_means: Current frame means [N_curr, 3].

        Returns:
            (curr_indices, slot_indices): parallel LongTensors where current
            Gaussian ``curr_indices[k]`` is assigned to previous slot
            ``slot_indices[k]``.
        """
        n_prev = prev_means.shape[0]
        n_curr = curr_means.shape[0]

        # matched_slot[c] = previous slot assigned to current Gaussian c (-1 = none)
        matched_slot = torch.full((n_curr,), -1, dtype=torch.long, device=self.device)
        unmatched_curr = torch.ones(n_curr, dtype=torch.bool, device=self.device)
        unclaimed_prev = torch.ones(n_prev, dtype=torch.bool, device=self.device)

        for _ in range(self.k_passes):
            curr_idx = torch.nonzero(unmatched_curr, as_tuple=True)[0]
            prev_idx = torch.nonzero(unclaimed_prev, as_tuple=True)[0]
            if curr_idx.numel() == 0 or prev_idx.numel() == 0:
                break

            nn_local, dist = self._nn_search(curr_means[curr_idx], prev_means[prev_idx])
            nn_prev_global = prev_idx[nn_local]  # [U] claimed prev slot per candidate

            winners = self._closest_per_target(nn_prev_global, dist, n_prev)
            if not winners.any():
                break

            win_curr = curr_idx[winners]
            win_prev = nn_prev_global[winners]
            matched_slot[win_curr] = win_prev
            unmatched_curr[win_curr] = False
            unclaimed_prev[win_prev] = False

        curr_indices = torch.nonzero(matched_slot >= 0, as_tuple=True)[0]
        slot_indices = matched_slot[curr_indices]
        return curr_indices, slot_indices

    def _match_vacant_slots(self, previous, current, vacant, candidates):
        """Bounded spatial candidates, ranked by position/color/log-scale change."""
        database = previous.means[vacant].detach().cpu().numpy().astype("float32")
        query = current.means[candidates].detach().cpu().numpy().astype("float32")
        index = faiss.IndexFlatL2(3)
        index.add(database)
        distances, neighbors = index.search(query, min(8, len(vacant)))
        neighbors = torch.from_numpy(neighbors).to(self.device, torch.long)
        distances = torch.from_numpy(distances).to(self.device)
        spatial_scale = distances[:, -1:].clamp_min(1e-12)
        slots = vacant[neighbors]
        color = (current.sh0[candidates, None] - previous.sh0[slots]).square().mean(dim=-1)
        scale = (current.scales[candidates, None] - previous.scales[slots]).square().mean(dim=-1)
        cost = distances / spatial_scale + 0.1 * color + 0.1 * scale
        best = cost.argmin(dim=1)
        chosen = neighbors[torch.arange(len(candidates), device=self.device), best]
        scores = cost[torch.arange(len(candidates), device=self.device), best]
        winners = self._closest_per_target(chosen, scores, len(vacant))
        rows = torch.where(winners)[0]
        return rows, chosen[rows]

    def _nn_search(
        self, query: torch.Tensor, database: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest neighbor for 3-D points via FAISS flat index.

        FAISS's blocked-SIMD flat index outperforms a brute-force ``cdist`` here:
        it avoids materializing the [Q, D] distance matrix and its memory traffic.

        Args:
            query: [Q, 3] points to match.
            database: [D, 3] candidate points.

        Returns:
            (nn_idx [Q] indices into database, nn_dist [Q] squared distances).
            Squared distance is monotonic in distance, which is all the
            conflict-resolution step needs.
        """
        q = query.detach().cpu().numpy().astype("float32")
        db = database.detach().cpu().numpy().astype("float32")
        index = faiss.IndexFlatL2(3)
        index.add(db)
        dist2, idx = index.search(q, 1)
        nn_idx = torch.from_numpy(idx.reshape(-1)).to(self.device, torch.long)
        nn_dist = torch.from_numpy(dist2.reshape(-1)).to(self.device, query.dtype)
        return nn_idx, nn_dist

    def _closest_per_target(
        self, targets: torch.Tensor, dist: torch.Tensor, n_slots: int
    ) -> torch.Tensor:
        """Select, per target slot, the single candidate with minimum distance.

        Args:
            targets: [U] target slot index claimed by each candidate.
            dist: [U] distance of each candidate to its claimed slot.
            n_slots: Total number of target slots.

        Returns:
            [U] bool mask: True for candidates that win their target slot (one
            winner per claimed slot, ties broken by lowest candidate index).
        """
        best = torch.full((n_slots,), float("inf"), dtype=dist.dtype, device=self.device)
        best.scatter_reduce_(0, targets, dist, reduce="amin", include_self=True)
        is_best = dist == best[targets]

        # Break ties (and duplicate-distance collisions) so each slot keeps one
        # winner: among is_best candidates, keep the first per target slot.
        cand = torch.nonzero(is_best, as_tuple=True)[0]
        order = torch.argsort(targets[cand], stable=True)
        cand_sorted = cand[order]
        tgt_sorted = targets[cand_sorted]
        keep = torch.ones_like(cand_sorted, dtype=torch.bool)
        keep[1:] = tgt_sorted[1:] != tgt_sorted[:-1]

        winners = torch.zeros_like(is_best)
        winners[cand_sorted[keep]] = True
        return winners

    def _retire_slots(self, frame: GSTensor, indices: torch.Tensor) -> None:
        """Disable slots without changing their retained attributes.

        Args:
            frame: Target GSTensor to modify in place.
            indices: Slot indices to retire.
        """
        frame.masks[indices] = False

    def _create_empty_like(self, template: GSTensor, size: int) -> GSTensor:
        """Create a zero-filled GSTensor matching template's structure.

        ``shN`` is included so higher-order SH survives matching/conforming;
        ``torch.zeros((size, *val.shape[1:]))`` handles its [N, K, 3] shape.
        """
        data: dict[str, torch.Tensor | None] = {}
        for key in _SLOT_ATTRS:
            val = getattr(template, key, None)
            if val is not None:
                data[key] = torch.zeros(
                    (size, *val.shape[1:]), dtype=val.dtype, device=self.device
                )
            else:
                data[key] = None
        data["masks"] = torch.zeros(size, dtype=torch.bool, device=self.device)
        data["quats"][:, 0] = 1
        return GSTensor(**data)

    def _copy_slots(
        self,
        target: GSTensor,
        source: GSTensor,
        indices: torch.Tensor,
    ) -> None:
        """Copy all attribute slots from source to target at given indices."""
        for key in _SLOT_ATTRS:
            tgt = getattr(target, key, None)
            src = getattr(source, key, None)
            if tgt is not None and src is not None:
                tgt[indices] = src[indices]

    def _assign_slots(
        self,
        target: GSTensor,
        source: GSTensor,
        target_indices: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> None:
        """Assign source[source_indices] into target[target_indices]."""
        for key in _SLOT_ATTRS:
            tgt = getattr(target, key, None)
            src = getattr(source, key, None)
            if tgt is not None and src is not None:
                tgt[target_indices] = src[source_indices]
