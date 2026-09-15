"""Synthetic camera generation for grid-snap fine-tuning.

Places cameras on an orbit sphere around the scene centroid using
a golden spiral distribution for uniform coverage.
"""

import math

import torch
from torch import Tensor


class SyntheticCameras:
    """Generates orbit cameras on a sphere around the scene.

    Attributes:
        viewmats: [C, 4, 4] world-to-camera transforms.
        Ks: [C, 3, 3] intrinsic matrices.
        n_cameras: Number of cameras.
    """

    def __init__(
        self,
        viewmats: Tensor,
        Ks: Tensor,
    ):
        self.viewmats = viewmats  # [C, 4, 4]
        self.Ks = Ks  # [C, 3, 3]
        self.n_cameras = viewmats.shape[0]

    @classmethod
    def from_scene(
        cls,
        means: Tensor,
        n_cameras: int = 12,
        height: int = 512,
        width: int = 512,
        fov_deg: float = 60.0,
        radius_scale: float = 0.3,
        device: str = "cuda:0",
    ) -> "SyntheticCameras":
        """Generate cameras on an orbit sphere from scene means.

        Args:
            means: [N, 3] Gaussian positions (any frame).
            n_cameras: Number of cameras to generate.
            height: Image height in pixels.
            width: Image width in pixels.
            fov_deg: Horizontal field of view in degrees.
            radius_scale: Orbit radius as multiple of scene extent.
            device: Torch device.

        Returns:
            SyntheticCameras instance with viewmats and Ks.
        """
        centroid = means.mean(dim=0)  # [3]
        extent = (means.max(dim=0).values - means.min(dim=0).values).norm()
        radius = extent * radius_scale

        # Golden spiral points on sphere
        positions = _golden_spiral_sphere(n_cameras, radius, centroid, device)

        # Build look-at view matrices
        viewmats = torch.zeros(n_cameras, 4, 4, device=device)
        for i in range(n_cameras):
            viewmats[i] = _look_at(positions[i], centroid, device)

        # Intrinsics from FOV
        fov_rad = fov_deg * math.pi / 180.0
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # square pixels

        Ks = torch.zeros(n_cameras, 3, 3, device=device)
        Ks[:, 0, 0] = fx
        Ks[:, 1, 1] = fy
        Ks[:, 0, 2] = width / 2.0
        Ks[:, 1, 2] = height / 2.0
        Ks[:, 2, 2] = 1.0

        return cls(viewmats=viewmats, Ks=Ks)

    def sample(self, n: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Sample n camera indices randomly.

        Returns:
            Tuple of (indices [n], viewmats [n, 4, 4], Ks [n, 3, 3]).
        """
        indices = torch.randperm(self.n_cameras, generator=generator)[:n]
        return indices, self.viewmats[indices], self.Ks[indices]

    def get(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """Get cameras by index.

        Returns:
            Tuple of (viewmats [n, 4, 4], Ks [n, 3, 3]).
        """
        return self.viewmats[indices], self.Ks[indices]


def _golden_spiral_sphere(
    n: int,
    radius: float,
    center: Tensor,
    device: str,
) -> Tensor:
    """Distribute n points on a sphere using golden spiral.

    Args:
        n: Number of points.
        radius: Sphere radius.
        center: [3] center position.
        device: Torch device.

    Returns:
        [n, 3] positions on the sphere.
    """
    golden_ratio = (1 + math.sqrt(5)) / 2
    points = torch.zeros(n, 3, device=device)

    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        points[i, 0] = radius * math.sin(theta) * math.cos(phi) + center[0]
        points[i, 1] = radius * math.sin(theta) * math.sin(phi) + center[1]
        points[i, 2] = radius * math.cos(theta) + center[2]

    return points


def _look_at(eye: Tensor, target: Tensor, device: str) -> Tensor:
    """Build a world-to-camera 4x4 matrix (OpenCV convention: +Z forward).

    Args:
        eye: [3] camera position.
        target: [3] look-at target.
        device: Torch device.

    Returns:
        [4, 4] view matrix.
    """
    forward = target - eye
    forward = forward / forward.norm()

    # Use world-up = Y unless forward is nearly parallel to Y
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if torch.abs(forward.dot(world_up)) > 0.99:
        world_up = torch.tensor([0.0, 0.0, 1.0], device=device)

    right = torch.linalg.cross(forward, world_up)
    right = right / right.norm()
    up = torch.linalg.cross(right, forward)

    # OpenCV: camera looks down +Z, Y points down, X points right
    mat = torch.eye(4, device=device)
    mat[0, :3] = right
    mat[1, :3] = -up
    mat[2, :3] = forward
    mat[:3, 3] = mat[:3, :3] @ (-eye)

    return mat
