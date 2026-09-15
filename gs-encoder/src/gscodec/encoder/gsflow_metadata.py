"""
GSFlow metadata parser for chunk boundaries and ranges.

GSFlow generates metadata.json files that contain:
- Chunk boundaries (which frames go in which chunk)
- Per-chunk quantization ranges (more precise than global ranges)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class ChunkRanges:
    """Per-chunk quantization ranges from GSFlow."""

    means_min: NDArray[np.float32]  # [3]
    means_max: NDArray[np.float32]  # [3]
    scales_min: NDArray[np.float32]  # [3]
    scales_max: NDArray[np.float32]  # [3]
    opacities_min: float
    opacities_max: float
    quats_min: float
    quats_max: float
    sh0_min: float
    sh0_max: float


@dataclass
class ChunkInfo:
    """Chunk metadata from GSFlow."""

    chunk_id: int
    start_frame: int
    end_frame: int  # Inclusive
    n_frames: int
    gaussian_count: int
    ranges: ChunkRanges


@dataclass
class GSFlowMetadata:
    """GSFlow metadata container."""

    version: str
    chunk_size: int
    total_frames: int
    total_chunks: int
    chunks: list[ChunkInfo]

    @classmethod
    def load(cls, path: str | Path) -> "GSFlowMetadata":
        """Load GSFlow metadata from JSON file.

        Args:
            path: Path to metadata.json file.

        Returns:
            Parsed GSFlowMetadata.
        """
        with open(path) as f:
            data = json.load(f)

        chunks = []
        for chunk_data in data["chunks"]:
            ranges_data = chunk_data["ranges"]

            # Handle both array and scalar formats for ranges
            def to_array(val: list | float, size: int) -> NDArray[np.float32]:
                if isinstance(val, list):
                    return np.array(val, dtype=np.float32)
                return np.full(size, val, dtype=np.float32)

            ranges = ChunkRanges(
                means_min=to_array(ranges_data["means_min"], 3),
                means_max=to_array(ranges_data["means_max"], 3),
                scales_min=to_array(ranges_data["scales_min"], 3),
                scales_max=to_array(ranges_data["scales_max"], 3),
                opacities_min=float(ranges_data["opacities_min"]),
                opacities_max=float(ranges_data["opacities_max"]),
                quats_min=float(ranges_data["quats_min"]),
                quats_max=float(ranges_data["quats_max"]),
                sh0_min=float(ranges_data["sh0_min"]),
                sh0_max=float(ranges_data["sh0_max"]),
            )

            chunks.append(
                ChunkInfo(
                    chunk_id=chunk_data["chunk_id"],
                    start_frame=chunk_data["start_frame"],
                    end_frame=chunk_data["end_frame"],
                    n_frames=chunk_data["n_frames"],
                    gaussian_count=chunk_data["gaussian_count"],
                    ranges=ranges,
                )
            )

        return cls(
            version=data.get("version", "1.0"),
            chunk_size=data.get("chunk_size", chunks[0].n_frames if chunks else 30),
            total_frames=data["total_frames"],
            total_chunks=data["total_chunks"],
            chunks=chunks,
        )
