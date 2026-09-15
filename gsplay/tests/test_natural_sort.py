"""Tests for natural sorting functionality."""

from pathlib import Path

from src.shared.math import natural_sort_key


class TestNaturalSortKey:
    """Test natural_sort_key function."""

    def test_numeric_sequence(self):
        """Test sorting of numeric sequences."""
        files = ["image_1.ply", "image_10.ply", "image_2.ply", "image_20.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ["image_1.ply", "image_2.ply", "image_10.ply", "image_20.ply"]

    def test_zero_padded_numbers(self):
        """Test sorting of zero-padded numbers."""
        files = ["frame_001.ply", "frame_100.ply", "frame_002.ply", "frame_010.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ["frame_001.ply", "frame_002.ply", "frame_010.ply", "frame_100.ply"]

    def test_mixed_formats(self):
        """Test sorting of mixed filename formats."""
        files = [
            "image_1.ply",
            "image_10.ply",
            "image_2.ply",
            "frame_001.ply",
            "frame_100.ply",
            "frame_002.ply",
        ]
        sorted_files = sorted(files, key=natural_sort_key)

        # Should group by prefix, then sort numerically
        assert sorted_files == [
            "frame_001.ply",
            "frame_002.ply",
            "frame_100.ply",
            "image_1.ply",
            "image_2.ply",
            "image_10.ply",
        ]

    def test_no_numbers(self):
        """Test sorting of filenames without numbers."""
        files = ["abc.ply", "xyz.ply", "def.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ["abc.ply", "def.ply", "xyz.ply"]

    def test_path_object(self):
        """Test sorting with Path objects."""
        files = [Path("image_1.ply"), Path("image_10.ply"), Path("image_2.ply")]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == [Path("image_1.ply"), Path("image_2.ply"), Path("image_10.ply")]

    def test_case_insensitive(self):
        """Test case-insensitive sorting."""
        files = ["Image_1.ply", "image_10.ply", "IMAGE_2.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        # Should sort case-insensitively
        assert sorted_files == ["Image_1.ply", "IMAGE_2.ply", "image_10.ply"]

    def test_multiple_number_segments(self):
        """Test sorting with multiple number segments."""
        files = ["v1_frame_10.ply", "v2_frame_1.ply", "v1_frame_2.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ["v1_frame_2.ply", "v1_frame_10.ply", "v2_frame_1.ply"]

    def test_large_numbers(self):
        """Test sorting with large numbers."""
        files = ["frame_1000000.ply", "frame_1.ply", "frame_100000.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ["frame_1.ply", "frame_100000.ply", "frame_1000000.ply"]

    def test_empty_string(self):
        """Test handling of empty strings."""
        files = ["", "frame_1.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        assert sorted_files == ["", "frame_1.ply"]

    def test_special_characters(self):
        """Test sorting with special characters."""
        files = ["frame-1.ply", "frame_10.ply", "frame-2.ply"]
        sorted_files = sorted(files, key=natural_sort_key)

        # Special characters should be treated separately
        assert "frame-1.ply" in sorted_files
        assert "frame-2.ply" in sorted_files
        assert "frame_10.ply" in sorted_files


class TestNaturalSortIntegration:
    """Integration tests for natural sorting in PLY loading."""

    def test_ply_model_uses_natural_sort(self, tmp_path, sample_gsdata):
        """Use real isolated files, not the user's export folder or Path mocks."""
        import gsply

        from src.models.ply.optimized_model import create_optimized_ply_model_from_folder

        for name in ("frame_10.ply", "frame_2.ply", "frame_1.ply"):
            gsply.plywrite(tmp_path / name, sample_gsdata)
        model = create_optimized_ply_model_from_folder(str(tmp_path), device="cpu")
        try:
            assert [path.name for path in model.ply_files] == [
                "frame_1.ply",
                "frame_2.ply",
                "frame_10.ply",
            ]
        finally:
            model.on_shutdown()
