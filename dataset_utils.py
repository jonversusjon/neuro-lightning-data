from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class PartitionConfig:
    """Configuration data for partitioning a dataset."""
    val_split: float = 0.2  # Validation split ratio
    test_split: float = 0.1  # Test split ratio
    save_parts: bool = False  # Save the partitioned data to disk
    use_saved_parts: bool = False  # Use previously saved partitions
    parts_filepath: str = ''  # File path for saved partitions
    print_stats_parts: bool = False  # Print statistics for each partition
    checksum_parts: List[str] = field(default_factory=list)  # Checksum for validating partitions

    def __post_init__(self):
        """Perform validation after initialization."""
        if self.val_split + self.test_split >= 1.0:
            raise ValueError("The sum of val_split and test_split should be less than 1.")

        if self.use_saved_parts:
            if not os.path.exists(self.parts_filepath):
                raise FileNotFoundError(f"Partitions file not found: {self.parts_filepath}")

    def to_dict(self):
        return {
            'val_split': self.val_split,
            'test_split': self.test_split,
            'save_parts': self.save_parts,
            'use_saved_parts': self.use_saved_parts,
            'parts_filepath': self.parts_filepath,
            'print_stats_parts': self.print_stats_parts,
            'checksum_parts': self.checksum_parts
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            val_split=d['val_split'],
            test_split=d['test_split'],
            save_parts=d['save_parts'],
            use_saved_parts=d['use_saved_parts'],
            parts_filepath=d['parts_filepath'],
            print_stats_parts=d['print_stats_parts'],
            checksum_parts=d['checksum_parts']
        )
