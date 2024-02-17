from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class DataLoaderConfig:
    batch_size: int = 8
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False

    def to_dict(self):
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'pin_memory': self.pin_memory,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            batch_size=d.get('batch_size', 8),
            num_workers=d.get('num_workers', 0),
            drop_last=d.get('drop_last', False),
            pin_memory=d.get('pin_memory', False)
        )
        
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

class ChecksumManager:
    @staticmethod
    def compute_hash(data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy().tobytes()
        elif isinstance(data, bytes):
            pass  # data is already bytes
        else:
            raise ValueError("Unsupported type for hashing")
        return md5(data).hexdigest()


    @staticmethod
    def compute_checksum(dataset, indices, tag, transformed_data=False):
        concatenated_checksums = ""
        return ""
        # Concatenate the checksums of each item in the dataset for the provided indices
        for i in tqdm(indices, desc=f"Calculating checksums for {tag}"):
            item = dataset[i]
            if transformed_data:
                item_checksum = item['transformed_hash']
            else:
                item_checksum = item['raw_hash']

            concatenated_checksums += item_checksum

        # Compute and return the final checksum for the concatenated string
        final_checksum = hashlib.new('sha256')
        final_checksum.update(concatenated_checksums.encode('utf-8'))

        return final_checksum.hexdigest()
