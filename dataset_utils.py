from dataclasses import dataclass, field
from typing import List
import os
from torchvision import transforms as T
import torch
import hashlib
from hashlib import md5
import tqdm
import DataLoaderConfig
from prettytable import PrettyTable
import numpy


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

class TransformationManager:
    def __init__(self, basic_transforms, normal_transforms, dataloader_config: DataLoaderConfig,):
        self.data_dir = None
        self.basic_transforms = basic_transforms
        self.normal_transforms = normal_transforms
        self.loader_config = dataloader_config
        self.fluo_channels = None
        self.batch_size = dataloader_config["batch_size"]

        # Initialize tracking variables
        self.min_value = {}
        self.max_value = {}
        self.sum_values = {}
        self.count_values = {}
        self.current_mean_train = None
        self.current_std_train = None
        self.current_mean_val = None
        self.current_std_val = None
        self.sum_square_values = defaultdict(float)

    def set_dataset(self, dataset: CustomImageDataset):
        # self.means, self.stds = self.check_and_update_normalize_params(self.t_config, dataset)
        self.fluo_channels = dataset.fluo_channels

    def print_stats_table(self):
        decimal_places = 4
        x = PrettyTable()
        x.align = "r"
        x.field_names = ["Statistic"] + [f"Channel {i}" for i in self.min_value.keys()]

        # Add rows for each statistic with float values rounded to 4 decimal places
        x.add_row(["Min"] + [round(self.min_value[i], decimal_places) for i in self.min_value.keys()])
        x.add_row(["Max"] + [round(self.max_value[i], decimal_places) for i in self.min_value.keys()])
        x.add_row(["Mean"] + [round(self.sum_values[i] / self.count_values[i], decimal_places) for i in self.min_value.keys()])
        x.add_row(["Std"] + [round(np.sqrt(self.sum_square_values[i] / self.count_values[i] -
                                        (self.sum_values[i] / self.count_values[i]) ** 2), decimal_places) for i in self.min_value.keys()])

        print(x)


    def update_stats(self, batch_tensor):
        # Assume tensor shape is [Batch, Channels, Height, Width]
        for i in range(batch_tensor.shape[0]):  # Loop over batch dimension
            tensor = batch_tensor[i]
            for channel_idx in range(tensor.shape[0]):  # Loop over channels
                channel_data = tensor[channel_idx, :, :]

                # Update Min and Max
                self.min_value[channel_idx] = min(torch.min(channel_data).item(), self.min_value.get(channel_idx, float('inf')))
                self.max_value[channel_idx] = max(torch.max(channel_data).item(), self.max_value.get(channel_idx, float('-inf')))

                # Update Sum and Count for Mean
                self.sum_values[channel_idx] = self.sum_values.get(channel_idx, 0) + torch.sum(channel_data).item()
                self.count_values[channel_idx] = self.count_values.get(channel_idx, 0) + channel_data.numel()

                # Update Sum Squares for Std Dev
                self.sum_square_values[channel_idx] = self.sum_square_values.get(channel_idx, 0) + torch.sum(channel_data ** 2).item()


    def get_transforms(self, data_dir=None):
        print(f'initializing transforms...')
        if data_dir is None:
            data_dir = self.data_dir
        else:
            self.data_dir = data_dir

        transform_list = []
        invert_transform_list = []

        dataset_min = float('inf')
        dataset_max = -float('inf')

        # Add basic transforms
        for b_transform in self.basic_transforms:
            params_copy = b_transform['params'].copy()  # Create a copy of params to possibly update

            if b_transform['operation'] == 'MinMaxScaler':
                partial_b_transforms = T.Compose(transform_list)
                print(f'partial_b_transforms: {partial_b_transforms}')

                temp_dataset = CustomImageDataset(data_dir, transforms=partial_b_transforms)
                temp_loader = DataLoader(temp_dataset, batch_size=self.batch_size, shuffle=False)

                for batch in temp_loader:
                    min_val, max_val = torch.min(batch['images']), torch.max(batch['images'])

                    dataset_min = min(dataset_min, min_val.item())
                    dataset_max = max(dataset_max, max_val.item())

                print(f'MinMaxScaler => dataset_min: {dataset_min}, dataset_max: {dataset_max}')
                params_copy['dataset_min'] = dataset_min
                params_copy['dataset_max'] = dataset_max

            self.add_transform(transform_list, invert_transform_list, b_transform['operation'], params_copy)


        # Add normal transforms
        print(f'self.normal_transforms: {self.normal_transforms}')
        for n_transform in self.normal_transforms:
            print(f'n_transform: {n_transform}')
            temp_dataset = None
            params = n_transform.get('params', {'mean': [], 'std': []})
            operation = n_transform['operation']

            if len(params['mean']) == 0 or len(params['std']) == 0:
                # If user does not provide mean/std, compute them
                partial_transform = T.Compose(transform_list)

                if temp_dataset is None:
                    temp_dataset = CustomImageDataset(data_dir, transforms=partial_transform)

                num_channels = temp_dataset.fluo_channels
                temp_loader = DataLoader(temp_dataset, batch_size=self.batch_size, shuffle=False)

                stats = compute_dataset_stats(temp_loader, num_channels=num_channels, stats_to_compute=['mean', 'std'])
                print("Debugging: stats =", stats)

                print(f"stats[mean]: {stats['mean']}, stats[std]: {stats['std']}")
                params['mean'], params['std'] = stats['mean'], stats['std']

            self.add_transform(transform_list, invert_transform_list, operation, params)

        # Reset tracking variables
        self.min_value = {}
        self.max_value = {}
        self.sum_values = {}
        self.count_values = {}
        self.current_mean_train = None
        self.current_std_train = None
        self.current_mean_val = None
        self.current_std_val = None
        self.sum_square_values = defaultdict(float)

        final_transform = T.Compose(transform_list)
        final_dataset = CustomImageDataset(data_dir, transforms=final_transform)
        final_loader = DataLoader(final_dataset, batch_size=self.batch_size, shuffle=False)

        for batch in final_loader:
            self.update_stats(batch['images'])
        print(' ' * 2, 'Transformed Dataset Descriptive Stats', ' ' * 2)
        self.print_stats_table()

        invert_transform_list_reversed = invert_transform_list[::-1]

        print('-'*80)
        print(' '*28, 'transforms initialized!', ' '*28)
        print('-'*80)
        print('transform_list:')
        for transform in transform_list:
            print(' '*8, transform)
        print('invert_transform_list:')
        for transform in invert_transform_list_reversed:
            print(' '*8, transform)
        print('-'*80)

        return T.Compose(transform_list), T.Compose(invert_transform_list_reversed)


    def add_transform(self, transform_list, invert_transform_list, operation, params):
        operation_found = False
        print(f'add_transform operation: {operation}, params: {params}')
        if hasattr(tv.transforms, operation):
            transform_obj = getattr(tv.transforms, operation)(**params)
            transform_list.append(transform_obj)
            operation_found = True
        elif operation in custom_transforms:
            transform_class = custom_transforms[operation]

            # Create transform object
            transform_obj = transform_class(**params)

            # Add the transform to the main transform list
            transform_list.append(transform_obj)

            # Check if the class has a 'get_inverse' method
            if hasattr(transform_class, 'get_inverse'):
                inverse_transform = transform_obj.get_inverse()
                print(f'inverse_transform: {inverse_transform} for operation: {operation}')
                invert_transform_list.append(inverse_transform)

            operation_found = True

        if not operation_found:
            raise ValueError(f"Invalid transform operation: {operation}")


class PartitionManager:
    """
    Records train, test, val partitions for resuming training
    Uses md5 hash and sha256 checksum to verify data when resuming training on a previously
    trained model
    """
    def __init__(self, dataset: CustomImageDataset,
                 config: any,
                 stage: Optional[str] = None,
                 ):

        assert not (config["save_parts"] or config["use_saved_parts"]) or config["parts_filepath"] != '', \
            "If save_parts or use_saved_parts is set to True, parts_filepath must not be empty."

        self.val_split = config["val_split"]
        self.test_split = config["test_split"]
        self.parts_filepath = config["load_dir"]
        self.save_parts = config["save_parts"]
        self.checksum_parts = config["checksum_parts"]

        print(f'config.use_saved_parts: {config["use_saved_parts"]}')
        self.use_saved_parts = config["use_saved_parts"]
        self.print_stats_parts = config["print_stats_parts"]
        self.fluo_channels = dataset.fluo_channels
        self.checksummer = ChecksumManager()
        self.checksums = {
            'dataset': None,
            'train': None,
            'val': None,
            'test': None
        }

        self.partitioned_data = self.get_partitions(dataset)
        print(f'Partition Manager self.fluo_channels: {self.fluo_channels}')

        if stage == 'fit' or stage is None:
            self.data_train = self.partitioned_data['train']['data']

            if self.print_stats_parts:
                dataloader = DataLoader(self.data_train, batch_size=self.batch_size,
                                        num_workers=self.num_workers, shuffle=True,
                                        drop_last=True, pin_memory=True)
                stats = self.compute_dataset_stats(dataloader, self.fluo_channels)
                min, max, mean, std = stats['min'], stats['max'], stats['mean'], stats['std']
                print(f'Training data min: {min}, max: {max}, mean: {mean}, std: {std}')

        if stage in {'fit', 'validate'} or stage is None:
            self.data_val = self.partitioned_data['val']['data']

            if self.print_stats_parts:
                dataloader = DataLoader(self.data_val, batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        drop_last=True, pin_memory=True)
                stats = self.compute_dataset_stats(dataloader, self.fluo_channels)
                min, max, mean, std = stats['min'], stats['max'], stats['mean'], stats['std']
                print(f'Training data min: {min}, max: {max}, mean: {mean}, std: {std}')

        if stage in {'test', 'predict'} or stage is None:
            self.data_test = self.partitioned_data['test']['data']

            if self.print_stats_parts:
                dataloader = DataLoader(self.data_test, batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        drop_last=True, pin_memory=True)
                stats = self.compute_dataset_stats(dataloader, self.fluo_channels)
                min, max, mean, std = stats['min'], stats['max'], stats['mean'], stats['std']
                print(f'Training data min: {min}, max: {max}, mean: {mean}, std: {std}')


    def get_partitions(self, dataset):
        if self.use_saved_parts:
            print(f'use_saved_parts set to True')
            partitioned_data = self.load_existing_partitions(dataset)
        else:
            print(f'use_saved_parts set to False')
            partitioned_data = self.create_new_partitions(dataset)

        if self.save_parts:
            self.save_partitions(dataset, partitioned_data)

        return partitioned_data


    def load_existing_partitions(self, dataset):
        if os.path.exists(self.parts_filepath):
            print(f'Partition file found, checking if datasets match')
            with open(self.parts_filepath, 'r') as f:
                self.partition = json.load(f)
        else:
            raise FileNotFoundError(f"Partition file not found: {self.parts_filepath}")

        indices = list(range(len(dataset)))
        full_dataset_checksum = self.checksummer.compute_checksum(dataset, indices, 'dataset')

        if self.verify_checksums(full_dataset_checksum, self.partition["checksum"], 'dataset'):

            print(f'Datasets match! Using saved train, val, and test partitions')

            data_train = Subset(dataset, self.partition['train']['indices'])
            data_val = Subset(dataset, self.partition['val']['indices'])
            data_test = Subset(dataset, self.partition['test']['indices'])

            partitioned_data = {
                "train": data_train,
                "val": data_val,
                "test": data_test,
            }

            if len(self.checksum_parts) > 0:
                for part in self.checksum_parts:
                    self.verify_checksums(dataset, part, self.partitioned_data[part])

            return partitioned_data
        else:
            raise ValueError("Saved dataset checksum does not match current dataset checksum")


    def verify_checksums(self, current_checksum, saved_checksum, tag):
        if saved_checksum is None:
            raise ValueError(f"No saved checksum for {tag} found in partition file.")

        if saved_checksum != current_checksum:
            raise ValueError(f"Checksum for {tag} does not match saved checksum")
        else:
            return True


    def create_new_partitions(self, dataset):
        train_split = 1 - self.val_split - self.test_split
        train_size = int(train_split * len(dataset)) if train_split > 0 else 0

        val_size = int(self.val_split * len(dataset)) if self.val_split > 0 else 0
        test_size = len(dataset) - train_size - val_size

        data_train, data_val, data_test = random_split(dataset, [train_size, val_size, test_size])

        print(f"Train size: {train_size}, Val size: {val_size},"
                f" Test size: {test_size}")

        partitioned_data = {
            "train": {"data": data_train},
            "val": {"data": data_val},
            "test": {"data": data_test},
        }

        return partitioned_data


    def save_partitions(self, dataset, partitioned_data):
        # saving partitions automatically includes saving their checksums regardless
        #    of whether they are enabled or not
        print(f"partitioned_data[train][data]: {partitioned_data['train']['data']}")

        partitions = ['train', 'val', 'test']

        for part in partitions:
            if 'checksum_raw' not in partitioned_data[part]:
                # Compute the checksum if it's not already calculated
                checksum = self.checksummer.compute_checksum(dataset, partitioned_data[part]['data'].indices, part)

                # Add the computed checksum to the dictionary
                partitioned_data[part]['checksum_raw'] = checksum

        partitioned_data_serializable = {
            "dataset": {"indices": dataset.indices, "checksum_raw": dataset.get_checksum(transformed=False)},
            "train": {"indices": partitioned_data['train']['data'].indices, "checksum_raw": partitioned_data['train']['checksum_raw']},
            "val": {"indices": partitioned_data['val']['data'].indices, "checksum_raw": partitioned_data['val']['checksum_raw']},
            "test": {"indices": partitioned_data['test']['data'].indices, "checksum_raw": partitioned_data['test']['checksum_raw']}
        }

        output_path = os.makedirs(os.path.dirname(self.parts_filepath), exist_ok=True)

        try:
            print(f'Saving partitions to {self.parts_filepath}')
            with open(self.parts_filepath, 'w') as f:
                json.dump(partitioned_data_serializable, f)
            print("Partitions successfully saved.")
        except PermissionError:
            print("Permission denied: Partition Manager Unable to write to the specified path.")
        except FileNotFoundError:
            print(f"Partition Manager: File path not found: {output_path}")
        except TypeError:
            print("Partition Manager Failed to serialize the data to JSON. Object types might not be serializable.")
        except Exception as e:  # General catch-all for other exceptions
            print(f"An unexpected error occurred: {e}")

        return partitioned_data
