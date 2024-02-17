import pytorch_lightning as pl
from torch.utils.data import DataLoader
# Ensure the necessary imports for your CustomImageDataset, TransformationManager, and PartitionManager
from your_dataset_file import CustomImageDataset
from your_transformation_file import TransformationManager
from your_partition_manager_file import PartitionManager
# Assuming dataloader_config is a dict defined elsewhere, it should be imported or defined here
from your_config_file import dataloader_config, partitions_config

class DataModule(LightningDataModule):
    def __init__(self,
                 transformation_manager: TransformationManager,
                 data_dir: str = None,
                 data_output_dir: str = None,
                 run_num: str = "",
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.data_output_dir = data_output_dir
        self.run_num = run_num
        self.partition_manager = None

        self.transformation_manager = transformation_manager
        self.batch_size = dataloader_config["batch_size"]
        self.num_workers = dataloader_config["num_workers"]
        self.drop_last = dataloader_config["drop_last"]
        self.pin_memory = dataloader_config["pin_memory"]

        self.fluo_channels = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.inverse_transforms = None

    def setup(self, stage: str = None):

        assert self.data_dir, "No directories provided for data."

        print('-'*88)
        print(' '*42, 'DATA', ' '*42)
        print('-'*88)
        transforms, self.inverse_transforms = self.transformation_manager.get_transforms(self.data_dir)

        dataset = CustomImageDataset(self.data_dir, stage=stage, transforms=transforms, inverse_transforms=self.inverse_transforms)

        # Get the number of fluo channels
        self.fluo_channels = dataset[0]['images'].shape[0]
        dataset.fluo_channels = self.fluo_channels
        self.transformation_manager.set_dataset(dataset)

        self.partition_manager = PartitionManager(dataset, partitions_config)
        partitioned_data = self.partition_manager.partitioned_data

        self.data_train = partitioned_data['train']['data']
        self.data_val = partitioned_data['val']['data']
        self.data_test = partitioned_data['test']['data']

        print('-'*88)


    def load_full_dataset(self, transforms=None):

        full_dataset_raw = CustomImageDataset(self.data_dir, stage=None, transforms=transforms)

        # Get the number of fluo channels
        self.fluo_channels = full_dataset_raw[0]['images'].shape[0]
        full_dataset_raw.fluo_channels = self.fluo_channels

        print(f'Num of fluo channels: {self.fluo_channels}')

        return full_dataset_raw


    def train_dataloader(self):
        if self.data_train is not None:
          return DataLoader(self.data_train, batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True,
                            drop_last=self.drop_last,
                            pin_memory=self.pin_memory)


    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(self.data_val, batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              drop_last=self.drop_last,
                              pin_memory=self.pin_memory)


    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(self.data_test, batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              drop_last=self.drop_last,
                              pin_memory=self.pin_memory)
