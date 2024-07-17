import torch.utils.data
from data.base_dataset import collate_fn
# for sampler
from data.constrained_grasp_sampling_data import ConstrainedGraspSamplingData
# for evaluator
from data.grasp_evaluator_data import GraspEvaluatorData


def CreateDataset(opt):
    """loads dataset class"""
    if opt.arch == 'sampler':
        return ConstrainedGraspSamplingData(opt)
    elif opt.arch == 'evaluator':
        return GraspEvaluatorData(opt)
    else:
        raise NameError("There is no architecture name. Check CreateDataset in data/__init__.py")


class VCGS_DataLoader:
    """multi-thread data loading"""

    def __init__(self, opt):
        self.opt = opt
        # self.create_dataset()
        self.dataset = CreateDataset(self.opt)
        self.train_set = None
        self.valid_set = None
        self.test_set = None


    def split_dataset(self, split_size_percentage=[0.8, 0.15, 0.05]):
        """
        split_size_percentage: [training, validation, test]
        """
        dataset_size = len(self.dataset)
        number_of_training_samples = round(split_size_percentage[0] * dataset_size)
        number_of_validation_samples = round(split_size_percentage[1] * dataset_size)
        number_of_test_samples = dataset_size - number_of_training_samples - number_of_validation_samples
        self.train_set, self.valid_set, self.test_set = torch.utils.data.random_split(self.dataset, [number_of_training_samples, number_of_validation_samples, number_of_test_samples])
        print(f"Training set: {len(self.train_set)} samples")
        print(f"Validation set: {len(self.valid_set)} samples")
        print(f"Test set: {len(self.test_set)} samples")
        return self.train_set, self.valid_set, self.test_set

    def create_dataloader(self, dataset, shuffle_batches):
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=self.opt.num_objects_per_batch,
                                                      shuffle=shuffle_batches,
                                                      num_workers = int(self.opt.num_threads),
                                                      collate_fn=collate_fn)
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data