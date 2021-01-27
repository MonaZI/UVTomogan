import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, meas, args):
        """
        The class for the dataset.
        :param meas: The set of measurements
        :param args: a set of arguments
        """
        self.args = args
        self.meas = meas
        self.num_samples = self.meas.shape[0]
        print('The number of samples: %d' %(self.num_samples))

    def __getitem__(self, index):
        return self.meas[index, ...]

    def __len__(self):
        return self.num_samples


def collate_fn(data):
    """
    Converts the batched data in torch tensor and returns them
    :param data: the data in form of numpy array
    :return: the data in torch.tensor
    """
    return torch.tensor(data).float()


def get_loader(dataset, args, is_test=False):
    """
    Creates the dataloader from dataset based on the given arguments
    :param dataset: the dataset
    :param args: the required arguments
    :param is_test: whether it is the train or test data
    :return: the dataloader object
    """
    dataLoader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn,
                                 shuffle=True if (is_test==False) else False)
    return dataLoader
