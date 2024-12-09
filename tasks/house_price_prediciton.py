import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import inspect
# STEP 1 : DOWNLOAD THE DATASETs (kaggle_house_pred_train.csv and kaggle_house_pred_test.csv)

# STEP 2 : data preprocessing


class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


class DataModule(HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError


    def train_dataloader(self):
        return self.get_dataloader(train=True)


    def val_dataloader(self):
        return self.get_dataloader(train=False)


    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)


class KaggleHouse(DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            #raw_train is a pandas DataFrame containing the training dataset. and download the training dataset from the D2L data server.
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            #raw_val is a pandas DataFrame containing the validation dataset. and download the validation dataset from the D2L data server
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
data = KaggleHouse(batch_size=64)
#The training dataset (raw_train) contains 1460 rows and 81 columns.
#The 81 columns likely include features (e.g., house characteristics) and the target variable (house price). (1460, 81)
print(data.raw_train.shape)
#The validation dataset (raw_val) contains 1459 rows and 80 columns.
#This dataset is typically used for testing or prediction purposes and doesn't include the target variable (house price). Hence, it has one fewer column than the training dataset.(1459, 80)
print(data.raw_val.shape)
print(data.raw_train.iloc[:5],[0,1,2,3,-3,-2,-1])