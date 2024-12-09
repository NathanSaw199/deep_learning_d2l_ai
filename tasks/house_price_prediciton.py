import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import inspect
# STEP 1 : DOWNLOAD THE DATASETs 
    #(kaggle_house_pred_train.csv and kaggle_house_pred_test.csv)

# STEP 2 : DATA PREPROCESSING
    #pd.concat(...) combines the training and validation/test features into a single features DataFrame which is one dataset after removing the id column, which is irrelevant for predictions.Removes the SalePrice column, as it's the target variable and not part of the features
    # start with the numerical features. First, we apply a heuristic, replacing all missing values by the corresponding feature’s mean. Then, to put all features on a common scale, we standardize the data by rescaling features to zero mean and unit variance: x <-(x-u)/sigma, where u is the mean and sigma is the standard deviation. To verify that this indeed transforms our data, note that after standardization the mean is 0 and the variance is 1.
    # discrete values These include features such as “MSZoning”. We replace them by a one-hot encoding  transformed multiclass labels into vectors. For instance, “MSZoning” assumes the values “RL” and “RM”. Dropping the “MSZoning” feature, two new indicator features “MSZoning_RL” and “MSZoning_RM” are created with values being either 0 or 1. According to one-hot encoding, if the original value of “MSZoning” is “RL”, then “MSZoning_RL” is 1 and “MSZoning_RM” is 0


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
# first four and final two features as well as the label (SalePrice) from the first five examples.
print(data.raw_train.iloc[:5,[0,1,2,3,-3,-2,-1]])

@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    #label = 'SalePrice' indicates that the target variable for predictions is SalePrice.
    label = 'SalePrice'
    #pd.concat(...) combines the training and validation/test features into a single features DataFrame which is one dataset.
    features = pd.concat(
        #Removes the id column, which is irrelevant for predictions.Removes the SalePrice column, as it's the target variable and not part of the features.In supervised learning, the target variable (SalePrice) is the value the model is trained to predict. It is not a feature of the dataset.
        (self.raw_train.drop(columns =['Id',label]),
         #Removes only the id column from the validation/test set since SalePrice is not present there
         self.raw_val.drop(columns = ['Id']))
         )
    #Standardize numerical columns

    #features.dtypes returns the data type (dtype) of each column in the features DataFrame.features.dtypes != 'object' creates a Boolean mask that identifies columns whose data type is not 'object'.'object' typically represents categorical or text data in pandas..index Retrieves the column names (index) corresponding to the True values in the Boolean mask.These column names are stored in the variable numeric_features.The index is the column names of the features DataFrame
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    #features[numeric_features] Selects only the numeric features from the features DataFrame, as identified by the numeric_features variable.apply(lambda x: ...) Applies the provided lambda function to each column in the selected numeric features.x - x.mean(): Subtracts the mean of the column from each value./ x.std(): Divides by the column's standard deviation.This transforms the values to have A mean of 0.A standard deviation of 1.
    features[numeric_features] = features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0 is handling missing values in the numeric features of the features DataFrame by replacing them with 0
    features[numeric_features] =features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features,dummy_na=True)   
    #save preporcessed features
    #features[:self.raw_train.shape[0]] Slices the first self.raw_train.shape[0] rows from the preprocessed features DataFrame.This corresponds to the rows originally from the training set (raw_train).copy() Creates a copy of the sliced DataFrame to avoid unintended changes to the original features DataFrame.self.train Stores the preprocessed features corresponding to the training dataset.
    self.train = features[:self.raw_train.shape[0]].copy()
    #self.raw_train[label] Retrieves the target column (SalePrice) from the original training dataset (raw_train).self.train[label] adds the target column (SalePrice) back to the preprocessed training DataFrame (self.train).The target variable (SalePrice) was excluded during preprocessing but needs to be included in the final training dataset for model training.
    self.train[label]= self.raw_train[label]
    #features[self.raw_train.shape[0]:]:Selects the rows from features starting at self.raw_train.shape[0] (the end of the training data) to the end of the DataFrame. These rows correspond to the validation/test set (raw_val).Stores the preprocessed features corresponding to the validation/test dataset.
    self.val =features[self.raw_train.shape[0]:].copy()

data.preprocess()
print(data.train.shape)