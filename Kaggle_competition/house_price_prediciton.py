import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import inspect
from IPython import display
import collections

# STEP 1 : DOWNLOAD THE DATASETs 
    #(kaggle_house_pred_train.csv and kaggle_house_pred_test.csv)

# STEP 2 : DATA PREPROCESSING
    #pd.concat(...) combines the training and validation/test features into a single features DataFrame which is one dataset after removing the id column, which is irrelevant for predictions.Removes the SalePrice column, as it's the target variable and not part of the features
    # start with the numerical features. First, we apply a heuristic, replacing all missing values by the corresponding feature’s mean. Then, to put all features on a common scale, we standardize the data by rescaling features to zero mean and unit variance: x <-(x-u)/sigma, where u is the mean and sigma is the standard deviation. To verify that this indeed transforms our data, note that after standardization the mean is 0 and the variance is 1.
    # discrete values These include features such as “MSZoning”. We replace them by a one-hot encoding  transformed multiclass labels into vectors. For instance, “MSZoning” assumes the values “RL” and “RM”. Dropping the “MSZoning” feature, two new indicator features “MSZoning_RL” and “MSZoning_RM” are created with values being either 0 or 1. According to one-hot encoding, if the original value of “MSZoning” is “RL”, then “MSZoning_RL” is 1 and “MSZoning_RM” is 0

# STEP 3 : Error Measures
    # about relative quantities more than absolute quantities, about the relative error y-y^/y than absolute error y-y^. Splitting into features (X) and target (Y).Converting them into PyTorch tensors.Creating a DataLoader for training or validation.The target variable (SalePrice) is log-transformed, which is a common practice in regression tasks with skewed target distributions.The method works for both training and validation by selecting the appropriate dataset (self.train or self.val) based on the train flag.Returns a PyTorch DataLoader containing batches of features (X) and log-transformed targets (Y).

#STEP 4: build linear regression model

#STEP 5: K-fold cross-validation
    #The training data is split into k equal-sized subsets (folds).
    #Create Training and Validation Sets. For each fold, the current subset (idx) is used as the validation set.The remaining subsets (data.train.drop(index=idx)) are combined to form the training set.Store Results:Each training-validation split is stored as a KaggleHouse instance in the rets list. Example Input: Training data with 1000 samples. k = 5 (5-fold cross-validation). Split the data into 5 folds, each containing 200 samples.For each fold: Use 800 samples for training.Use the remaining 200 samples for validation.Output:A list of 5 KaggleHouse instances, each containing a unique train-validation split.
#STEP 6: train the model with trainer

#STEP 7:  Model Selection
    #The average validation error is returned when we train k times in the k-fold cross-validation.

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


class Trainer(HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)


    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()


    def fit_epoch(self):
        raise NotImplementedError

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1


    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
    

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch

    

    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model


    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm


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

class ProgressBoard(d2l.HyperParameters):
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        d2l.use_svg_display()
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class Module(nn.Module,HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError


    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)


    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))


    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l


    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)


    def configure_optimizers(self):
        raise NotImplementedError

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return torch.optim.SGD(self.parameters(), lr=self.lr)


    def apply_init(self, inputs, init=None):
        """Defined in :numref:`sec_lazy_init`"""
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

class linear_regression_model(Module):
    def __init__(self,num_inputs,lr,sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = nn.Parameter(torch.normal(0,sigma,(num_inputs,1),requires_grad=True))
        self.b = nn.Parameter(torch.zeros(1,requires_grad=True))
        self.lr = lr

@d2l.add_to_class(linear_regression_model)
def forward(self,X):
    return torch.matmul(X,self.w)+self.b
@d2l.add_to_class(linear_regression_model)
def loss(self,y_hat,y):
    l =((y_hat-y)**2)/2
    return l.mean()

class SGD(HyperParameters):
    def __init__(self,params,lr):
        self.save_hyperparameters()
    
    def step(self):
        for param in self.params:
            param -=self.lr*param.grad
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero()

d2l.add_to_class(linear_regression_model)
def configure_optimizers(self):
    return SGD([self.w,self.b],self.lr)
       

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
# print(data.train.shape)

#defines the get_dataloader method for the KaggleHouse class. It is used to prepare the data (features and labels) and convert it into a PyTorch DataLoader for training or validation.
@d2l.add_to_class(KaggleHouse)
#takes a parameter train, which determines whether to return the training or validation data.
def get_dataloader(self,train):
    #label = 'SalePrice'Specifies the target variable (SalePrice) that the model will predict.
    label = 'SalePrice'
    #data = self.train if train else self.val Selects the training data (self.train) if train=True, otherwise selects the validation/test data (self.val).
    data = self.train if train else self.val
    # If the SalePrice column is not in the selected data (data), the method returns None.This ensures the method doesn't proceed when the dataset lacks the target variable
    if label not in data: return
    #get_tensor A lambda function that converts a pandas DataFrame or Series (x) into a PyTorch tensor.
    # x.values.astype(float) Extracts the underlying numpy array and ensures the values are of type float. torch.tensor(..., dtype=torch.float32): Converts the numpy array into a PyTorch tensor of type float32.
    get_tensor = lambda x: torch.tensor(x.values.astype(float),dtype=torch.float32)
    #data.drop(columns=[label])Removes the SalePrice column, leaving only the features (X).
    # get_tensor(data.drop(columns=[label]))Converts the features into a PyTorch tensor.
    # torch.log(get_tensor(data[label])) Applies a logarithmic transformation to the target variable (SalePrice) to Reduce the effect of outliers.The transformed target (Y) is reshaped into a 2D tensor with one column using .reshape((-1, 1)).
    # tensors = tuple containing X (features tensor).Y (log-transformed target tensor).
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               torch.log(get_tensor(data[label])).reshape((-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)

#The K_fold_data function is designed to generate K-folds for cross-validation in the context of training and evaluating a machine learning model
#data is An instance of the KaggleHouse class containing the dataset and related parameters.k is The number of folds to divide the data into for cross-validation.
def K_fold_data(data,k):
    #initializes an empty list to store the folds.
    rets =[]
    #Calculates the number of samples in each fold by dividing the total number of rows in the training data by the number of folds (k)
    fold_size = data.train.shape[0] // k
    #for j in range(k) Iterates over each fold index from 0 to k-1.
    for j in range(k):
        #Defines the indices for the current fold
        #If fold_size = 100 and j = 0, idx = range(0, 100).If j = 1, idx = range(100, 200), and so on.
        idx = range(j*fold_size,(j+1)*fold_size)
        #data.train.drop(index=idx) Removes the rows corresponding to the current fold indices (idx) to create the training subset for this fold.data.train.loc[idx] Selects the rows corresponding to the current fold indices (idx) to create the validation subset for this fold.
        rets.append(KaggleHouse(data.batch_size,data.train.drop(index=idx),data.train.loc[idx]))
    #rets.append(...):passing:data.batch_size: The batch size for data loading.data.train.drop(index=idx): The training subset for this fold.data.train.loc[idx]: The validation subset for this fold.
    #Returns a list of KaggleHouse instances, where each instance represents one fold of the K-fold cross-validation.
    return rets

#The average validation error is returned when we train k times in the k-fold cross-validation.
#trainer: An instance of a training utility that manages the training process.data: An instance of the KaggleHouse class containing the dataset.k: Number of folds for K-fold cross-validation.lr: Learning rate for training the model.
def k_fold(trainer, data, k, lr):
    #val_loss: Stores the validation loss (log MSE) for each fold.models: Stores the trained model for each fold.
    val_loss, models = [], []
    #K_fold_data(data, k) generates k train-validation splits.
    #data_fold is a KaggleHouse instance for the current fold containing:Training data (data_fold.train).Validation data (data_fold.val).
    for i, data_fold in enumerate(K_fold_data(data, k)):
        #A new instance of the LinearRegression model is created with the specified learning rate (lr).
        num_inputs = data_fold.train.shape[1] - 1  # Exclude target column
        model = linear_regression_model(num_inputs, lr)  # Pass num_inputs
        #model.board.yscale = 'log' sets the y-axis of the loss plot to a logarithmic scale.
        model.board.yscale='log'
        #if i != 0: model.board.display = False ensures only the first fold's training progress is displayed.
        if i != 0: model.board.display = False
        #The model is trained on the current fold using the trainer.fit method, which: Trains the model on data_fold.train.Validates the model on data_fold.val.
        trainer.fit(model, data_fold)
        #After training, the validation loss (log mean squared error) of the last epoch is retrieved and stored in val_loss.
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        #The trained model for the current fold is stored in the models list.
        models.append(model)
        #Computes and prints the average validation loss across all folds.
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    #Returns a list of trained models, one for each fold.
    return models
trainer = Trainer(max_epochs=20)
models = k_fold(trainer, data, k=5, lr=0.01)