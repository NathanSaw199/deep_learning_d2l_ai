import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import inspect
from IPython import display
import collections

 
#STEP 1 : The linear_regression_model class defines a basic linear regression model with parameters w and b.

#STEP 2 : A forward pass computes Xw + b

#STEP 3 : A mean squared error loss is defined to measure how well predictions match the targets

#STEP 4 : A custom SGD optimizer class updates these parameters using gradient descent.

#STEP 5 : configure_optimizers integrates the optimizer with the model, returning an SGD instance that the trainer can use.

#class HyperParameters: A base class to easily save function arguments as hyperparameters.
class HyperParameters:
    """The base class of hyperparameters."""

    #save_hyperparameters: A method that, when called inside an __init__ of a subclass, captures the arguments passed to that __init__ and saves them as attributes of self.
    #save_hyperparameters is an instance method of HyperParameters.It takes one optional argument: ignore, which is a list of parameter names that should not be saved.By default, ignore is an empty list, meaning that unless specified, no parameters are ignored.
    #This pattern greatly simplifies code in classes that need to remember all their initialization parameters, because you don’t have to manually write lines like self.lr = lr for every parameter in your __init__. Instead, you just call save_hyperparameters() once.

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

#class Trainer(HyperParameters): The Trainer class inherits from HyperParameters. By doing this, the trainer can easily capture and store its initialization parameters (like max_epochs and num_gpus) using the save_hyperparameters() method.
class Trainer(HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'
#prepare_data(self, data): This method sets up the training and validation data loaders from a data object (which should be a DataModule or similar class providing train_dataloader() and val_dataloader() methods).
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

#Attaches the trainer to the model: model.trainer = self. This allows the model to access trainer attributes, for example when logging training progress.
    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model
#fit(model, data) is the main training loop entry point.Calls prepare_data(data) to set up the dataloaders.Calls prepare_model(model) to prepare the model and attach the trainer to it.self.optim = model.configure_optimizers(): Retrieves the optimizer from the model. The model defines how it should be optimized, returning an optimizer object.    def fit(self, model, data):self.epoch = 0 to track the current epoch.self.train_batch_idx = 0 and self.val_batch_idx = 0 to track the number of batches processed.
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
#fit_epoch runs one full epoch of training and validation.self.model.train(): Sets the model to training mode (important for layers like dropout or batch normalization).
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
#class linear_regression_model(Module): Defines a model class that inherits from Module. Module is a base class defined elsewhere, likely a wrapper class around nn.Module and a HyperParameters class. This inheritance allows the model to have methods like forward, loss, and configuration for optimizers.
class linear_regression_model(Module):
    #def __init__(self, num_inputs, lr, sigma=0.01)::num_inputs: the number of input features for the linear regression (i.e., the dimension of X). lr: the learning rate for optimization.sigma=0.01: the standard deviation used for initializing the weights.
    def __init__(self,num_inputs,lr,sigma=0.01):
        #super().__init__(): Calls the constructor of the parent class Module to ensure it initializes properly.
        super().__init__()
        #self.save_hyperparameters(): A method from the HyperParameters class that automatically saves all arguments passed to __init__ (num_inputs, lr, sigma) as attributes of the instance. This avoids manually writing self.num_inputs = num_inputs, etc.
        self.save_hyperparameters()
        #Creates a weight parameter w of shape (num_inputs, 1) sampled from a normal distribution with mean 0 and standard deviation sigma.nn.Parameter tells PyTorch that w is a trainable parameter.(num_inputs, 1) means it’s a column vector, one weight per input feature.requires_grad=True ensures gradients are computed for this parameter during backpropagation.
        self.w = nn.Parameter(torch.normal(0,sigma,(num_inputs,1),requires_grad=True))
        #Creates a bias parameter b as a single scalar initialized to 0.Also a trainable parameter (requires_grad=True).
        self.b = nn.Parameter(torch.zeros(1,requires_grad=True))
        self.lr = lr

@d2l.add_to_class(linear_regression_model)
#def forward(self, X):: Defines the forward pass of the model.X is a batch of input features with shape (batch_size, num_inputs).
def forward(self,X):
    #Computes the linear regression prediction: y_hat = Xw + b.
    # X is (batch_size, num_inputs) and w is (num_inputs, 1), so Xw is (batch_size, 1). Adding b (a scalar) broadcasts to (batch_size, 1).
    return torch.matmul(X,self.w)+self.b

@d2l.add_to_class(linear_regression_model)
#def loss(self, y_hat, y):: The loss function receives predictions y_hat and true labels y.
# l = ((y_hat - y)**2)/2: Computes the squared error divided by 2 for each prediction. Dividing by 2 is a common practice in older formulations of MSE, but it doesn’t change the optimization.
#return l.mean(): Takes the mean over all samples in the batch, resulting in the mean squared error. This is a standard regression loss.
def loss(self,y_hat,y):
    l =((y_hat-y)**2)/2
    return l.mean()

# @d2l.add_to_class(linear_regression_model)
# def loss(self,y_hat,y):
#     return ((y_hat-y)**2).mean()

#Defines a simple Stochastic Gradient Descent (SGD) optimizer
class SGD(HyperParameters):
    def __init__(self,params,lr):
        self.save_hyperparameters()
    
    def step(self):
        #Iterates over each parameter in self.params
        #param -= self.lr * param.grad: Updates the parameter by subtracting lr times the gradient. This is the core of SGD: parameter update rule param = param - lr * grad.
        # Iterates over each parameter in self.params
        for param in self.params:
            ## Updates the parameter by subtracting (lr * param.grad) from param
            # param.grad holds the gradient of the loss function with respect to param
            # self.lr is the learning rate
            # By performing param = param - lr * grad, we perform a step of gradient descent
            param -=self.lr*param.grad
    
    def zero_grad(self):
        #Iterates over parameters.
        for param in self.params:
            #If a parameter has a gradient (param.grad is not None), it resets it to zero by calling param.grad.zero_()
            if param.grad is not None:
                param.grad.zero()

d2l.add_to_class(linear_regression_model)
#A convention used in this framework to define how optimizers are created for the model.
def configure_optimizers(self):
    #return SGD([self.w, self.b], self.lr):Creates and returns an SGD instance using the model’s parameters self.w and self.b and the stored learning rate self.lr
    #When training, the trainer will call model.configure_optimizers() to get the optimizer and then call optimizer.step() and optimizer.zero_grad() during training loops.
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
print(data.raw_val.iloc[:5,[0,1,2,3,-3,-2,-1]])

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
    #numeric_features finds the names of all numeric columns.features.dtypes gives the type of each column.features.dtypes != 'object' creates a boolean mask for columns that are not of type object (i.e., likely numeric types: int, float)..index retrieves the column names that match this criterion.
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    #For each numeric column, we subtract the mean and divide by the standard deviation. This ensures each numeric feature has zero mean and unit variance, which helps many models converge faster.
    features[numeric_features] = features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    #Some numeric columns might have missing values. After standardization, missing values might still persist. Here, we fill them with 0, a common strategy when dealing with missing numeric data. This ensures no NaN values remain
    features[numeric_features] =features[numeric_features].fillna(0)
    #pd.get_dummies(features, dummy_na=True) converts all categorical (object-type) columns into one-hot encoded columns.dummy_na=True creates an additional indicator column for missing values in the categorical variables.After this step, all features are numeric. Categorical data is represented by binary indicator columns.
    features = pd.get_dummies(features,dummy_na=True)   
    # features[:self.raw_train.shape[0]]: takes the first `self.raw_train.shape[0]` rows of the combined features. These rows correspond to the original training set.We slice features back into training and validation sets. 
    # self.train = features[:self.raw_train.shape[0]].copy() takes the first N rows, where N is the size of the training set, effectively reversing the initial concatenation.

    self.train = features[:self.raw_train.shape[0]].copy()
    #self.raw_train[label] retrieves the original SalePrice column from the raw training data.
    #self.train[label] = self.raw_train[label] adds it back to the preprocessed training DataFrame. Now self.train contains both the standardized and encoded features plus the target SalePrice
    
    self.train[label]= self.raw_train[label]
    #features[self.raw_train.shape[0]:] selects rows after the training set, which correspond to the test dataset (originally raw_val).
    #Assigning this to self.val means self.val now contains the preprocessed features for the test set, ready for inference or validation.
    self.val =features[self.raw_train.shape[0]:].copy()
#At this point, self.train is a DataFrame with all the preprocessed features and the target SalePrice for training. self.val is a DataFrame with all the preprocessed features (without SalePrice) for the test set.
data.preprocess()
# print(data.train.shape)

#@d2l.add_to_class(KaggleHouse): This decorator adds the get_dataloader method to the KaggleHouse class. It’s a pattern used in the D2L codebase to attach methods after the class definition.
@d2l.add_to_class(KaggleHouse)
#def get_dataloader(self, train):: The method takes a boolean flag train to indicate whether to retrieve a training or validation dataloader.
def get_dataloader(self,train):
    #label = 'SalePrice': Defines the target variable (house price).
    label = 'SalePrice'
    #if train is True, select the training dataset (self.train). If False, select the validation/test dataset (self.val).
    data = self.train if train else self.val
    #Checks if the SalePrice column is present in the selected dataset. If not, it means this dataset does not have labels (as is the case with the test set), so it returns None. Without labels, a supervised training dataloader cannot be created.
    if label not in data: return
   #get_tensor is a lambda function that converts a pandas DataFrame or Series x into a PyTorch float32 tensor.
    get_tensor = lambda x: torch.tensor(x.values.astype(float),dtype=torch.float32)
    #data.drop(columns=[label]) removes the SalePrice column, leaving only the features (X).
    #get_tensor(data.drop(columns=[label])) converts all the features into a PyTorch tensor.
    #data[label] selects the target column SalePrice. Applying get_tensor(data[label]) converts it into a tensor.
    #torch.log(get_tensor(data[label])): Takes the natural logarithm of the target values. This is a common practice in regression tasks where the target distribution is skewed. Logging the target can help stabilize training and improve performance..reshape((-1, 1)) ensures the target tensor Y has the shape (num_samples, 1) instead of (num_samples,). Having a 2D shape is often more convenient for models expecting inputs and outputs in a batch-friendly format.
    #So tensors is a tuple: (X_tensor, Y_tensor), where X_tensor are the features and Y_tensor is the log-transformed target.
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               torch.log(get_tensor(data[label])).reshape((-1, 1)))  # Y
    #self.get_tensorloader is a method defined in the DataModule base class (or elsewhere in the code) that takes these tensors and train flag, and returns a DataLoader.This DataLoader will Create a TensorDataset from the (X, Y) tensors.Wrap it in a PyTorch DataLoader object with the specified batch_size, shuffling if train=True and not if train=False.
    return self.get_tensorloader(tensors, train)

#How K-fold Cross-validation Works:Divide the Data into K Folds: Suppose you have a dataset with N samples. You choose a number k, such as 5 or 10. You then split the N samples into k roughly equal-sized subsets, known as "folds." For instance, if k=5, you might split your data into 5 subsets of equal size.Selecting the Validation Fold: For each iteration of the cross-validation process, you pick one of these k folds as the validation set, and combine the remaining k-1 folds to form the training set.In the first iteration, fold 1 is the validation set, and folds 2–5 are the training set.In the second iteration, fold 2 is the validation set, and folds 1,3–5 are the training set.This process repeats until each of the k folds has served as the validation set exactly once.Training and Evaluating the Model, For each iteration, Train the model on the k-1 training folds.Evaluate (test) the model on the remaining 1 fold that was held out for validation.This means you will train and test the model k times, each time with a different validation set.Averaging the Performance: Because each fold of the data got a chance to serve as the validation set, you obtain k different performance measures (e.g., mean squared error or accuracy values). To get a single performance estimate, you average these k performance scores. This average provides a more stable and less biased estimate of the model’s ability to generalize to new, unseen data than a single train-validation split would.K-fold cross-validation works by repeatedly splitting the dataset into training and validation sets in a systematic way, training and evaluating the model multiple times, and then averaging the results.
def K_fold_data(data,k):
    #initializes an empty list to store the folds.
    rets =[]
    #Calculates the number of samples in each fold by dividing the total number of rows in the training data by the number of folds (k)
    fold_size = data.train.shape[0] // k
    #for j in range(k) Iterates over each fold index from 0 to k-1.Determines how many samples go into each fold. For example, if there are 1000 samples and k=5, fold_size = 200
    for j in range(k):
        #Defines the indices for the current fold
        #idx = range(j*fold_size, (j+1)*fold_size): Defines the indices of the validation fold for the current iteration. For example, if j=0, idx = range(0,200). If j=1, idx = range(200,400), and so on in 1000-sample dataset.
        idx = range(j*fold_size,(j+1)*fold_size)
        #data.train.drop(index=idx) Removes the rows corresponding to the current fold indices (idx) to create the training subset for this fold.data.train.loc[idx] Selects the rows corresponding to the current fold indices (idx) to create the validation subset for this fold.
        rets.append(KaggleHouse(data.batch_size,data.train.drop(index=idx),data.train.loc[idx]))
    #rets.append(...):passing:data.batch_size: The batch size for data loading.data.train.drop(index=idx): The training subset for this fold.data.train.loc[idx]: The validation subset for this fold.
    #Returns a list of KaggleHouse instances, where each instance represents one fold of the K-fold cross-validation.Returns a list of KaggleHouse instances, each corresponding to one of the k folds. Each instance has its own train and val sets.
    return rets
#k_fold function implements K-fold cross-validation:
#It obtains k folds from K_fold_data.
#For each fold: Creates a new linear regression model.Trains the model on k-1 folds of data.Validates it on the remaining 1 fold.Records the validation loss.
#After all folds are processed, it prints the average validation loss.Returns all the trained models.

def k_fold(trainer, data, k, lr):
    #val_loss is initialized as an empty list. It will hold the validation losses (log mean squared errors) obtained from each of the k folds.models is an empty list that will store the trained models from each fold.
    val_loss, models = [], []
    #K_fold_data(data, k) is a function that splits the training data into k folds. It returns a list of k KaggleHouse objects, each containing:data_fold.train: the training subset for that fold (k-1 folds of data).data_fold.val: the validation subset for that fold (the remaining 1 fold of data).enumerate(...) allows us to loop over these folds with an index i (ranging from 0 to k-1) and the data_fold (which is a KaggleHouse instance for that fold).
    for i, data_fold in enumerate(K_fold_data(data, k)):
        #data_fold.train.shape[1] gives the number of columns in the training dataset for this fold. One column is the target variable (SalePrice), so to get the number of input features, we subtract 1.
        num_inputs = data_fold.train.shape[1] - 1  # Exclude target column
        #Instantiates a new linear_regression_model with num_inputs (the number of features) and lr (the learning rate).Each fold gets its own fresh model instance, so the model is always trained from scratch on that fold’s train/val data.
        model = linear_regression_model(num_inputs, lr)  # Pass num_inputs
        #trainer.fit(model, data_fold) runs the training loop. Uses data_fold.train_dataloader() to get training batches. Uses data_fold.val_dataloader() to get validation batches.Runs for max_epochs epochs (in this example, max_epochs=10 as specified in the trainer initialization).During training, the model’s parameters are updated.After each epoch, validation performance is computed on data_fold.val.
        trainer.fit(model, data_fold)
        #Once training is complete for this fold, model.board.data['val_loss'] contains the recorded validation losses per epoch.
        # model.board.data['val_loss'][-1].y retrieves the last recorded validation loss (the validation loss at the final epoch).float(...) converts it to a regular Python float, and val_loss.append(...) adds it to the val_loss list.At the end of all folds, val_loss will contain k validation loss values, one for each fold
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        #After finishing training on the current fold, we store the trained model in models.
        models.append(model)
        #Once the loop ends, we have trained on all k folds and recorded all the validation losses.
        # sum(val_loss)/len(val_loss) computes the average of these validation losses, giving an overall performance estimate of the model on unseen data.
    print(f'average validation log mse of K_fold  = {sum(val_loss)/len(val_loss)}')
    #The function returns the models list, which contains k trained models (one per fold)
    return models
trainer = Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)


print(data.train.iloc[:5,[0,1,2,3,-3,-2,-1]])

def l2_penalty(w):
    return (w**2).sum()/2


class weightdecay(linear_regression_model):
    def __init__(self,num_inputs,lambd,lr,sigma=0.01):
        super().__init__(num_inputs,lr,sigma)
        self.save_hyperparameters()
    
    def loss(self,y_hat,y):
        return(super().loss(y_hat,y)+self.lambd*l2_penalty(self.w))
    
def K_foldwith_weight_decay(trainer,lambd,data,k):
    val_loss, models = [], []
    for i, data_fold in enumerate(K_fold_data(data, k)):
        #make changes in lambda
        model = weightdecay(num_inputs=data.train.shape[1]-1,lambd=lambd,lr=0.01)
        trainer.fit(model,data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
        print(f'l2 norm of w:',float(l2_penalty(model.w)))
    print(f'average validation log mse of k_fold with weight decay= {sum(val_loss)/len(val_loss)}')
    return models

trainer = Trainer(max_epochs=10)
#make changes in lambda
K_foldwith_weight_decay(trainer,50,data,5)


def droput_layer(X,dropout):
    assert 0<=dropout<=1
    if dropout ==1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape)>dropout).float()
    return mask*X/(1.0-dropout)

class Droput_MLP(linear_regression_model):
    def __init__(self,num_inputs,num_outputs,num_hidden_1,num_hidden_2,dropout1,dropout2,lr):
        super().__init__(num_inputs=num_inputs,lr=lr)
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hidden_1)
        self.lin2 = nn.LazyLinear(num_hidden_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()
    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = droput_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = droput_layer(H2, self.dropout2)
        return self.lin3(H2)



def k_fold_with_dropout(trainer, data, k, num_outputs,num_hidden_1,num_hidden_2,dropout1,dropout2,lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(K_fold_data(data, k)):
        num_inputs= data.train.shape[1]-1 # Exclude target column
        model = Droput_MLP(num_inputs,num_outputs,num_hidden_1,num_hidden_2,dropout1,dropout2,lr)

        # Train the model on the current fold
        trainer.fit(model, data_fold)
        # Log the validation loss
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        # Store the trained model
        models.append(model)
    # Compute and print the average validation loss
    print(f'Average validation log MSE of k_fold_with_dropout = {sum(val_loss) / len(val_loss)}')
    return models


trainer = Trainer(max_epochs=10)

models = k_fold_with_dropout(trainer, data, k=5, num_outputs=1,num_hidden_1=128,num_hidden_2=64,dropout1=0.5,dropout2=0.5,lr=0.01)
