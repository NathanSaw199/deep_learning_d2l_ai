#pytorch computer vision 
#conputer vision library in pytorch torch vision

#torchvision.datasets: contains a number of popular datasets for computer vision tasks.
#torchvision.models: contains popular pre-trained models for computer vision.
#torchvision.transforms: contains common image transformations that can be chained together using Compose.
#torchvision.utils.data.DataLoader: a utility for loading and iterating over data in batches.
#torchvision.utils.data.Dataset: an abstract class representing a dataset.
import torch.utils.data.dataloader
from tqdm.auto import tqdm
import requests
from pathlib import Path
import torch
from torch import nn 
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helper_functions import accuracy_fn
import pandas as pd 
import random
#Step 1: Getting the dataset and turn into tensors
#Step 2: build or pick a pretrained model-> build a training loop

#getting a dataset 
#The FashionMNIST dataset contains images of clothing items from torchvision.datasets

#1. setup training data
train_data = datasets.FashionMNIST(
    root = 'data', #where to download the data
    train = True, #specifies training or test dataset
    download = True, #download the data or not
    transform=torchvision.transforms.ToTensor(), #convert the image to tensor
    target_transform=None #convert the target to tensor

)
test_data = datasets.FashionMNIST(
    root = 'data', #where to download the data
    train = False, #specifies training or test dataset
    download = True, #download the data or not
    transform=torchvision.transforms.ToTensor(), #convert the image to tensor
    target_transform=None #convert the target to tensor

)



#the first training example 
image,label = train_data[0]
# print(image,label)
# print(image.shape,label)


class_names = train_data.classes
# print(class_names)

class_to_idx = train_data.class_to_idx
# print(class_to_idx)


image,label = train_data[0]
# print(image.shape)

#Creates a new figure using Matplotlib with a specified size of 9 inches by 9 inches.fig is the figure object that will hold the subplots.
# fig = plt.figure(figsize=(9,9))
# #Defines the number of rows and columns in the grid of subplots. This creates a 4x4 grid (16 subplots in total).
# rows,cols = 4,4
# #A loop that iterates from 1 to 16 (inclusive). Each iteration corresponds to creating one subplot.The range(1, rows*cols + 1) ensures that the subplot index matches Matplotlib’s 1-based numbering.
# for i in range(1,rows*cols+1):
#     #Randomly selects an index from the train_data dataset.torch.randint(0, len(train_data), size=[1]): Generates a random integer between 0 and the length of train_data (exclusive)..item(): Converts the single-element tensor to a Python integer. size=[1] ensures that a scalar tensor is returned.

#     random_idx = torch.randint(0,len(train_data),size =[1]).item()
#     #Retrieves an image and its corresponding label from the dataset at the randomly selected index.image: The input data (usually a tensor representing pixel values of the image).label: The target class label for the image.

#     image,label = train_data[random_idx]
#     #Adds a subplot to the figure.rows and cols specify the grid dimensions.i specifies the position of the subplot in the grid.
#     fig.add_subplot(rows,cols,i)    
#     #image.squeeze(): Removes any singleton dimensions from the tensor (e.g., converts a shape of [1, 28, 28] to [28, 28]).cmap='gray': Sets the colormap to grayscale, suitable for single-channel images like MNIST digits.

#     plt.imshow(image.squeeze(),cmap='gray')
#     plt.title(class_names[label])
#     plt.axis(False)
# # plt.show()


#2. prepare dataloader
# data is in tehe form of pytoch datasets. dataloar turns dataset into a python iterable. we want to turn data into batches( mini batches) of data because we can't pass the entire dataset into the model at once. we break it down to 32 images at a time (batch size = 32). parameter update its gradient per epoch.

#set up the batch size hyperparameter. hyperparameter is a parameter whose value is set before the learning process begins. In this case, the batch size is set to 32. wwe can set hyperparameters to different values and compare the model's performance.
Batch_Size = 32
#turn datasets into iterable (batches)
train_dataloader = DataLoader(dataset=train_data,batch_size=Batch_Size,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=Batch_Size,shuffle=False)


# print(train_dataloader,test_dataloader)
#batches of 32 images and labels and 1875 batches in total (60,000 images divided into batches of 32 images each). batchsize
# print(len(train_dataloader),len(test_dataloader))



#check out what is inside the training data loader
train_features_batch,train_labels_batch = next(iter(train_dataloader))

# print(train_features_batch.shape,train_labels_batch.shape)
#show a sample 
torch.manual_seed(42)
random_idx = torch.randint(0,len(train_features_batch),size=[1]).item()
image,label = train_features_batch[random_idx], train_labels_batch[random_idx]

plt.imshow(image.squeeze(),cmap='gray')
plt.title(class_names[label])
plt.axis(False)
print(f"images size {image.shape},label : {label},label size {label.shape}")
# plt.show()


#3. create a model baseline
#best practice to create baseline model is a simple model that can be improved upon.


##create a flatten layer 
flatten_model = nn.Flatten()
#get a single sample 
x = train_features_batch[0]

#flattlen the sample
output = flatten_model(x) 
print(x.shape) #original shape
print(output.shape) #flattened shape


class FashionMNISTModelV0(nn.Module):
    def __init__(self,input_shape :int,hidden_units : int,output_shape: int):
        super().__init__()
        self.Layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )

    def forward(self,x):
        return self.Layer_stack(x)
torch.manual_seed(42)
#set up model with input paramters

model_0 = FashionMNISTModelV0(input_shape=28*28,hidden_units=10,output_shape=len(class_names)).to('cpu')

# print(model_0)
dummy_x = torch.rand([1,1,28,28])
# print(model_0(dummy_x))

#4. set up loss function and optimizer and evaluation metric

# loss function  for multi class data is cross entropy loss
# optimzer is stochastic gradient descent
# evaluation metric is accuracy


#download the helper function from learn pytoch repo 

if Path("helper_functions.py").is_file():
    print("Helper functions file found")
else:
    print("Downloading helper functions file")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py","wb") as file:
        file.write(request.content)

# from helper_functions import calculate_accuracy
#import accuracy metric 
# set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)


#3.2 create a function to time the experiments
def print_train_time(start : float,end : float, device : torch.device = None):
    total_time = end - start
    print(f"Training time: on{device} : {total_time:.2f} seconds")
    return total_time

start_time = timer()
# write codes here
end_time = timer()

print(print_train_time(start_time,end_time,device='cpu'))

#3.3 creating a training loop and training the model on batches of data
# loop through epochs
# loop through training batches and calcuate loss per batch
# loop trhough test batches and perform tesing steps and calculate test loss per batch
# print out what is happening 
#Time it all 

#set the seed and start the timer
#optimizer will update a model's parameters once per batch rather than once per epoch
#manuel_seed is used to ensure that the results are reproducible. The seed is set to 42, which is a common practice in machine learning experiments. The seed value can be any integer. The same seed value will produce the same random numbers every time the code is run. This is useful for debugging and reproducing results. 
torch.manual_seed(42)
train_time_start_on_cpu = timer()   
#set number of epochs
epochs = 3
#create training and test loop 
for epoch in tqdm(range(epochs)):
    print(f"epoch : {epoch}\n-------------------------------")
    #training 
    train_loss = 0 
    #add a loop to loop through the batches of training data
    for batch,(X,y) in enumerate(train_dataloader):
        model_0.train()
        #1. forward pass
        y_pred = model_0(X)
        #2, Calculate loss (per batch)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        #3. optimizer zero grad
        optimizer.zero_grad()
        #4. backward pass
        loss.backward()
        #5. optimizer step
        optimizer.step()
        #print out 
    if batch % 400  == 0:
        print(f"Looked at {batch*len(X)}/{len(train_dataloader.dataset)} samples")
    #divide total train loss by length of train data loader
    train_loss /= len(train_dataloader)

    #testing 
    test_loss, test_acc =0,0
    model_0.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            #forward pass
            test_pred= model_0(X_test)
            #calculate loss
            test_loss += loss_fn(test_pred,y_test)
            #calculate accuracy
            test_acc+= accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))    
        #divide total test loss by length of test data loader to get average test loss per batch
        test_loss /= len(test_dataloader)
        #divide total test accuracy by length of test data loader to get average test accuracy per batch
        test_acc /= len(test_dataloader)

    
    # print(f"\n train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")


train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(train_time_start_on_cpu,train_time_end_on_cpu,device=str(next(model_0.parameters()).device))

#4. make prediction and get model 0 result 
def eval_model(model : torch.nn.Module,
                data_loader : torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn):
    
    #return a dictionary containing the results of model predicting on data_loader
    loss, acc =0 ,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            #make predictions 
            y_pred = model(X)
            #accumulate the loss and acc values per batch
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true = y,y_pred = y_pred.argmax(dim=1))
        
        #scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"Model Name" : model.__class__.__name__,
            "Model loss" : loss.item(),
            "Model acc" : acc }

#calculate model 0 result on test dataset
model_0_result = eval_model(model=model_0,data_loader=test_dataloader,loss_fn= loss_fn,accuracy_fn=accuracy_fn)

# print(model_0_result)


#set up device agnostic-code ( using a GPU)

device = "cuda" if torch.cuda.is_available() else "cpu"


#improve through experimentation 

#create a model with non linear and linear data
class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape : int,
                 hidden_units : int,
                 output_shape : int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shape),
            nn.ReLU()
        )


    def forward(self, x: torch.tensor):
            return self.layer_stack(x)
        


    #create and intance of model
torch.manual_seed(42)
model_l= FashionMNISTModelV1(input_shape = 784,
                                  hidden_units =10,
                                  output_shape = len(class_names)).to(device)
    

    
print(next(model_l.parameters()).device)


# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_l.parameters(),
                             lr=0.1)

#functionizing training and evaluation/testing loops


#training loop = train_step()
#testing loop = test_step()

def train_step(model: torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               accuracy_fn,
               device : torch.device = device
                ):
    train_loss, train_acc = 0,0
    model.train()
    #add a loop to loop through the batches of training data
    for batch,(X,y) in enumerate(data_loader):
        #put data on target device
        X,y = X.to(device),y.to(device)
        #1. forward pass
        y_pred = model(X)
        #2, Calculate loss (per batch)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        train_acc +=accuracy_fn(y_true =y,y_pred=y_pred.argmax(dim=1))
        #3. optimizer zero grad
        optimizer.zero_grad()
        #4. backward pass
        loss.backward()
        #5. optimizer step
        optimizer.step()
        #print out 

    #divide total train loss and accuracy by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"train loss :{train_loss :.5f}, train acc : {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              accuracy_fn,
              device : torch.device = device):
    test_loss, test_acc =0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X, y= X.to(device),y.to(device)
                #forward pass
            test_pred= model(X)
                #calculate loss
            test_loss += loss_fn(test_pred,y)
                #calculate accuracy
            test_acc+= accuracy_fn(y_true=y,y_pred=test_pred.argmax(dim=1))    
            #divide total test loss by length of test data loader to get average test loss per batch
        test_loss /= len(data_loader)
            #divide total test accuracy by length of test data loader to get average test accuracy per batch
        test_acc /= len(data_loader)

        
        print(f"\n test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

torch.manual_seed(42)

# Measure time
train_time_start_on_gpu = timer()
# Train and test model 
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_l,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_l,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

train_time_end_on_gpu = timer()

total_train_time_model_1 = print_train_time(start = train_time_start_on_gpu, end = train_time_end_on_gpu,device=device)

print(total_train_time_model_1)

print(model_0_result,total_train_time_model_0)



#get model 1 result dictionary
def eval_model(model : torch.nn.Module,
                data_loader : torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                device = device):
    
    #return a dictionary containing the results of model predicting on data_loader
    loss, acc =0 ,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device),y.to(device)
            #make predictions 
            y_pred = model(X)
            #accumulate the loss and acc values per batch
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true = y,y_pred = y_pred.argmax(dim=1))
        
        #scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"Model Name" : model.__class__.__name__,
            "Model loss" : loss.item(),
            "Model acc" : acc }

#calculate model 0 result on test dataset
model_1_result = eval_model(model=model_l,data_loader=test_dataloader,loss_fn=loss_fn,accuracy_fn=accuracy_fn,device=device)
print(model_1_result)


class FashionMNISTModelV2(nn.Module):
    #replicates the TINY VGG
    def __init__(self, input_shape : int, hidden_units : int, output_shape : int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)


        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )
    def forward(self,x):
        x = self.conv_block_1(x)
        # print(f"output shape of conv block 1 {x.shape}")
        x = self.conv_block_2(x)
        # print(f"output shape of conv block 2 {x.shape}")
        x = self.classifier(x)
        # print(f"output shape of classifier {x.shape}")
        return x 

model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=30,
                              output_shape=len(class_names)).to(device)
# print(image.shape)

rand_image_tensor = torch.randn(size=(1,28,28))
# print(rand_image_tensor.shape)

# print(model_2(rand_image_tensor.unsqueeze(0).to(device)))


##training CNN model with our own dataset loss function and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_2.parameters(),lr=0.1)



train_time_start_model_2 = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch : {epoch}\n---------")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_2,
               data_loader=test_dataloader,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               device=device)
    

train_time_end_model_2 = timer()

total_train_time = print_train_time(start=train_time_start_model_2,end = train_time_end_model_2)



model_2_result = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device

)
print(model_2_result)

compare_results = pd.DataFrame([model_0_result,
                                model_1_result,
                                model_2_result])


compare_results["training_time"] =[total_train_time_model_0,
                                   total_train_time_model_1,
                                   
                                   total_train_time]
print(compare_results)

#make and evluate random predciton with the best model 


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device= device):
    pre_probs = []
    model.eval()
    model.to(device)
    with torch.inference_mode():
        for sample in data:
            #prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample,dim=0).to(device)

            #forward pass ( model outputs raw logtis)
            pred_logit = model(sample)
            #get prediction probability ( logit - > prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)

            # get pred_prob off the GPU for futther calculations 
            pre_probs.append(pred_prob.cpu())
    #stack the pred probs to turn list into a tensor
    return torch.stack(pre_probs)

random.seed(42)
test_samples = []
test_labels = []

for sample,label in random.sample(list(test_data),k=9):
    test_samples.append(sample)
    test_labels.append(label)

test_samples[0].shape

plt.imshow(test_samples[0].squeeze(),cmap="gray")
plt.title(class_names[test_labels[0]])
plt.show()


#make predictions 
pred_probs = make_predictions(model=model_2,data=test_samples)


#view first two prediction probabilities 
pred_probs[:2]


#convert prediction problity to labels 
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)


#making confusing matrix for futher prediction evaluation 


