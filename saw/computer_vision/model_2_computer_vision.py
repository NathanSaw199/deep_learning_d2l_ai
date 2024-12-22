#Convolutional neural network

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



import torch

from torch import nn 

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

image,label = train_data[0]

class_names = train_data.classes

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
device = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
# print(image.shape)

rand_image_tensor = torch.randn(size=(1,28,28))
# print(rand_image_tensor.shape)

# print(model_2(rand_image_tensor.unsqueeze(0).to(device)))


##training CNN model with our own dataset loss function and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_2.parameters(),lr=0.1)
# print(model_2.state_dict())




# training and testing model 2 using training and test functions 


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
torch.cuda.manual_seed(42)


Batch_Size = 32
def print_train_time(start : float,end : float, device : torch.device = None):
    total_time = end - start
    print(f"Training time: on{device} : {total_time:.2f} seconds")
    return total_time

#measure time
train_dataloader = DataLoader(dataset=train_data,batch_size=Batch_Size,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=Batch_Size,shuffle=False)


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
            #make predictions 
            X,y = X.to(device),y.to(device)
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


model_2_result = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
    device=device

)
print(model_2_result)








# plt.imshow(image.squeeze(),cmap="gray")

# plt.show()

# #nn.conv2d

# torch.manual_seed(42)
# #create a batch of images 
# images = torch.randn(size = (32,3,64,64))
# test_image = images[0]
# # print(f"image batch shape: {images.shape}")
# # print(f"single image shape : {test_image.shape}")
# # print(f"Test image: \n{test_image}")


# torch.manual_seed(42)
# #create a single conv2d layer 
# conv_layer = nn.Conv2d(in_channels=3,
#                        out_channels=64,
#                        kernel_size=(3,3),
#                        stride=1,
#                        padding=1)
# #pass the data thru conv layer

# conv_output = conv_layer(test_image)
# print(conv_output.shape)

# print(test_image.shape)



# max_pool_layer = nn.MaxPool2d(kernel_size=2)
# #pass data through just conv layer
# test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
# print(f"Shape after going through conv_layer : {test_image_through_conv.shape}")


# #pass data through the max pool layer
# test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
# print(f"shape after going through conv layer and max pool layer : {test_image_through_conv_and_max_pool.shape}")

