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
        print(f"output shape of conv block 1 {x.shape}")
        x = self.conv_block_2(x)
        print(f"output shape of conv block 2 {x.shape}")
        x = self.classifier(x)
        print(f"output shape of classifier {x.shape}")
        return x 
device = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
# print(image.shape)

rand_image_tensor = torch.randn(size=(1,28,28))
print(rand_image_tensor.shape)

print(model_2(rand_image_tensor.unsqueeze(0).to(device)))

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

