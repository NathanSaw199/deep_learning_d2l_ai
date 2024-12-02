import os
import pandas as pd
import torch


# ------------------------------------------------------
#2.2.1. Reading the Dataset

import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
    
data = pd.read_csv(data_file)
# print(data)

# ------------------------------------------------------
#2.2.2 Data Preparation
inputs,targets = data.iloc[:,0:2], data.iloc[:,2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.fillna(inputs.mean())
print(inputs)

# ------------------------------------------------------
#2.2.3 Conversion to the Tensor Format

x = torch.tensor(inputs.to_numpy(dtype='float'))
y = torch.tensor(targets.to_numpy(dtype='float'))

print(x, y)