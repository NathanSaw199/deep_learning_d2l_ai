import os
import pandas as pd
import torch 

data = pd.read_csv(r"C:\Users\User\Desktop\d2l_ai\deep_learning_d2l_ai\joseph\abalone.csv", usecols=['Sex', 
'Length', 'Shell weight', 'Rings'])

# print(data)

inputs,targets = data.iloc[:,0:1], data.iloc[:,2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# # It search how many missing values are in the dataset
# print(data.isnull().sum())




