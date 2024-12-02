import pandas as pd

data = pd.read_csv(r"C:\Users\Saw\Desktop\deep_learning_d2l_ai\saw\abalone.csv", usecols=["Sex"])
# print(data.isnull().sum())
inputs, targets = data.iloc[:, 0:1], data.iloc[:, 0]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
# print(data)