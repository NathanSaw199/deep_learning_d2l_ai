import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
cols = ["fLength","fWidth", "fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df = pd.read_csv(r'C:\Users\Saw\Desktop\UCI_ML\magic+gamma+telescope\magic04.data',names = cols)
# print(file_read)
# print(df.head())

# print(df["class"].unique())
df["class"] = (df["class"] == "g").astype(int)
# print(df["class"].unique())
# print(df.head())



#l1 loss is loss = sum(|y-y^|) 
#l2 loss is loss = sum((y-y^)2)


# binary crosss - entropy loss for binary classification
# loss = -1/N*sum(y*log(y)+(1-y)*log((1-y^)))


# for  label in cols[:-1]:
#     plt.hist(df[df["class"]==1][label],color= "blue",label= "gamma",alpha =0.7,density = True)
#     plt.hist(df[df["class"]==0][label],color= "red",label= "hadron",alpha =0.7,density = True)
#     plt.title(label)
#     plt.ylabel("probability")
#     plt.xlabel(label)
#     plt.legend()
#     # plt.show()



#train,validation, test datasets

# train,valid, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y

# print(len(train[train["class"]==1])) #gamma
# print(len(train[train["class"]==0])) #gamma
train,X_train, Y_train = scale_dataset(train,oversample=True)
valid,X_valid, Y_valid = scale_dataset(valid,oversample=False)
test,X_test, Y_test = scale_dataset(test,oversample=False)


# print(f" y train = 1 : {sum(Y_train==1)}")
# print(f" X train = 0 : {sum(Y_train==0)}")
# print(len(train))

# k nearest neighbours 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,Y_train)

y_pred = knn_model.predict(X_test)
# print(y_pred)
# print(Y_test)

print(classification_report(Y_test,y_pred))

#bayes rule
#P(A|B) = (P(B|A).P(A))/P(B)