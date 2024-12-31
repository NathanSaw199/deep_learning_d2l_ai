import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from sklearn.svm import SVC
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

# print(classification_report(Y_test,y_pred))

#bayes rule
#P(A|B) = (P(B|A).P(A))/P(B)

#navie bayes is bayes applied in classification 

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train,Y_train)


y_pred_nb = nb_model.predict(X_test)
# print(classification_report(Y_test,y_pred_nb))

#logistic regression 
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train,Y_train)


y_pred_lg = lg_model.predict(X_test)
# print(classification_report(Y_test,y_pred_lg))

#SVM
svm_model = SVC()
svm_model = svm_model.fit(X_train,Y_train)


y_pred_svm = svm_model.predict(X_test)
# print(classification_report(Y_test,y_pred_svm))


#Neural network 
def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  # plt.show()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs=3
for num_nodes in [16, 32, 64]:
  for dropout_prob in[0, 0.2]:
    for lr in [0.01, 0.005, 0.001]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, Y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history)
        val_loss = model.evaluate(X_valid, Y_valid)[0]
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model



leastloss = least_loss_model.predict(X_test)
leastloss = (leastloss>0.5).astype(int).reshape(-1,)
print(leastloss)
print(classification_report(Y_test,leastloss))
