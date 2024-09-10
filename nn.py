import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# split test, train data
df = pd.read_csv("processed_data.csv")
attr_cols = ["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S"]
X = df[attr_cols]# attributes
y = df["Survived"] # prediction label
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

row, col = X_train.shape
print(X_train.shape)


# Convert NumPy arrays to TensorFlow tensors: tf only takes tensors
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)


# model implementation
# structure
model = keras.models.Sequential()
model.add(layers.Dense(10, input_dim= 12, activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))

# compile
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history= model.fit(X_train,y_train, epochs= 28) 

# evaluate model on test data
print("Evaluate on test data:")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# plot loss & accuracy
epochs = range(1,29) 
plt.figure(figsize=(14,5))
plt.plot(epochs, history.history['loss'], 'r*-', label='Test Loss', color="blue")
plt.plot(epochs, history.history['accuracy'], 'r*-', label='Test Accuracy', color="red")
plt.title(' Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss and Accuracy')
plt.legend()
plt.savefig("loss_acc.png")
plt.show()


# plot confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to binary class predictions
cm = confusion_matrix(y_test, y_pred_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
cm_display.plot()
plt.savefig("confusion_matrix.png")
plt.show()