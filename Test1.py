# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau

# reading the train and the test databases
# ========================================
train_df = pd.read_csv("input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test_df = pd.read_csv("input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

# saving the labels of test separately
# ========================================
test = pd.read_csv("input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
y = test['label']

# presenting top rows of train
# ========================================
train_df.head()

# shows how many training examples of each label exist
# ========================================
plt.figure(figsize = (10,10)) # Label Count
sns.set_style("darkgrid")
sns.countplot(train_df['label'])
# plt.show()

# delete from train and test data the label and save it separately
# ========================================
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# perform a grayscale normalization to reduce the effect of illumination's differences
# ========================================
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

# save the train and test without labels
# ========================================
x_train = train_df.values
x_test = test_df.values

# ========================================
# the labels of train and test are saved in y_train y_test and the rest is saved in x_train x_test
# ========================================

# normalize the data
# ========================================
x_train = x_train / 255
x_test = x_test / 255

# reshaping the data from 1-D to 3-D as required through input by CNN's
# ========================================
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1);

# preview of first 10 images
# ========================================
f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28), cmap="gray")
        k += 1
    plt.tight_layout()
# plt.show()

# data augmentation(artificial extension of our data set) to prevent overfitting
# ========================================
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)

# ========================================
# Training The Model
# ========================================

# define the parameters to reduce learning rate when a metric has stopped improving
# ========================================
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

#  group a linear stack of layers
# ========================================
model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

# final fit of the model
# ========================================
history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 1 , validation_data = (x_test, y_test) ,
                    callbacks = [learning_rate_reduction])

# present percent of accuracy
# ========================================
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

# ========================================
# analysing after Model Training
# ========================================

# present accuracy and loss in a graph
# ========================================
epochs = [i for i in range(1)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

# predict the test data set, classify each image to the right label
# ========================================
predictions = model.predict_classes(x_test)
for i in range(len(predictions)):
    # ignore J letter
    if(predictions[i] >= 9):
        predictions[i] += 1
# predictions[:5]

# build a text report showing the main classification metrics
# ========================================
classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y, predictions, target_names = classes))

# draw a confusion matrix
# ========================================
cm = confusion_matrix(y,predictions)

cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
plt.figure(figsize = (15,15))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')

# check the correct predictions
# ========================================
# תוספת
# y = np.array(y)
# predictions.reshape(y.shape)

correct = np.nonzero(predictions == y)[0]

# some of the correctly predicted classes
# ========================================
i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.imshow(x_test[c].reshape(28,28), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y[c]))
    plt.tight_layout()
    i += 1











# making good mood:)
# ========================================
print("אהלן אחי מה המצב?")

