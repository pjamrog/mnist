
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline

#Training data
dtrain = pd.read_csv('../input/train.csv')
print(dtrain.shape)
dtrain.head()
#Training data
dtrain = pd.read_csv('../input/train.csv')
print(dtrain.shape)
dtrain.head()

z_train = Counter(train['label'])
z_train
z_train = Counter(train['label'])
z_train

#Testing data
dtest = pd.read_csv('../input/test.csv')
print(dtest.shape)
dtest.head()
#Testing data
dtest = pd.read_csv('../input/test.csv')
print(dtest.shape)
dtest.head()

x_train = (dtrain.iloc[:,1:].values).astype('float32')
y_train = dtrain.iloc[:,0].values.astype('int32') 
x_test = dtest.values.astype('float32')
x_train = (dtrain.iloc[:,1:].values).astype('float32')
y_train = dtrain.iloc[:,0].values.astype('int32') 
x_test = dtest.values.astype('float32')

#Data Visualization
%matplotlib inline
plt.figure(figsize=(14,10))
x, y = 10, 4
for i in range(20):
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape((28,28)), cmap='brg')
plt.show()
#Data Visualization
%matplotlib inline
plt.figure(figsize=(14,10))
x, y = 10, 4
for i in range(20):
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape((28,28)), cmap='brg')
plt.show()
​

x_train = x_train/255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train/255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test/255.0
print(x_test.shape[0], 'test samples')
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
x_test = x_test/255.0
print(x_test.shape[0], 'test samples')
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)

y_train
y_train

batch_size = 64
num_classes = 10
epoch = 20
input_shape = (28, 28, 1)
batch_size = 64
num_classes = 10
epoch = 20
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)

-
model = Sequential()
model.add(Conv2D(filters=16, kernel_size = (3,3), activation ='relu', input_shape= (28,28,1)))
model.add(Conv2D(filters=16, kernel_size = (3,3), activation ='relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))
​
model.add(Conv2D(filters=32, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(filters=32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))
​
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
​
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
​
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1)

model.summary()
model.summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

datagen.fit(x_train)
h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epoch, validation_data = (x_val,y_val),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction],)
datagen.fit(x_train)
h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epoch, validation_data = (x_val,y_val),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction],)

final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

plt.plot(h.history['loss'], color='g')
plt.plot(h.history['val_loss'], color='c')
plt.show()
plt.plot(h.history['acc'], color='g')
plt.plot(h.history['val_acc'], color='c')
plt.show()
plt.plot(h.history['loss'], color='g')
plt.plot(h.history['val_loss'], color='c')
plt.show()
plt.plot(h.history['acc'], color='g')
plt.plot(h.history['val_acc'], color='c')
