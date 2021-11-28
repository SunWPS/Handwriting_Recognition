import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model



# Load mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Prepare data
rows = x_train[0].shape[0]
cols = x_train[0].shape[1]

# reshape
x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test.reshape(x_test.shape[0], rows, cols, 1)

# change data type
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Normalize data (0 -> 255) to (0 -> 1)
x_train /= 255
x_test /= 255

# print(x_train[0])

# One Hot Encod Labels
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(y_test[:20])
# print(y_test.shape)

n_classes = y_test.shape[1]
n_pixels = x_train.shape[1] * x_train.shape[2]

# Build Model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(rows, cols, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# training
history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ' + str(score[1]))
print('Test loss: ' + str(score[0]))


# plot graph
# accuracy and epoch
plt.plot(history.history["accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

# loss and epoc
plt.plot(history.history["loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

model.save('mnist_recognition.h5')
print("Saved")
