import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# from keras.models import load_model
# import keras
# import matplotlib.pyplot as plt
# import os


def gesture_to_name(ges):
    switcher = {
        1: "up",
        2: "left",
        3: "down",
        4: "right",
    }
    return switcher.get(ges, "invalid")

ges = 1
count = 0

train_images = []
train_labels = []
test_images = []
test_labels = []

##loading images from disk
for i in range(0, 400):

    img = cv2.imread("pp_imgs/" + gesture_to_name(ges) + "_" + str(count+1) + ".png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (32, 32))
    # resizing the images to 32X32 to make training easier
    img.resize(32, 32, 1)

    if count % 10 == 0:
        test_images.append(img)
        test_labels.append(ges)

    else:
        train_images.append(img)
        train_labels.append(ges)

    if count == 99:
        ges += 1
        count = 0
    else:
        count += 1

    if ges == 5:
        break

## Converting to np arrays
train_images = np.array(train_images, dtype="uint8")
train_labels = np.array(train_labels)
test_images = np.array(test_images, dtype="uint8")
test_labels = np.array(test_labels)

## Checking the number of images
print("Images loaded: ", len(train_images))
print("Labels loaded: ", len(train_labels))
print("Images test: ", len(test_images))
print("Labels test: ", len(test_labels))
print("test: " + str(test_images.shape))
print("train: " + str(train_images.shape))

train_images, test_images = train_images / 255, test_images / 255

print(train_images.shape)
print(test_images.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=10)

#Evaluating the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

#Saving model to disk
model.save('hg_trained_model.h5')

#Testing model
model.reset_metrics()
predictions = model.predict(test_images)

#Testing new loaded model
new_model = tf.keras.models.load_model('hg_trained_model.h5')
new_model.summary()

new_predictions = new_model.predict(test_images)
test_loss, test_acc = new_model.evaluate(test_images, test_labels, verbose=2)

