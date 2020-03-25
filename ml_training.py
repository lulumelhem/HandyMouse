import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt

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

images_train = []
labels_train = []
images_test = []
labels_test = []

for i in range(0, 400):

    img = cv2.imread("pp_imgs/" + gesture_to_name(ges) + "_" + str(count+1) + ".png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (32, 32))
    # resizing the images to 32X32 to make training easier
    img.resize(32, 32, 3)

    if count % 10 == 0:
        images_test.append(img)
        labels_test.append(ges)

    else:
        images_train.append(img)
        labels_train.append(ges)

    if count == 99:
        ges += 1
        count = 0
    else:
        count += 1

    if ges == 5:
        break

images_train = np.array(images_train, dtype="uint8")
labels_train = np.array(labels_train)
images_test = np.array(images_test, dtype="uint8")
labels_test = np.array(labels_test)

print("Images loaded: ", len(images_train))
print("Labels loaded: ", len(labels_train))
print("Images test: ", len(images_test))
print("Labels test: ", len(labels_test))
print ("test: " + str(images_test.shape))
print ("train: " + str(images_train.shape))

images_train, images_test = images_train/255, images_test/255

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(images_train, labels_train, epochs=10,
                    validation_data=(images_test, labels_test))

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(images_test,  labels_test, verbose=2)

# print(test_acc)

model.reset_metrics()
predictions = model.predict(images_test)

## Saving the model to disk
model.save('hg_trained_model.h5')

new_model = keras.models.load_model('hg_trained_model.h5')

## Checking that the state of the model is preserved
new_predictions = new_model.predict(images_test)
test_loss, test_acc = new_model.evaluate(images_test,  labels_test, verbose=2)