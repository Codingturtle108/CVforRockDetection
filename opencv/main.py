# Dependencies
import os
# import cv2
import tensorflow as tf
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imghdr
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy



# Avoid error
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# # Base code for the cnn


# # img = cv2.imread(os.path.join('data', 'sandstone', '8c0a1e9b6ac682d2a8d2aee4eca2dd6a.jpg'))
# # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# # plt.show()
# # print(img.shape)



# # Remove dodgy images

data_dir = "./Dataset"
image_extensions = ["bmp", "gif", "jpeg", "png"]
img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]

def clean_data(data_dir: str):
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                os.remove(filepath)
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                os.remove(filepath)


# Loading dataset
data = tf.keras.utils.image_dataset_from_directory('Dataset', batch_size = 32, label_mode = 'categorical')

def batch_info():
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    return f'(Batch Size, (Image Size), Channel Size) -> {batch[0].shape}'


class_names = data.class_names

# Print the class names
print("Class Names:", class_names)

# Scale Data 
def scale_data(data):
    scaled_data = data.map(lambda x, y: (x / 255, y))
    batch = scaled_data.as_numpy_iterator().next()
    return f'After resizing -> max = {batch[0].max()}, min = {batch[0].min()}'

train_size = int(len(data)*.7)+1
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

def train_val_test_data(data, train_size: int, val_size:int, test_size: int):
    return data.take(train_size), data.skip(train_size).take(val_size), data.skip(train_size+val_size).take(test_size)

train, val, test = train_val_test_data(data, train_size, val_size, test_size)

# Building the model

model = Sequential()

#layer 1
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

# layer 2
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#layer 3
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

#flatten
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

# compile the model
model.compile('adam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Training model

#logging 
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Evaluate

# pre = Precision()
# recall = Recall()
# cat_acc = CategoricalAccuracy()

# for batch in test.as_numpy_iterator():
#     X, y = batch
#     yhat = model.predict(X)
#     pre.update_state(y, yhat)
#     recall.update_state(y ,yhat)
#     cat_acc.update_state(y, yhat)
# print(f'Precision{pre.result()}, Recall {recall.result()}, Accuracy {cat_acc.result()}')



# Testing data

img = cv.imread('coalimage.jpeg')
plt.imshow(img)

resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

#Prediction

yhat = model.predict(np.expand_dims(resize / 255, 0))
one_hot_predictions = np.eye(yhat.shape[1])[np.argmax(yhat, axis=1)]
print(one_hot_predictions)



