# Dependencies
import os
import cv2
import tensorflow as tf
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imghdr
from pathlib import Path



# Avoid error
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# Base code for the cnn


# img = cv2.imread(os.path.join('data', 'sandstone', '8c0a1e9b6ac682d2a8d2aee4eca2dd6a.jpg'))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
# print(img.shape)



# Remove dodgy images

# data_dir = "./Dataset"
# image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]  # add there all your images file extensions

# img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
# for filepath in Path(data_dir).rglob("*"):
#     if filepath.suffix.lower() in image_extensions:
#         img_type = imghdr.what(filepath)
#         if img_type is None:
#             print(f"{filepath} is not an image")
#             os.remove(filepath)
#         elif img_type not in img_type_accepted_by_tf:
#             print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
#             os.remove(filepath)


# Loading dataset
data = tf.keras.utils.image_dataset_from_directory('Dataset', batch_size = 16)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(batch[0].shape)

fig, ax = plt.subplots(ncols=4, figsize=(10, 10))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Scale Data 

# data = data.map(lambda x, y: (x / 255, y))
# scaled_iterator = data.as_numpy_iterator()
# batch = scaled_iterator.next()
# print(batch[0].max())







