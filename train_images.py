import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import math
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_examples = 26351
cv_examples = 3410
test_examples = 3365
img_height = img_width = 224
batch_size = 32

# Data Generators
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range=40,
    zoom_range = (0.95, 0.95),
    horizontal_flip = True,
    vertical_flip = True,
    data_format = "channels_last"

)

cv_datagen = ImageDataGenerator(rescale = 1.0/255)

test_datagen = ImageDataGenerator(rescale = 1.0/255)

# Data Generation
train_gen = train_datagen.flow_from_directory(
    "sorted_images/train/",
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "binary",
    shuffle = True,
    seed = 123
)

cv_gen = cv_datagen.flow_from_directory(
    "sorted_images/cv/",
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "binary",
    shuffle = True,
    seed = 123
)

test_gen = test_datagen.flow_from_directory(
    "sorted_images/test/",
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "binary",
    shuffle = True,
    seed = 123
)

# Building the model
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc")
]

model.compile(
    keras.optimizers.Adam(3e-4), 
    loss=tf.losses.BinaryCrossentropy(from_logits=False), 
    metrics=METRICS
)

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

model.fit(
    train_gen, 
    epochs = 15,
    steps_per_epoch = train_examples // batch_size,
    validation_data = cv_gen,
    validation_steps = cv_examples // batch_size,
    callbacks = [tensorboard_callback],
    verbose=2
)

model.save("trained_net")
