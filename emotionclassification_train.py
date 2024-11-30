'''
python3.7.7 cuda11.2
pip install tensorflow-gpu==2.6.0
pip install protobuf==3.20.*
pip install matplotlib
python -m pip install --upgrade pip
pip install opencv-python
pip install keras==2.6
'''
"""# Importing Libraries"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os

from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from keras.models import Sequential

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.python.keras import backend as K

sess = tf.compat.v1.Session()
K.set_session(sess)

# Generators

train_data = keras.utils.image_dataset_from_directory(
    directory = 'images/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (224, 224)
)

validation_data = keras.utils.image_dataset_from_directory(
    directory = 'images/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (224, 224)
)

"""# Normalizing the data"""

# Normalization

def Normalize(image, label):
  image = tf.cast(image/255.0, tf.float32)
  return image,label

train = train_data.map(Normalize)
validate = validation_data.map(Normalize)


"""# Loading the model to be used for Transfer Learning"""

import os.path
model_file = "my_model.keras"
if os.path.isfile(model_file):
    print(f"Loaded existing model: {model_file}")
    model = tf.keras.models.load_model(model_file)
else:
    num_classes = 7
    
    model = Sequential()
    model.add(tf.keras.layers.Rescaling(1./255))
    model.add(Conv2D(64, (5,5), padding="Same", activation='relu', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3))) 
    model.add(Conv2D(64, (5,5), padding="Same", activation='relu'))
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (4,4), padding="Same", activation='relu'))
    model.add(Flatten())
    model.add(Dense(3072, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(train, epochs = 3, validation_data = validate)
model.save(model_file)

plt.plot(history.history['accuracy'], color = 'red', label = 'train')
plt.plot(history.history['val_accuracy'], color = 'blue', label = 'validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color = 'red', label = 'train')
plt.plot(history.history['val_loss'], color = 'blue', label = 'validation')
plt.legend()
plt.show()


def data_augmentation(X, y, df, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1,
                      brightness_range=(0.95, 1.05), horizontal_flip=True, vertical_flip=True, fill_mode='nearest', target_count=None):
 datagen = tf.keras.preprocessing.image.ImageDataGenerator(
     rotation_range=rotation_range,
     width_shift_range=width_shift_range,
     height_shift_range=height_shift_range,
     zoom_range=zoom_range,
     brightness_range=brightness_range,
     horizontal_flip=horizontal_flip,
     vertical_flip=vertical_flip,
     fill_mode=fill_mode
 )


 balanced_X = []
 balanced_y = []


 target_count = target_count if target_count else max(df['emotion'].value_counts())


 distribution = []


 for class_label in df['emotion'].unique():
     class_indices = np.where(y[:, class_label] == 1)[0]
     class_images = X[class_indices]
     class_labels = y[class_indices]
     num_images = class_images.shape[0]
     distribution.append(num_images)
    
     augmentations_needed = target_count - num_images
    
     while True:
         for img, label in zip(class_images, class_labels):
             if (augmentations_needed <= 0):
               break
             img = img.reshape((1,) + img.shape)
             label = label.reshape((1,) + label.shape)
             augmented_img = next(datagen.flow(img))
             balanced_X.append(augmented_img.squeeze())
             balanced_y.append(label.squeeze())
             augmentations_needed -= 1
         if (augmentations_needed <= 0):
               break


     balanced_X.extend(class_images)
     balanced_y.extend(class_labels)


 balanced_X = np.array(balanced_X)
 balanced_y = np.array(balanced_y)


 return balanced_X, balanced_y, distribution
