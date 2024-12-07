import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import pandas as pd
from collections import Counter
import cv2



# Load your trained model
model_file = "my_model.keras"
model = load_model(model_file)

# Define emotion class labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to your dataset
dataset_path = 'images/test'

# Load the dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_path,
    labels='inferred',           # Automatically assign labels based on folder names
    label_mode='int',            # Labels as integers (e.g., 0, 1, 2, ...)
    batch_size=32,               # Adjust batch size as needed
    image_size=(48, 48),         # Resize all images to 48x48
    shuffle=False                # Set to False if order matters
)

# Print class names and dataset details
class_names = test_data.class_names
print(f"Class Names: {class_names}")
print(f"Number of batches: {len(test_data)}")

for images, labels in test_data.take(1):  # Take the first batch
    print(images.shape)  # Shape of the batch: (batch_size, 48, 48, 3)
    print(labels)        # Corresponding labels


for image_batch, label_batch in test_data:
    for img, lbl in zip(image_batch, label_batch):
        img = img.numpy() / 255.0  # Normalize
        print(f"Image shape: {img.shape}, Label: {lbl}")

results = []

for images, labels in test_data:
    predictions = model.predict(images)
    for pred, true_label in zip(predictions, labels):
        predicted_class = class_names[np.argmax(pred)]
        true_class = class_names[true_label.numpy()]
        results.append({"True Label": true_class, "Predicted Label": predicted_class})

# Create a DataFrame to display results
results_df = pd.DataFrame(results)
Counter(results_df)
print(results_df)

