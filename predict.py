import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load your trained model
model_file = "my_model.keras"
model = load_model(model_file)

# Define emotion class labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess an image for prediction
def preprocess_image(img_path):
  img = cv2.imread(img_path)
  img = cv2.resize(img, (48, 48))
  img = img.astype("float32") / 255.0
  img = tf.expand_dims(img, axis=0)  # Add a batch dimension
  return img

# Get the image path for prediction
image_path = "neutral.jpg"  # Replace with your image path

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make prediction
prediction = model.predict(preprocessed_image)
predicted_class = class_names[np.argmax(prediction[0])]

# Print the predicted emotion
print(f"Predicted Emotion: {predicted_class}")