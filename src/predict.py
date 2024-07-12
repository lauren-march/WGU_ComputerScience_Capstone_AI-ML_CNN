import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
from tensorflow.keras.preprocessing import image
from skimage.feature import local_binary_pattern

model_path = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\models\mushroom_identifier.keras"
class_names_path = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\processedData\class_names.pkl"
test_data_dir = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\testData"

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

print(f"Loading class names from: {class_names_path}")
with open(class_names_path, 'rb') as f:
    class_names = pickle.load(f)
print("Class names loaded successfully.")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img).astype(np.uint8)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
    lbp = lbp / np.max(lbp)
    lbp = np.expand_dims(lbp, axis=-1)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)
    edges = edges / 255.0
    edges = np.expand_dims(edges, axis=-1)

    # Stack original RGB, LBP, and edges
    img_combined = np.concatenate((img_array, lbp, edges), axis=-1)
    
    img_combined = img_combined / 255.0 
    img_combined = np.expand_dims(img_combined, axis=0)
    return img_combined

def find_and_preprocess_images(test_data_dir):
    images = []
    img_paths = []
    for filename in os.listdir(test_data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_data_dir, filename)
            print(f"Loading and preprocessing image from: {img_path}")
            images.append(preprocess_image(img_path))
            img_paths.append(img_path)
    if not images:
        raise FileNotFoundError("No image files found in the test data directory.")
    return images, img_paths

test_images, img_paths = find_and_preprocess_images(test_data_dir)

for i, test_image in enumerate(test_images):
    print(f"Found image {img_paths[i]}. Starting predictions...")
    predictions = model.predict(test_image)

    for j, prob in enumerate(predictions[0]):
        print(f"Class '{class_names[j]}': {prob * 100:.2f}%")

    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    predicted_confidence = predictions[0][predicted_class_index]

    print(f"The image {img_paths[i]} most likely belongs to '{predicted_class_name}' with a confidence of {predicted_confidence * 100:.2f} percent.")
