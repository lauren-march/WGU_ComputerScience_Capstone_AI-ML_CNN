import pathlib
import tensorflow as tf
import numpy as np
import pickle
import os
import h5py
from tensorflow.keras.utils import image_dataset_from_directory
from tqdm import tqdm
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern


dataset_directory = pathlib.Path(r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\rawData")
processed_data_dir = pathlib.Path(r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\processedData")

batch_size = 32
img_height = 224  
img_width = 224   

def load_datasets(dataset_directory, img_height, img_width, batch_size):
    """Load datasets from directory."""
    return image_dataset_from_directory(
        dataset_directory,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

def split_dataset(full_ds, split_ratio=0.8):
    """Split dataset into training and validation sets."""
    dataset_size = tf.data.experimental.cardinality(full_ds).numpy()
    train_size = int(dataset_size * split_ratio)
    train_ds = full_ds.take(train_size)
    val_ds = full_ds.skip(train_size)
    return train_ds, val_ds

def preprocess_image(img):
    """Preprocess image by adding LBP and edge detection features."""
    img_array = img.astype(np.uint8)

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
    
    return img_combined / 255.0  

def save_images(images, labels, class_names, directory):
    """Save images to the specified directory."""
    for i, img in enumerate(images):
        class_name = class_names[labels[i]]
        class_dir = directory / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_img = preprocess_image(img)
        img_rgb = preprocessed_img[:, :, :3] 
        img_path = class_dir / f"{i}.jpg"
        img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8)).convert('RGB')  
        img_pil.save(img_path, format='JPEG', quality=85)

def save_dataset_to_hdf5(dataset, directory, dataset_name, class_names):
    """Save preprocessed dataset to HDF5 file."""
    dataset_dir = processed_data_dir / directory
    dataset_dir.mkdir(parents=True, exist_ok=True)

    images_list = []
    labels_list = []

    for images_batch, labels_batch in tqdm(dataset, desc=f"Saving {dataset_name}"):
        images_list.append(images_batch.numpy().astype(np.uint8))
        labels_list.append(labels_batch.numpy())

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    save_images(images, labels, class_names, processed_data_dir / directory)

    with h5py.File(dataset_dir / f'{dataset_name}.h5', 'w') as hf:
        hf.create_dataset('images', data=np.array([preprocess_image(img) for img in images]), compression="gzip")
        hf.create_dataset('labels', data=labels, compression="gzip")

def main():
    """Main function to load, preprocess, and save datasets."""
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    full_ds = load_datasets(dataset_directory, img_height, img_width, batch_size)
    
    class_names = full_ds.class_names
    print("Class names:", class_names)

    print("Splitting dataset...")
    train_ds, val_ds = split_dataset(full_ds, split_ratio=0.8)

    print("Saving training dataset...")
    save_dataset_to_hdf5(train_ds, "train", "Training Dataset", class_names)

    print("Saving validation dataset...")
    save_dataset_to_hdf5(val_ds, "val", "Validation Dataset", class_names)

    with open(processed_data_dir / 'class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)

    print("Data processing and saving completed.")

if __name__ == "__main__":
    main()
