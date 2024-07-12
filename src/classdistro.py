# src/data_process.py
import os
import pathlib
import matplotlib.pyplot as plt

raw_dataset_directory = pathlib.Path(r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\rawData")
processed_train_directory = pathlib.Path(r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\processedData\train")
processed_val_directory = pathlib.Path(r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\processedData\val")

def count_images_in_folders(directory):
    folder_image_counts = {}
    total_images = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            num_images = len([file for file in os.listdir(subdir_path) if file.endswith(('.png', '.jpg', '.jpeg'))])
            folder_image_counts[subdir] = num_images
            total_images += num_images
    return folder_image_counts, total_images

def plot_histogram(image_counts, title):
    folders = list(image_counts.keys())
    counts = list(image_counts.values())
    plt.figure(figsize=(12, 6))
    plt.barh(folders, counts, color='skyblue')
    plt.xlabel('Number of Images')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=8) 
    plt.show()

def main():
    raw_image_counts, total_raw_images = count_images_in_folders(raw_dataset_directory)
    train_image_counts, total_train_images = count_images_in_folders(processed_train_directory)
    val_image_counts, total_val_images = count_images_in_folders(processed_val_directory)

    sorted_raw_image_counts = dict(sorted(raw_image_counts.items(), key=lambda x: x[1]))
    sorted_train_image_counts = dict(sorted(train_image_counts.items(), key=lambda x: x[1]))
    sorted_val_image_counts = dict(sorted(val_image_counts.items(), key=lambda x: x[1]))

    print(f"Raw Data folder counts: ")
    for folder, count in sorted_raw_image_counts.items():
        print(f"Folder: {folder}, Number of images: {count}")
    print(f"Total number of raw images: {total_raw_images}")

    print(f"Processed Train Data folder counts: ")
    for folder, count in sorted_train_image_counts.items():
        print(f"Folder: {folder}, Number of images: {count}")
    print(f"Total number of train images: {total_train_images}")

    print(f"Processed Val Data folder counts: ")
    for folder, count in sorted_val_image_counts.items():
        print(f"Folder: {folder}, Number of images: {count}")
    print(f"Total number of val images: {total_val_images}")

    plot_histogram(sorted_raw_image_counts, 'Number of Images per Folder (Raw Data)')
    plot_histogram(sorted_train_image_counts, 'Number of Images per Folder (Processed Train Data)')
    plot_histogram(sorted_val_image_counts, 'Number of Images per Folder (Processed Val Data)')

if __name__ == "__main__":
    main()
