import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pickle
import h5py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Allow memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Custom Data Adapter
class MyHDF5DatasetAdapter(tf.keras.utils.Sequence):
    def __init__(self, hdf5_file, batch_size, shuffle=True, augment=False):
        self.hdf5_file = hdf5_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self._load_data()

    def _load_data(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            self.data = f['images'][:]
            self.labels = f['labels'][:]
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]
        if self.augment:
            batch_data = next(self.datagen.flow(batch_data, batch_size=self.batch_size, shuffle=False))
        return batch_data, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Define paths
processed_data_dir = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\data\processedData"
model_save_path = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\models\mushroom_identifier.keras"
kpi_save_path = r"C:\Users\psycl\Documents\GitHub\WGU_CS_Capstone\models\model_kpis.pkl"
train_hdf5_file = os.path.join(processed_data_dir, 'train', 'Training Dataset.h5')
val_hdf5_file = os.path.join(processed_data_dir, 'val', 'Validation Dataset.h5')

print("Loading class names...")
with open(os.path.join(processed_data_dir, 'class_names.pkl'), 'rb') as f:
    class_names = pickle.load(f)
num_classes = len(class_names)
print("Class names loaded successfully.")

# Custom data generators
train_generator = MyHDF5DatasetAdapter(train_hdf5_file, batch_size=32, augment=False) 
val_generator = MyHDF5DatasetAdapter(val_hdf5_file, batch_size=32, shuffle=False)

print("Building the model...")
# Build the model
def build_model(input_shape, num_classes):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    # Split input channels into RGB, LBP, and edges
    rgb = input_layer[:, :, :, :3]
    lbp = input_layer[:, :, :, 3:4]
    edges = input_layer[:, :, :, 4:5]

    # Process RGB channels with ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    x = base_model(rgb)

    # Process LBP and edges with a small CNN
    lbp_conv = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(lbp)  
    lbp_conv = tf.keras.layers.MaxPooling2D((2, 2))(lbp_conv)
    lbp_conv = tf.keras.layers.Flatten()(lbp_conv)

    edges_conv = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(edges)  
    edges_conv = tf.keras.layers.MaxPooling2D((2, 2))(edges_conv)
    edges_conv = tf.keras.layers.Flatten()(edges_conv)

    # Concatenate processed features
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    combined = tf.keras.layers.Concatenate()([x, lbp_conv, edges_conv])
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)

    model = tf.keras.Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (224, 224, 5)  
model = build_model(input_shape, num_classes)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("Training the model...")
# Train the model
history = model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[early_stopping, reduce_lr])

# Save the model in Keras format
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

# Evaluate the model and calculate KPIs
print("Evaluating the model...")
y_val_pred = model.predict(val_generator)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)

# Debugging: Print shapes of true and predicted values
print(f"Shape of val_generator.labels: {val_generator.labels.shape}")
print(f"Shape of y_val_pred_classes: {y_val_pred_classes.shape}")

# Adjust labels to match the length of predictions
min_length = min(len(val_generator.labels), len(y_val_pred_classes))
labels_adjusted = val_generator.labels[:min_length]
pred_classes_adjusted = y_val_pred_classes[:min_length]

accuracy = accuracy_score(labels_adjusted, pred_classes_adjusted)
precision = precision_score(labels_adjusted, pred_classes_adjusted, average='weighted', zero_division=0)
recall = recall_score(labels_adjusted, pred_classes_adjusted, average='weighted', zero_division=0)
f1 = f1_score(labels_adjusted, pred_classes_adjusted, average='weighted', zero_division=0)

kpis = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1
}

# Save KPIs
with open(kpi_save_path, 'wb') as f:
    pickle.dump(kpis, f)

print("Model training complete and saved to", model_save_path)
print("Model KPIs saved to", kpi_save_path)

# Print overall validation accuracy and KPIs
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(labels_adjusted, pred_classes_adjusted, normalize='true')
plt.figure(figsize=(20, 20)) 
sns.heatmap(conf_matrix, annot=False, fmt=".2f", cmap='viridis', cbar=True)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Normalized Confusion Matrix', fontsize=18)
plt.savefig('confusion_matrix.png') 

# Save confusion matrix to CSV
conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_df.to_csv('confusion_matrix.csv', index=False)

# Plot learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_validation_accuracy_loss.png') 
plt.show()
