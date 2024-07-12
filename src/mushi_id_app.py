import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import pickle
import cv2
from tensorflow.keras.preprocessing import image
from skimage.feature import local_binary_pattern
import os
import csv

base_dir = os.path.join(os.path.expanduser("~"), "Documents", "GitHub", "WGU_CS_Capstone")

model_path = os.path.join(base_dir, "models", "mushroom_identifier.keras")
class_names_path = os.path.join(base_dir, "data", "processedData", "class_names.pkl")
conf_matrix_img_path = os.path.join(base_dir, "data", "modelDataVisualizations", "ConfusionMatrix.jpg")
accuracy_graph_img_path = os.path.join(base_dir, "data", "modelDataVisualizations", "TrainingAndValidation.jpg")
sample_image_path = os.path.join(base_dir, "data", "modelDataVisualizations", "CaloceraViscosaSample.jpg")
user_feedback_path = os.path.join(base_dir, "data", "user_feedback.csv")

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

def predict_image(model, img_path, class_names):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    predicted_confidence = predictions[0][predicted_class_index]
    return predicted_class_name, predicted_confidence

class MushiIDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mushi ID")
        self.geometry("1500x600")  

        self.tab_control = ttk.Notebook(self)
        
        # Getting Started Tab
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text='Getting Started')
        self.create_getting_started_tab()

        # Upload Images Tab
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab2, text='Upload Images')
        self.create_upload_images_tab()

        # Metrics Dashboard Tab
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab3, text='Metrics Dashboard')
        self.create_metrics_dashboard_tab()

        # User Feedback Tab
        self.tab4 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab4, text='User Feedback')
        self.create_user_feedback_tab()

        self.tab_control.pack(expand=1, fill='both')

        self.current_prediction = None

    def create_getting_started_tab(self):
        guide_label = tk.Label(self.tab1, text="How to Use this App", font=("Helvetica", 29))
        guide_label.pack(pady=20)
        
        steps = """1. Navigate to the Upload Images Tab.
        
        2. Click the Upload button and select the image you wish to upload. Images must be in jpg format.
        
        3. Click okay and wait for the image to upload.
        
        4. View the prediction and confirm if this was the correct classification of your mushroom.
        
        5. Click the Metrics Dashboard to see visualization of the model and a sample images.
        
        6. Use the User Feedback tab to log the accuracy of the model based on your feedback.

        Note: You can expand the window if necessary to see the metric graphs more clearly"""
        
        steps_label = tk.Label(self.tab1, text=steps, font=("Helvetica", 15), justify=tk.CENTER, wraplength=681)
        steps_label.pack(pady=20)

    def create_upload_images_tab(self):
        upload_label = tk.Label(self.tab2, text="Upload Your Image", font=("Helvetica", 29))
        upload_label.pack(pady=20)
        
        self.progress_bar = ttk.Progressbar(self.tab2, orient='horizontal', mode='determinate', length=200)
        self.progress_bar.pack(pady=20)
        self.progress_bar.pack_forget() 

        self.uploading_image_percent_label = tk.Label(self.tab2, text="uploading image... %")
        self.uploading_image_percent_label.pack(pady=5)
        self.uploading_image_percent_label.pack_forget()  

        self.upload_button = tk.Button(self.tab2, text="Upload", font=("Helvetica", 16), width=10, height=1, command=self.handle_upload_button)
        self.upload_button.pack(pady=10)

        self.result_label = tk.Label(self.tab2, text="Result: The mushroom is likely a { } with a ", font=("Helvetica", 18))
        self.result_label.pack(pady=20)
        self.result_label.pack_forget()  

        self.confidence_label = tk.Label(self.tab2, font=("Helvetica", 18))
        self.confidence_label.pack_forget()  

        self.image_label = tk.Label(self.tab2)
        self.image_label.pack(pady=10)

        self.feedback_frame = tk.Frame(self.tab2)
        self.feedback_frame.pack(pady=20)
        self.feedback_frame.pack_forget()  

        self.yes_button = tk.Button(self.feedback_frame, text="Yes", font=("Helvetica", 16), width=10, height=1, command=lambda: self.log_feedback("correct"))
        self.yes_button.pack(side=tk.LEFT, padx=10)

        self.no_button = tk.Button(self.feedback_frame, text="No", font=("Helvetica", 16), width=10, height=1, command=lambda: self.log_feedback("incorrect"))
        self.no_button.pack(side=tk.LEFT, padx=10)

    def handle_upload_button(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.upload_button.config(state=tk.DISABLED)
            self.progress_bar.pack()
            self.uploading_image_percent_label.pack()
            self.upload_image(file_path)

    def upload_image(self, file_path):
        def upload():
            try:
                predicted_class, confidence = predict_image(model, file_path, class_names)
                confidence_percentage = confidence * 100

                self.current_prediction = predicted_class  

                color = "green" if confidence_percentage > 50 else "red"
                self.result_label.config(text=f"Result: The mushroom is likely a {predicted_class} with a ", fg="black")
                self.confidence_label.config(text=f"{confidence_percentage:.2f}% confidence", fg=color)
                self.result_label.pack()
                self.confidence_label.pack()

                self.show_image(file_path)
                self.feedback_frame.pack()  
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                self.progress_bar.pack_forget()
                self.uploading_image_percent_label.pack_forget()
                self.upload_button.config(state=tk.NORMAL)

        self.after(100, upload)

    def show_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((224, 224), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def log_feedback(self, feedback_type):
        if self.current_prediction:
            with open(user_feedback_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.current_prediction, feedback_type])
            self.update_feedback_stats()
            self.feedback_frame.pack_forget()

    def create_user_feedback_tab(self):
        feedback_label = tk.Label(self.tab4, text="User Feedback", font=("Helvetica", 29))
        feedback_label.pack(pady=20)

        self.feedback_text = tk.Text(self.tab4, height=15, width=70, font=("Helvetica", 12), state=tk.DISABLED)
        self.feedback_text.pack(pady=20)

        self.update_feedback_stats()

    def update_feedback_stats(self):
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete('1.0', tk.END)
        feedback_stats = self.calculate_feedback_stats()
        self.feedback_text.insert(tk.END, feedback_stats)
        self.feedback_text.config(state=tk.DISABLED)

    def calculate_feedback_stats(self):
        if not os.path.exists(user_feedback_path):
            return "No feedback available."

        feedback_data = {}
        total_predictions = 0
        correct_predictions = 0

        with open(user_feedback_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                predicted_class, feedback = row
                if predicted_class not in feedback_data:
                    feedback_data[predicted_class] = {'correct': 0, 'incorrect': 0}
                feedback_data[predicted_class][feedback] += 1
                total_predictions += 1
                if feedback == 'correct':
                    correct_predictions += 1

        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

        feedback_stats = f"Total Predictions: {total_predictions}\n"
        feedback_stats += f"Total Correct: {correct_predictions}\n"
        feedback_stats += f"Total Incorrect: {total_predictions - correct_predictions}\n"
        feedback_stats += f"User Feedback Accuracy: {accuracy:.2f}%\n\n"
        feedback_stats += "Detailed Feedback:\n"

        for mushroom, stats in feedback_data.items():
            correct = stats['correct']
            incorrect = stats['incorrect']
            total = correct + incorrect
            mushroom_accuracy = (correct / total) * 100 if total > 0 else 0
            feedback_stats += f"{mushroom}: Predicted {total}, Correct {correct}, Incorrect {incorrect}, Accuracy {mushroom_accuracy:.2f}%\n"

        return feedback_stats

    def create_metrics_dashboard_tab(self):
        metrics_label = tk.Label(self.tab3, text="Model Metrics", font=("Helvetica", 29))
        metrics_label.pack(pady=20)

        metrics_frame = tk.Frame(self.tab3)
        metrics_frame.pack(pady=20, fill='both', expand=True)

        confusion_matrix_label = tk.Label(metrics_frame, text="Confusion Matrix")
        confusion_matrix_label.grid(row=0, column=0, padx=10, pady=5)

        accuracy_graph_label = tk.Label(metrics_frame, text="Training and Validation Metrics")
        accuracy_graph_label.grid(row=0, column=1, padx=10, pady=5)

        image_sample_label = tk.Label(metrics_frame, text="Sample Images from Dataset")
        image_sample_label.grid(row=0, column=2, padx=10, pady=5)

        self.confusion_matrix_canvas = tk.Canvas(metrics_frame, bg='lightgrey')
        self.confusion_matrix_canvas.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.load_and_display_image(conf_matrix_img_path, self.confusion_matrix_canvas)

        self.accuracy_graph_canvas = tk.Canvas(metrics_frame, bg='lightgrey')
        self.accuracy_graph_canvas.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.load_and_display_image(accuracy_graph_img_path, self.accuracy_graph_canvas)

        self.image_sample_canvas = tk.Canvas(metrics_frame, bg='lightgrey')
        self.image_sample_canvas.grid(row=1, column=2, padx=10, pady=5, sticky="nsew")
        self.load_and_display_image(sample_image_path, self.image_sample_canvas)

        metrics_frame.grid_rowconfigure(1, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        metrics_frame.grid_columnconfigure(2, weight=1)

    def load_and_display_image(self, file_path, canvas):
        try:
            image = Image.open(file_path)
            def resize_image(event):
                new_width = event.width
                new_height = event.height
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                photo_image = ImageTk.PhotoImage(resized_image)
                canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
                canvas.image = photo_image

            canvas.bind("<Configure>", resize_image)
        except Exception as e:
            print(f"Failed to load image from {file_path}: {e}")

def run_gui():
    app = MushiIDApp()
    app.mainloop()

if __name__ == "__main__":
    run_gui()

