# 🍄 Mushroom Identification Application

Welcome to my capstone project for my Computer Science degree at WGU! 🎓

![image](https://github.com/user-attachments/assets/e56b4f2f-0c56-4be3-a4b3-32b7062c8040)

## Overview

For this project, I have created a mushroom identification application using transfer learning with a pretrained model (ResNet-50) and a supplemental custom CNN for additional layers. The mushroom identification application is built with Python and TensorFlow along with a user-friendly GUI using Tkinter.

## Features

- **Transfer Learning**: Utilizes ResNet-50 pretrained model for initial layers and a custom CNN for additional layers.
- **User-Friendly GUI**: Built with Tkinter to provide an intuitive and easy-to-navigate interface.
- **Image Processing**: Incorporates advanced image processing techniques including Local Binary Pattern (LBP) and Canny edge detection.
- **Real-Time Prediction**: Quickly uploads an image and returns the mushroom classification with confidence level.

## Getting Started

### Prerequisites

- Python 3.10.0
- TensorFlow 2.9.1
- Pillow
- OpenCV
- Requests
- Skimage

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/mushroom-identification-app.git
    ```
2. Navigate to the project directory:
    ```sh
    cd mushroom-identification-app
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Note on Model File

⚠️ **Important**: The `.keras` model file is too large to be hosted directly on GitHub. Therefore, the actual model file (`mushroom_identifier.keras`) is not included in this repository. To run this application, you will need to download the model file separately and place it in the `models/` directory. 

Please contact me to obtain the model file.

### Running the Application

1. Ensure the `.keras` model file is placed in the `models/` directory:
    ```
    models/mushroom_identifier.keras
    ```
2. Navigate to the `src` directory:
    ```sh
    cd src
    ```
3. Run the main script:
    ```sh
    python main.py
    ```

### Project Structure

- `src/`
  - `main.py`: Entry point to start the Flask server and Tkinter GUI.
  - `flask_app.py`: Contains the Flask application for handling API requests.
  - `gui.py`: Contains the Tkinter GUI application.
  - `data_process.py`: Script for preprocessing images.
  - `train_model.py`: Script for training the model.
  - `predict.py`: Script for making predictions using the trained model.
- `models/`
  - `mushroom_identifier.keras`: **Note: This file needs to be manually added.**
  - `model_kpis.pkl`: Pickle file containing model key performance indicators.
- `data/`
  - `modelDataVisualizations/`: Directory containing visualization images (confusion matrix, accuracy graphs, etc).

## Usage

1. **Upload Images**: Navigate to the 'Upload Images' tab and select an image of a mushroom to upload.
![image](https://github.com/user-attachments/assets/97bd67c6-6c93-468d-9dc2-c634c1fe1415)

2. **View Prediction**: The application will display the predicted mushroom class and confidence level.
![image](https://github.com/user-attachments/assets/ff89f130-b8a1-469b-87f5-ec8898c8e7dd)

3. **Model Metrics**: Navigate to the 'Metrics Dashboard' tab to view the confusion matrix, training and validation accuracy, and a sample image.
![image](https://github.com/user-attachments/assets/0fb0f83e-bfdc-4041-a06a-7f31508d6e74)

4. **User Feedback**: Navigate to the User feed back tab to see user defined metrics on mushroom prediction accuracy.
![image](https://github.com/user-attachments/assets/a2d0ac5f-fe60-4619-8afc-c71cfd63f4b4)


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

If you have any questions or suggestions, feel free to reach out!
