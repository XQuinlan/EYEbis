# Raspberry Pi Image Classifier with Buzzer

This script runs on a Raspberry Pi (tested on Pi 5) to capture an image using a PiCamera, classify it using a TensorFlow Lite model, and activate a buzzer if a specific condition (e.g., "contaminated") is detected.

## Features

- Captures images using the Picamera2 library.
- Preprocesses images (resize, normalize) for model compatibility.
- Performs inference using a TFLite model (`.tflite`).
- Controls a buzzer connected to GPIO pins based on classification results.
- Configurable settings for GPIO pin, image size, model path, labels, etc.
- Includes error handling and resource cleanup.

## Hardware Requirements

- Raspberry Pi (e.g., Pi 5, Pi 4)
- Raspberry Pi Camera Module (e.g., NoIR Camera V2)
- Active Buzzer
- Jumper Wires

## Software Requirements

- Raspberry Pi OS (or compatible Linux distribution)
- Python 3
- Required Python libraries (see Installation)
- A trained TensorFlow Lite classification model (`model.tflite`)

## Hardware Setup

1.  **Camera:** Connect the Raspberry Pi Camera Module to the CSI port on the Raspberry Pi.
2.  **Buzzer:**
    - Connect one leg of the buzzer to GPIO pin 18 (using BCM numbering).
    - Connect the other leg of the buzzer to a Ground (GND) pin on the Raspberry Pi.

## Installation

1.  **Clone or Download:** Get the `image_classifier_pi.py` script and this `README.md` file onto your Raspberry Pi.

2.  **Enable Camera:** Ensure the camera interface is enabled in your Raspberry Pi configuration:
    ```bash
    sudo raspi-config
    ```
    Navigate to `Interface Options` -> `Camera` -> `Enable`.

3.  **Install Dependencies:** Open a terminal on your Raspberry Pi and run the following commands:
    ```bash
    sudo apt update
    sudo apt install -y python3-picamera2 python3-pil python3-numpy
    pip3 install rpi-gpio tflite-runtime
    ```
    *Note: The `tflite-runtime` installation might vary depending on your specific Raspberry Pi OS version and Python version. If the above command fails, refer to the official TensorFlow Lite installation guide for Raspberry Pi.*

4.  **Place Model:** Copy your trained TensorFlow Lite model file (it must be named `model.tflite`) into the same directory as the `image_classifier_pi.py` script.

## Configuration

You can modify the following constants at the beginning of the `image_classifier_pi.py` script if needed:

- `BUZZER_PIN`: The BCM GPIO pin number connected to the buzzer (default: 18).
- `IMAGE_SIZE`: The target size (width, height) the image will be resized to for the model (default: (224, 224)). Match this to your model's input requirement.
- `MODEL_PATH`: The filename of your TFLite model (default: "model.tflite").
- `CLASS_LABELS`: A list of strings representing the class labels your model predicts, in the correct order (default: ["clean", "contaminated"]). The script assumes the second label ("contaminated") triggers the buzzer.
- `NUM_THREADS`: Number of threads for the TFLite interpreter (default: 4). Adjust based on your Pi model.
- `CAPTURE_RESOLUTION`: The resolution used for capturing the image initially (default: (640, 480)).
- `BUZZ_DURATION`: How long the buzzer stays on in seconds when activated (default: 0.5).

## Usage

1.  Navigate to the directory containing the script and the model file in the terminal:
    ```bash
    cd /path/to/your/script
    ```

2.  Run the script using Python 3. You might need `sudo` for GPIO access:
    ```bash
    sudo python3 image_classifier_pi.py
    ```

The script will then:
- Initialize GPIO.
- Initialize the camera.
- Capture an image.
- Preprocess the image.
- Load the TFLite model.
- Run inference.
- Print the predicted label and confidence.
- If the prediction is "contaminated" (or the second label in `CLASS_LABELS`), it will activate the buzzer for `BUZZ_DURATION` seconds.
- Clean up GPIO resources and close the camera before exiting.

## Troubleshooting

- **GPIO Permissions:** If you get permission errors, try running with `sudo`.
- **Model Not Found:** Ensure `model.tflite` is in the same directory as the script and the filename matches `MODEL_PATH`.
- **Incorrect Prediction:** Verify `IMAGE_SIZE` matches your model's input. Ensure preprocessing steps (normalization, etc.) are correct for your specific model. Check if `CLASS_LABELS` match the output order of your model.
- **Camera Errors:** Ensure the camera is properly connected and enabled via `raspi-config`. Check `libcamera` compatibility if issues persist.
- **Library Issues:** Double-check that all required libraries are installed correctly using the installation commands. 