# MobileNetV2 Classifier Training

This script trains a MobileNetV2-based image classifier for 4 classes: 'grease', 'tomato_sauce', 'milk', 'clean'.

## Setup

1.  **Create Data Directories:**
    Ensure you have the following directory structure inside `classifierNet`:
    ```
    classifierNet/
    ├── data/
    │   ├── train/
    │   │   ├── grease/
    │   │   │   └── ... images ...
    │   │   ├── tomato_sauce/
    │   │   │   └── ... images ...
    │   │   ├── milk/
    │   │   │   └── ... images ...
    │   │   └── clean/
    │   │       └── ... images ...
    │   └── validation/
    │       ├── grease/
    │       │   └── ... images ...
    │       ├── tomato_sauce/
    │       │   └── ... images ...
    │       ├── milk/
    │       │   └── ... images ...
    │       └── clean/
    │           └── ... images ...
    ├── train_mobilenet.py
    ├── requirements.txt
    └── README.md
    ```
    Populate the `train` and `validation` subdirectories with your images for each class.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the training script from the `classifierNet` directory:

```bash
cd classifierNet
python train_mobilenet.py
```

The script will:
- Load the pre-trained MobileNetV2 model.
- Add a custom classification head.
- Set up data generators for training and validation.
- Train the model using the images in the `data` directory.
- Save the best performing model (based on validation accuracy) to `best_mobilenet_model.keras`.

## Post-Training Quantization

After training and saving the Keras model (`best_mobilenet_model.keras`), you can quantize it to TensorFlow Lite format for deployment on edge devices.

Make sure you have validation images in `data/validation` as they are used for the representative dataset during INT8 quantization.

Run the quantization script:

```bash
python quantize_model.py
```

This will:
- Load the saved Keras model.
- Perform post-training INT8 quantization (default) using a representative dataset from `data/validation`.
- Save the quantized model as `best_mobilenet_model_quant_int8.tflite`.

### Quantization Types

- **INT8 (Default):** Provides significant size reduction and potential speedup on compatible hardware, with a minor potential impact on accuracy. Requires a representative dataset (`representative_dataset_gen` function uses `data/validation`).
- **Float16:** Reduces model size by about half compared to the original float32 model with minimal accuracy loss. Does not require a representative dataset. To use float16 quantization, edit `quantize_model.py` and change `quantization_type = 'int8'` to `quantization_type = 'float16'`. The output file will be `best_mobilenet_model_quant_float16.tflite`.

## Running Inference on Raspberry Pi

This project includes a script (`rpi_classify.py`) to run the quantized model on a Raspberry Pi with a PiCamera.

### Setup on Raspberry Pi

1.  **Transfer Files:** Copy the `classifierNet` directory (or at least `rpi_classify.py` and the generated `.tflite` model file, e.g., `best_mobilenet_model_quant_int8.tflite`) to your Raspberry Pi.

2.  **Install Dependencies:**
    Install the required Python libraries on your Raspberry Pi:

    ```bash
    # Update package list
    sudo apt update
    sudo apt install -y python3-pip python3-dev

    # Install Pillow (PIL)
    pip3 install Pillow

    # Install picamera
    pip3 install picamera
    # For newer Pi OS versions (Bullseye+), you might need the new picamera2 interface
    # sudo apt install -y python3-picamera2
    # If using picamera2, the script rpi_classify.py would need modifications.

    # Install TensorFlow Lite Runtime
    # Follow the official TensorFlow guide for Python quickstart:
    # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
    # Choose the wheel appropriate for your Pi OS version and Python version.
    # Example for Pi 4, Python 3.9 (check the guide for your specific setup):
    # pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl
    # OR find a suitable wheel for your architecture (armv7l, aarch64)
    pip3 install tflite-runtime
    ```
    *Note: Finding the correct `tflite-runtime` wheel can sometimes be tricky. Refer to the official TensorFlow Lite documentation or Coral AI website for the latest wheels compatible with your Raspberry Pi model and OS version.* 

3.  **Enable Camera:** Ensure the camera interface is enabled on your Raspberry Pi using `sudo raspi-config` (Interfacing Options -> Camera -> Enable).

### Usage on Raspberry Pi

1.  Navigate to the directory containing `rpi_classify.py` and the `.tflite` model.
2.  Run the script:

    ```bash
    python3 rpi_classify.py
    ```

The script will:
- Load the specified `.tflite` model.
- Initialize the PiCamera.
- Capture an image.
- Preprocess the image (resize, quantize).
- Run inference using the TFLite interpreter.
- Print the predicted class label and confidence score.
