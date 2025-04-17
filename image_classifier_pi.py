#!/usr/bin/env python3

"""
Script for Raspberry Pi 5 to capture an image when Enter is pressed,
classify it using a TFLite model, and print the result (no preview).
"""

import time
# import RPi.GPIO as GPIO # Removed GPIO import
from picamera2 import Picamera2 # Removed Preview import
# from picamera2.previews.qt import QtGlPreview # Removed Qt preview
# from PyQt5.QtWidgets import QApplication # Removed QApplication
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import sys # Import sys for error handling

# --- Configuration ---
# BUZZER_PIN = 18         # GPIO pin connected to the buzzer (BCM numbering) # Removed Buzzer Pin
IMAGE_SIZE = (224, 224) # Target image size for the model
MODEL_PATH = "best_mobilenet_model_quant_float16.tflite" # Path to the TensorFlow Lite model file
CLASS_LABELS = ["clean", "contaminated"] # Labels corresponding to model output indices
NUM_THREADS = 4         # Number of threads for TFLite interpreter (adjust as needed for Pi 5)
CAPTURE_RESOLUTION = (640, 480) # Initial capture resolution
# BUZZ_DURATION = 0.5     # Duration in seconds to activate the buzzer # Removed Buzz Duration

# --- GPIO Setup --- # Removed GPIO setup function
# def setup_gpio():
#     """Initializes GPIO settings."""
#     print("Setting up GPIO...")
#     GPIO.setmode(GPIO.BCM)       # Use Broadcom pin numbering
#     GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW) # Set buzzer pin as output, initially low
#     GPIO.setwarnings(False)      # Disable GPIO warnings

# --- Camera Operations ---
def capture_image(picam2):
    """Captures a single frame from the camera and returns it as a PIL Image."""
    print("Capturing image...")
    # Capture an image (returns a NumPy array) - camera is started just before this call
    image_array = picam2.capture_array()
    print("Image captured.")

    # Convert the NumPy array to a PIL Image (RGB)
    pil_image = Image.fromarray(image_array)

    # Ensure image is RGB if it has an alpha channel (rare for direct capture)
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')

    return pil_image

# --- Image Preprocessing ---
def preprocess_image(pil_image):
    """Resizes, normalizes, and prepares the image for the TFLite model."""
    print("Preprocessing image...")
    # Resize the image
    resized_image = pil_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Convert PIL image to NumPy array
    image_array = np.array(resized_image, dtype=np.float32)

    # Normalize pixel values to [0, 1] range
    normalized_image = image_array / 255.0

    # Expand dimensions to create a batch of one: (1, height, width, channels)
    input_data = np.expand_dims(normalized_image, axis=0)
    print("Image preprocessed.")
    return input_data

# --- TFLite Inference ---
def run_inference(input_data):
    """Loads the TFLite model and performs inference."""
    print(f"Loading TFLite model from {MODEL_PATH}...")
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
        interpreter.allocate_tensors()
    except ValueError as e:
        print(f"Error loading model or allocating tensors: {e}")
        print("Ensure the 'model.tflite' file exists and is a valid TFLite model.")
        sys.exit(1) # Exit if model loading fails
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        sys.exit(1)


    print("Model loaded. Running inference...")
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if the input tensor shape matches the preprocessed image shape
    if not np.array_equal(input_details[0]['shape'], input_data.shape):
        print(f"Error: Model input shape {input_details[0]['shape']} does not match processed image shape {input_data.shape}")
        sys.exit(1)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Inference complete.")
    return output_data

# --- Decision Making ---
def process_prediction(prediction):
    """Analyzes the prediction and prints the result."""
    # Get the index of the highest probability score
    predicted_index = np.argmax(prediction[0])

    # Ensure the index is within the bounds of our labels
    if predicted_index >= len(CLASS_LABELS):
        print(f"Error: Predicted index {predicted_index} is out of bounds for labels {CLASS_LABELS}")
        # Instead of returning, maybe print an error state or default value
        print("Classification: Unknown")
        return

    predicted_label = CLASS_LABELS[predicted_index]
    confidence = prediction[0][predicted_index] # Confidence is still available if needed later

    # Print the classification result directly
    print(f"{predicted_label}")

    # Removed buzzer logic
    # print(f"Predicted label: {predicted_label} (Confidence: {confidence:.2f})")
    #
    # # Activate buzzer if "contaminated"
    # if predicted_label == "contaminated":
    #     print("Contamination detected! Activating buzzer.")
    #     GPIO.output(BUZZER_PIN, GPIO.HIGH) # Turn buzzer ON
    #     time.sleep(BUZZ_DURATION)          # Keep it on for the specified duration
    #     GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn buzzer OFF
    #     print("Buzzer deactivated.")
    # else:
    #     print("Classification result: Clean.")

# --- Main Execution ---
def main():
    """Main function to orchestrate the process in a loop (no preview)."""
    # setup_gpio() # Removed GPIO setup call
    # app = QApplication(sys.argv) # Removed Qt application initialization
    picam2 = None # Initialize picam2 variable

    try:
        # Initialize the camera object (outside the loop)
        print("Initializing camera object...")
        picam2 = Picamera2()

        while True: # Loop indefinitely
            input("Press Enter to capture and classify...") # Wait for user input

            # --- Configure and Start Camera for Capture ---
            print("Configuring camera for capture...")
            config = picam2.create_still_configuration(main={"size": CAPTURE_RESOLUTION})
            picam2.configure(config)
            print("Starting camera...")
            picam2.start()
            time.sleep(1) # Allow camera to adjust focus, exposure, etc.

            # --- Capture and Process ---
            pil_image = capture_image(picam2) # Capture the image
            print("Stopping camera...")
            picam2.stop() # Stop the camera stream after capture

            input_data = preprocess_image(pil_image)

            # --- Run Inference ---
            prediction = run_inference(input_data)

            # --- Process Prediction ---
            process_prediction(prediction)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.") # Handle Ctrl+C gracefully
    except ImportError as e:
        print(f"Error importing necessary libraries: {e}")
        # Removed PyQt5 from the message
        print("Please ensure Picamera2, PIL, numpy, and tflite_runtime are installed.")
        print("Example installation: pip install picamera2 Pillow numpy tflite-runtime")
    except FileNotFoundError:
        print(f"Error: The model file '{MODEL_PATH}' was not found.")
        print("Please ensure the TFLite model is in the same directory as the script or provide the correct path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Add specific check for common camera errors
        if "Camera component couldn't be enabled" in str(e) or "failed to import fd" in str(e):
             print("Hint: Ensure the camera is connected and enabled in raspi-config.")
        elif "ENOSPC" in str(e):
             print("Hint: Camera error ENOSPC: Out of resources. Try rebooting the Pi.")

    finally:
        # --- Cleanup ---
        # print("Cleaning up GPIO...") # Removed GPIO cleanup message
        # GPIO.cleanup() # Reset GPIO pin configuration # Removed GPIO cleanup call
        if picam2:
            # Removed preview check
            # if picam2.preview_running:
            #      print("Stopping preview...")
            #      picam2.stop_preview()
            # Ensure camera is stopped if the loop was interrupted while it was running
            if picam2.started:
                 print("Ensuring camera stream is stopped...")
                 picam2.stop()
            # Close might implicitly stop/release, but explicit is good practice
            # picam2.close() # Ensure camera resources are released if it was opened
            print("Camera closed/stopped.")
        # app.quit() # Optional: explicitly quit Qt app if needed
        print("Script finished.")

if __name__ == "__main__":
    main() 