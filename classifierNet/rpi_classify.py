import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import picamera
import time
import io
import os

# --- Configuration ---
MODEL_PATH = 'best_mobilenet_model_quant_int8.tflite'
INPUT_WIDTH, INPUT_HEIGHT = 224, 224
NUM_THREADS = 4
# Class labels must match the order from training (usually alphabetical)
CLASS_LABELS = ['clean', 'grease', 'milk', 'tomato_sauce']

def preprocess_image(image_pil, input_details):
    """Preprocesses PIL image for the TFLite model."""
    # Resize
    image_resized = image_pil.resize((INPUT_WIDTH, INPUT_HEIGHT))
    
    # Convert to NumPy array
    image_data = np.array(image_resized, dtype=np.float32) # Start as float32

    # Get quantization parameters
    input_scale, input_zero_point = input_details['quantization']

    # Quantize the image data
    image_quantized = image_data / input_scale + input_zero_point
    
    # Cast to the required input type (e.g., int8)
    image_final = image_quantized.astype(input_details['dtype'])

    # Add batch dimension
    input_data = np.expand_dims(image_final, axis=0)
    
    return input_data

def dequantize_output(output_data_raw, output_details):
    """Dequantizes the raw output tensor."""
    output_scale, output_zero_point = output_details['quantization']
    # Ensure raw data is float before dequantization calculation
    output_float = (output_data_raw.astype(np.float32) - output_zero_point) * output_scale
    return output_float

def main():
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the quantized model is in the same directory as this script.")
        return

    # --- Load TFLite Model ---
    print(f"Loading model: {MODEL_PATH}")
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
        interpreter.allocate_tensors()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return

    # Get input and output tensor details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Print input/output details (optional, for debugging)
    # print("Input Details:", input_details)
    # print("Output Details:", output_details)
    
    # Verify input type is quantized
    if input_details['dtype'] != np.int8 and input_details['dtype'] != np.uint8:
         print(f"Warning: Expected quantized input type (int8/uint8) but got {input_details['dtype']}.")
         print("Ensure you are using the INT8 quantized model.")
         # Continue anyway, but preprocessing might be incorrect

    # --- Capture Image ---
    print("Initializing camera...")
    try:
        with picamera.PiCamera() as camera:
            # Camera warm-up time
            camera.resolution = (640, 480) # Set a reasonable resolution
            camera.start_preview() # Optional: Display preview
            time.sleep(2) 
            print("Capturing image...")
            
            # Capture to an in-memory stream
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            stream.seek(0) # Rewind stream to the beginning
            
            # Open image using PIL
            img_pil = Image.open(stream).convert('RGB') # Ensure image is RGB
            
            camera.stop_preview() # Optional: Close preview
            print("Image captured.")

    except picamera.exc.PiCameraMMALError:
         print("Error: Camera not detected or enabled.")
         print("Please ensure the camera is connected and enabled in raspi-config.")
         return
    except Exception as e:
        print(f"Error capturing image: {e}")
        return

    # --- Preprocess Image ---
    print("Preprocessing image...")
    try:
        input_data = preprocess_image(img_pil, input_details)
        # print("Input data shape:", input_data.shape) # Debug
        # print("Input data type:", input_data.dtype)  # Debug
    except Exception as e:
         print(f"Error during image preprocessing: {e}")
         return


    # --- Perform Inference ---
    print("Running inference...")
    start_time = time.time()
    
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    # Get raw output
    output_data_raw = interpreter.get_tensor(output_details['index'])[0] # Remove batch dim
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # Time in ms
    print(f"Inference complete ({inference_time:.2f} ms).")
    
    # --- Postprocess and Print Result ---
    # Dequantize output
    output_data_float = dequantize_output(output_data_raw, output_details)
    
    # Get predicted class index
    predicted_index = np.argmax(output_data_float)
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = output_data_float[predicted_index] # This is the softmax probability

    print(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
    # print("Raw Output (quantized):", output_data_raw) # Debug
    # print("Dequantized Output (probabilities):", output_data_float) # Debug

if __name__ == '__main__':
    main()
