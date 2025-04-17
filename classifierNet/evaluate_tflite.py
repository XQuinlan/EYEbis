import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# --- Configuration ---
# Select the quantized model to evaluate
MODEL_PATH = 'best_mobilenet_model_quant_float16.tflite'
TEST_DATA_DIR = 'processed_dataset/validation' # Use validation set as test set, or create a separate test set
INPUT_WIDTH, INPUT_HEIGHT = 224, 224
NUM_THREADS = 4
# Class labels must match the order used during training/generation
# Usually alphabetical if using flow_from_directory
CLASS_LABELS = sorted(os.listdir(TEST_DATA_DIR)) # Infer from directory names


def preprocess_image(image_path, input_details):
    """Loads and preprocesses an image file for the TFLite model."""
    try:
        img_pil = Image.open(image_path).convert('RGB')
        # Resize
        image_resized = img_pil.resize((INPUT_WIDTH, INPUT_HEIGHT))
        
        # Convert to NumPy array
        image_data = np.array(image_resized, dtype=np.float32) # Start as float32

        # Get quantization parameters if the model is quantized
        if 'quantization' in input_details and input_details['quantization'] != (0.0, 0):
            input_scale, input_zero_point = input_details['quantization']
            # Quantize the image data
            image_quantized = image_data / input_scale + input_zero_point
            # Cast to the required input type (e.g., int8)
            image_final = image_quantized.astype(input_details['dtype'])
        else: # Handle non-quantized models (e.g., float32 or float16)
            # Normalize if it's a standard float model (like MobileNet expects)
            image_normalized = image_data / 255.0 # Or specific normalization if needed
            image_final = image_normalized.astype(input_details['dtype'])

        # Add batch dimension
        input_data = np.expand_dims(image_final, axis=0)
        return input_data
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def dequantize_output(output_data_raw, output_details):
    """Dequantizes the raw output tensor if needed."""
    if 'quantization' in output_details and output_details['quantization'] != (0.0, 0):
        output_scale, output_zero_point = output_details['quantization']
        # Ensure raw data is float before dequantization calculation
        output_float = (output_data_raw.astype(np.float32) - output_zero_point) * output_scale
        return output_float
    else:
        # If output is already float (float32/float16), return as is
        return output_data_raw.astype(np.float32)

def evaluate_model():
    # --- Validate Paths ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.exists(TEST_DATA_DIR) or not os.listdir(TEST_DATA_DIR):
        print(f"Error: Test data directory '{TEST_DATA_DIR}' not found or is empty.")
        return
    
    global CLASS_LABELS
    CLASS_LABELS = sorted([d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))])
    if not CLASS_LABELS:
        print(f"Error: No class subdirectories found in {TEST_DATA_DIR}")
        return
    print(f"Found classes: {CLASS_LABELS}")

    # --- Load TFLite Model ---
    print(f"Loading model: {MODEL_PATH}")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
        interpreter.allocate_tensors()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # --- Evaluate on Test Data ---
    print(f"Evaluating model on data from: {TEST_DATA_DIR}")
    all_true_labels = []
    all_pred_labels = []
    inference_times = []
    processed_images = 0

    for class_index, class_name in enumerate(CLASS_LABELS):
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        image_paths = (
            glob.glob(os.path.join(class_dir, '*.jpg')) + 
            glob.glob(os.path.join(class_dir, '*.jpeg')) + 
            glob.glob(os.path.join(class_dir, '*.png'))
        )
        
        if not image_paths:
            print(f"Warning: No images found in {class_dir}")
            continue
            
        print(f"Processing class: {class_name} ({len(image_paths)} images)")
        for image_path in image_paths:
            input_data = preprocess_image(image_path, input_details)
            if input_data is None: # Skip if preprocessing failed
                continue

            start_time = time.time()
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            output_data_raw = interpreter.get_tensor(output_details['index'])[0]
            end_time = time.time()
            inference_times.append(end_time - start_time)

            # Dequantize and get prediction
            output_data_float = dequantize_output(output_data_raw, output_details)
            predicted_index = np.argmax(output_data_float)
            
            all_true_labels.append(class_index)
            all_pred_labels.append(predicted_index)
            processed_images += 1
            
            # Optional: print progress periodically
            if processed_images % 50 == 0:
                print(f"  Processed {processed_images} images...")

    if processed_images == 0:
        print("Error: No images were successfully processed. Check image paths and formats.")
        return

    # --- Calculate and Print Metrics ---
    print("\n--- Evaluation Results ---")
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # Use numeric indices for report calculation, then map to labels
    report = classification_report(all_true_labels, all_pred_labels, 
                                   target_names=CLASS_LABELS)
    print(report)

    print("Confusion Matrix:")
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    print(cm)

    avg_inference_time = np.mean(inference_times) * 1000 # milliseconds
    print(f"\nAverage Inference Time: {avg_inference_time:.2f} ms per image")

if __name__ == '__main__':
    evaluate_model()
