import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# --- Configuration ---
KERAS_MODEL_PATH = 'best_mobilenet_model.keras'
QUANTIZED_TFLITE_MODEL_PATH_INT8 = 'best_mobilenet_model_quant_int8.tflite'
QUANTIZED_TFLITE_MODEL_PATH_FLOAT16 = 'best_mobilenet_model_quant_float16.tflite' # Optional path for float16
VALID_DIR = 'processed_dataset/validation'
IMG_WIDTH, IMG_HEIGHT = 224, 224 # Should match the training input size
BATCH_SIZE = 1 # Batch size for the representative dataset generator
NUM_CALIBRATION_STEPS = 100 # Number of samples for representative dataset

def representative_dataset_gen():
    """Generates a representative dataset from the validation directory."""
    # Use ImageDataGenerator to load and preprocess images like in validation
    datagen = ImageDataGenerator(rescale=1./255) # Only rescale
    
    generator = datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical', # Doesn't matter for quantization data, but needed by flow_from_directory
        shuffle=True # Shuffle to get diverse samples
    )

    print(f"Providing {NUM_CALIBRATION_STEPS} samples for calibration...")
    for i in range(NUM_CALIBRATION_STEPS):
        try:
            batch_images, _ = next(generator)
            # Ensure the data type is float32, as required by the converter
            yield [batch_images.astype(np.float32)]
        except StopIteration:
            print(f"Warning: Ran out of validation images after {i} steps. Used available images for calibration.")
            break
        except Exception as e:
            print(f"Error during data generation: {e}")
            break

def quantize_model(quant_type='int8'):
    """Performs post-training quantization."""
    
    # Check if Keras model exists
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"Error: Keras model file not found at {KERAS_MODEL_PATH}")
        return

    # Check if validation directory exists
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR):
         print(f"Error: Validation directory '{VALID_DIR}' not found or is empty. Cannot create representative dataset.")
         return

    print(f"Loading Keras model from: {KERAS_MODEL_PATH}")
    # Load the trained Keras model
    try:
        model = load_model(KERAS_MODEL_PATH)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    print("Initializing TFLiteConverter...")
    # Initialize the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- Configure Quantization ---
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quant_type == 'int8':
        print("Configuring for INT8 quantization...")
        # Full Integer Quantization (INT8)
        # Requires a representative dataset
        converter.representative_dataset = representative_dataset_gen
        # Ensure integer quantization for inputs and outputs
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8 depending on your model/preference
        converter.inference_output_type = tf.int8 # or tf.uint8
        output_path = QUANTIZED_TFLITE_MODEL_PATH_INT8

    elif quant_type == 'float16':
        print("Configuring for FLOAT16 quantization...")
        # Float16 Quantization
        # Does not require a representative dataset
        converter.target_spec.supported_types = [tf.float16]
        output_path = QUANTIZED_TFLITE_MODEL_PATH_FLOAT16
        
    else:
        print(f"Error: Unsupported quantization type '{quant_type}'. Choose 'int8' or 'float16'.")
        return

    # --- Convert Model ---
    print("Starting TFLite conversion and quantization...")
    try:
        tflite_quant_model = converter.convert()
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        # Provide more specific advice if possible
        if "representative_dataset" in str(e):
             print("This might be related to the representative dataset generator. Check image paths and formats.")
        return

    # --- Save Quantized Model ---
    try:
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
        print(f"Quantized ({quant_type}) model saved successfully to: {output_path}")
        print(f"Original model size: {os.path.getsize(KERAS_MODEL_PATH) / (1024*1024):.2f} MB")
        print(f"Quantized model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error saving quantized model: {e}")


if __name__ == '__main__':
    # --- Choose Quantization Type ---
    # Change this to 'float16' for float16 quantization
    quantization_type = 'float16'
    
    quantize_model(quantization_type)
