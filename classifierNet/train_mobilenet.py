import tensorflow as tf
import os # Keep os import

# --- Force CPU (for debugging GPU issues) ---
try:
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
    print("GPU disabled, forcing CPU execution.")
except Exception as e:
    print("Could not disable GPU, proceeding anyway. Error:", e)
# ---------------------------------------------

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse # Import argparse

# Constants - Keep ones not handled by CLI args
IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 4 # This might need to be dynamic based on train_dir in the future
MODEL_SAVE_PATH = 'best_mobilenet_model.keras' # Use .keras format
# Removed EPOCHS, BATCH_SIZE, TRAIN_DIR, VALID_DIR as they will be CLI args

def build_model(input_shape, num_classes):
    """Builds the MobileNetV2-based classification model."""
    # Load MobileNetV2 base model (pre-trained on ImageNet)
    # include_top=False removes the final classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x) # Intermediate dense layer
    predictions = Dense(num_classes, activation='softmax')(x) # Final output layer

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Modified main to accept arguments
def main(epochs, batch_size, learning_rate, train_dir, valid_dir):
    """Main function to set up data generators, compile, and train the model."""
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    # --- Data Preparation --- (Using args)
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Rescale pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation data generator (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir, # Use arg
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size, # Use arg
        class_mode='categorical' # for multi-class classification
    )

    # Flow validation images in batches using validation_datagen generator
    validation_generator = validation_datagen.flow_from_directory(
        valid_dir, # Use arg
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size, # Use arg
        class_mode='categorical',
        shuffle=False # No need to shuffle validation data
    )

    # Verify class indices
    print("Class Indices:", train_generator.class_indices)
    # Dynamically determine NUM_CLASSES from the generator
    actual_num_classes = len(train_generator.class_indices)
    if actual_num_classes == 0:
        print(f"Error: No classes found in {train_dir}. Please ensure subdirectories exist.")
        return
    print(f"Found {actual_num_classes} classes.")

    # --- Model Building and Compilation --- (Using args)
    # Build model using the dynamically found number of classes
    model = build_model(input_shape, actual_num_classes)

    model.compile(optimizer=Adam(learning_rate=learning_rate), # Use arg
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Print model summary

    # --- Callbacks ---
    # Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=False,
        mode='max'
    )

    callbacks_list = [checkpoint]

    # --- Model Training --- (Using args)
    print(f"Starting training for {epochs} epochs...")

    # Check if data generators are empty
    if not train_generator.samples:
        print(f"Error: No training images found in {train_dir}. Please add images.")
        return
    if not validation_generator.samples:
        print(f"Error: No validation images found in {valid_dir}. Please add images.")
        return

    # Calculate steps per epoch carefully to avoid division by zero
    steps_per_epoch = train_generator.samples // batch_size if batch_size > 0 else 0
    validation_steps = validation_generator.samples // batch_size if batch_size > 0 else 0

    if steps_per_epoch == 0:
        print(f"Warning: batch_size ({batch_size}) is larger than the number of training samples ({train_generator.samples}). Training cannot proceed.")
        return
    if validation_steps == 0:
        print(f"Warning: batch_size ({batch_size}) is larger than the number of validation samples ({validation_generator.samples}). Validation may not run correctly.")
        # Allow training to proceed, but validation might be skipped or incomplete

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs, # Use arg
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list
    )

    # --- Explicitly save the final model ---
    print(f"Explicitly saving final model to {MODEL_SAVE_PATH}...")
    try:
        model.save(MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model explicitly: {e}")
    # -----------------------------------------

    print(f"Training process finished.") # Changed wording slightly

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a MobileNetV2 image classification model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--train_dir', type=str, default='data/train', help='Path to the training data directory.')
    parser.add_argument('--valid_dir', type=str, default='data/validation', help='Path to the validation data directory.')

    # Parse arguments
    args = parser.parse_args()

    # Ensure the data directories exist before calling main
    if not os.path.exists(args.train_dir) or not os.path.isdir(args.train_dir):
        print(f"Error: Training directory '{args.train_dir}' not found or is not a directory.")
    elif not os.path.exists(args.valid_dir) or not os.path.isdir(args.valid_dir):
         print(f"Error: Validation directory '{args.valid_dir}' not found or is not a directory.")
    else:
        # Call main with parsed arguments
        main(args.epochs, args.batch_size, args.learning_rate, args.train_dir, args.valid_dir)
