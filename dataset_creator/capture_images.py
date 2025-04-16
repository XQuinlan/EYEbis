"""
Script to capture images from Raspberry Pi Camera for dataset creation.
Prompts for a label and saves images sequentially into a folder named after the label.
"""
import picamera
import time
import os

# Configuration
RESOLUTION = (1024, 768)
ROTATION = 180  # Adjust as needed for your camera mounting (0, 90, 180, 270)
BASE_SAVE_DIR = "dataset"  # Base directory to save labeled folders
PREVIEW_ALPHA = 200 # Preview window transparency (0-255)


def capture_images():
    """Handles camera setup, preview, and image capture loop."""
    label = input("Enter the class label for these images (e.g., grease, clean): ").strip().lower()
    if not label:
        print("Error: Label cannot be empty.")
        return

    save_dir = os.path.join(BASE_SAVE_DIR, label)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving images to: {save_dir}")

    image_count = 0
    # Find the highest existing image number in the directory to avoid overwriting
    try:
        existing_files = [f for f in os.listdir(save_dir) if f.startswith(f"{label}_") and f.endswith(".jpg")]
        if existing_files:
            numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            image_count = max(numbers)
            print(f"Starting image count from {image_count + 1}")
    except Exception as e:
        print(f"Warning: Could not determine starting image count. Starting from 1. Error: {e}")
        image_count = 0

    try:
        with picamera.PiCamera() as camera:
            camera.resolution = RESOLUTION
            camera.rotation = ROTATION
            # camera.framerate = 15 # Optional: Can be set if needed

            print("\nStarting camera preview. Position the object.")
            print("Press ENTER to capture an image.")
            print("Press Ctrl+C or Q then ENTER to quit.")

            # Allow camera to warm up
            time.sleep(2)

            camera.start_preview(alpha=PREVIEW_ALPHA)

            while True:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        break

                    # Capture image
                    image_count += 1
                    filename = f"{label}_{image_count:04d}.jpg"
                    filepath = os.path.join(save_dir, filename)

                    # Optional: Add a small delay for auto adjustments before capture
                    # time.sleep(1) 

                    camera.capture(filepath)
                    print(f"Captured: {filepath}")

                except KeyboardInterrupt:
                    break
                except EOFError: # Handles case where input stream is closed (e.g., redirected input)
                    break
                except Exception as e:
                    print(f"An error occurred during capture: {e}")
                    time.sleep(1) # Avoid rapid error loops

            camera.stop_preview()

    except picamera.exc.PiCameraMMALError:
        print("Error: Camera module not found or enabled.")
        print("Ensure the camera is connected and enabled in raspi-config.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("\nScript finished.")

if __name__ == "__main__":
    capture_images()
