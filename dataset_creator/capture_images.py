"""
Script to capture images from Raspberry Pi Camera for dataset creation.
Prompts for a label and saves images sequentially into a folder named after the label.
Uses picamera2 library (libcamera backend).
"""
# import picamera
from picamera2 import Picamera2, Preview
from libcamera import Transform
import time
import os

# Configuration
RESOLUTION = (1024, 768)
ROTATION = 180  # Adjust as needed for your camera mounting (0, 90, 180, 270)
BASE_SAVE_DIR = "dataset"  # Base directory to save labeled folders
# PREVIEW_ALPHA = 200 # Preview window transparency (0-255) - Alpha not directly supported in picamera2 preview same way

def capture_images():
    """Handles camera setup, preview, and image capture loop using picamera2."""
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

    picam2 = None # Initialize picam2 to None for finally block
    try:
        picam2 = Picamera2()

        # Configure transform based on rotation
        transform = Transform()
        if ROTATION == 180:
            transform = Transform(hflip=1, vflip=1)
        elif ROTATION == 90:
            # Note: Precise 90/270 rotation might depend on sensor orientation vs. libcamera interpretation
            # This is a common mapping. Adjust if necessary.
            transform = Transform(hflip=1, transpose=1) 
        elif ROTATION == 270:
             transform = Transform(vflip=1, transpose=1)

        # Create camera configuration
        # Adjust 'queue=False' and buffer_count if experiencing delays or memory issues
        config = picam2.create_still_configuration(
            main={"size": RESOLUTION},
            lores={"size": (640, 480)}, # Low-res stream for preview efficiency
            display="lores", # Use low-res stream for preview window
            transform=transform,
            buffer_count=3 # Increase buffer count slightly for potentially smoother capture
        )
        picam2.configure(config)

        print("\nStarting camera preview. Position the object.")
        print("Press ENTER to capture an image.")
        print("Press Q then ENTER to quit.")

        # Start preview (using QtGL backend, ensure Qt dependencies are installed if needed)
        # For headless operation, use Preview.DRM or Preview.NULL
        picam2.start_preview(Preview.QTGL) 
        
        picam2.start() # Start the camera stream
        time.sleep(2) # Allow camera to warm up

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
                # time.sleep(1) # Picamera2 might handle adjustments faster

                # Capture and save the image
                metadata = picam2.capture_file(filepath)
                print(f"Captured: {filepath}")

            except KeyboardInterrupt:
                break
            except EOFError: # Handles case where input stream is closed (e.g., redirected input)
                break
            except Exception as e:
                print(f"An error occurred during capture: {e}")
                time.sleep(1) # Avoid rapid error loops

        # No stop_preview() or stop() here, handled in finally

    except Exception as e: # General exception handler
        print(f"An unexpected error occurred: {e}")
        if "Camera component couldn't be enabled" in str(e) or "failed to import fd" in str(e):
             print("Hint: Ensure the camera is connected, enabled in raspi-config (Interface Options -> Camera),")
             print("      and that 'libcamerautils' and necessary Qt libraries (e.g., python3-pyqt5) are installed.")
        elif "ENOSPC" in str(e):
             print("Hint: Camera error ENOSPC: Out of resources. Try rebooting the Pi.")

    finally:
        if picam2:
            if picam2.preview_running:
                 picam2.stop_preview()
            if picam2.started:
                 picam2.stop()
            # picam2.close() # Close is often implicitly handled, but can be added if needed
        print("\nScript finished.")

if __name__ == "__main__":
    capture_images()
