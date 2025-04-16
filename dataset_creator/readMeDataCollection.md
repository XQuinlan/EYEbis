# Data Collection Script (`capture_images.py`)

This script uses a Raspberry Pi with a connected PiCamera (specifically tested with the NoIR v2) to capture images for creating an image dataset. It prompts the user for a class label and saves captured images into a structured directory.

## Prerequisites

1.  **Raspberry Pi:** Any model with a camera connector.
2.  **PiCamera:** A compatible Raspberry Pi camera module (e.g., Camera Module v1/v2/v3, NoIR Camera, High Quality Camera).
3.  **Python 3:** Usually pre-installed on Raspberry Pi OS.
4.  **`picamera` Library:** The Python interface for the camera.
5.  **Camera Enabled:** The camera interface must be enabled in the Raspberry Pi configuration.

## Setup

1.  **Enable Camera:**
    *   Open a terminal on your Raspberry Pi.
    *   Run `sudo raspi-config`.
    *   Navigate to `Interface Options` -> `Camera`.
    *   Select `Yes` to enable the camera interface.
    *   Select `Finish` and reboot if prompted.

2.  **Install Dependencies:**
    *   **Ensure Python 3 is installed:** Python 3 is usually pre-installed on Raspberry Pi OS. You can check by running `python3 --version`. If it's not installed, you might need to run `sudo apt-get update && sudo apt-get install python3`.
    *   **Install `picamera` Library:** Open a terminal and run the following commands:
        ```bash
        # Update package list
        sudo apt-get update
        # Install the Python 3 picamera library
        sudo apt-get install python3-picamera
        ```
    *   *(Alternative for pip/venv users): If you are using a virtual environment, you might prefer:*
        ```bash
        pip install "picamera[array]"
        ```

## Running the Script

1.  Navigate to the directory containing `capture_images.py` in the terminal.
2.  Run the script using Python 3:
    ```bash
    python capture_images.py
    ```

## Usage

1.  **Enter Label:** When prompted, type the class label for the images you are about to capture (e.g., `grease`, `tomato_sauce`, `milk`, `clean`) and press Enter. The label should describe the object or condition being photographed.
2.  **Camera Preview:** A preview window will appear, showing the camera's live feed. Position the object you want to photograph under the camera.
3.  **Capture Image:** Press `Enter` to capture an image. The script will save the image and print the filename.
4.  **Capture More:** Continue pressing `Enter` to capture more images for the *same* label.
5.  **Quit:** To stop capturing, press `Q` and then `Enter`, or press `Ctrl+C`.
6.  **New Label:** To capture images for a different label, simply run the script again (`python capture_images.py`) and enter the new label when prompted.

## Output

*   The script creates a base directory named `dataset` (if it doesn't exist) in the same location where the script is run.
*   Inside `dataset`, it creates subdirectories named after the labels you provide (e.g., `dataset/grease/`, `dataset/clean/`).
*   Images are saved within their respective label directories.
*   Filenames are sequential, formatted as `label_####.jpg` (e.g., `grease_0001.jpg`, `grease_0002.jpg`). The script checks for existing images to avoid overwriting and continues numbering from the highest existing number for that label.

## Configuration (Optional)

You can modify the following constants at the top of the `capture_images.py` script:

*   `RESOLUTION`: Set the image resolution (width, height) in pixels. Default is `(1024, 768)`.
*   `ROTATION`: Adjust the camera rotation if the image appears upside down or sideways. Accepts values `0`, `90`, `180`, `270`. Default is `180`, suitable for many top-down mounts.
*   `BASE_SAVE_DIR`: Change the name of the main directory where labeled folders are stored. Default is `"dataset"`.
*   `PREVIEW_ALPHA`: Controls the transparency of the preview window (0=invisible, 255=opaque). Default is `200`. 