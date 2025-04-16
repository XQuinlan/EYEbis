from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """Preprocesses an image for MobileNetV2.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array with shape 
                    (1, 224, 224, 3) and values normalized between 0 and 1.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1]
    normalized_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    preprocessed_array = np.expand_dims(normalized_array, axis=0)

    return preprocessed_array 