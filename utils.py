import os
from pathlib import Path

# The supported formats for images
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}

def validate_image_path(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        return False, "File does not exist."
    
    # Check if the format is supported
    extension = Path(image_path).suffix.lower()
    if extension not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported formats are: {', '.join(SUPPORTED_FORMATS)}."
    
    return True, "Valid"

def generate_output_path(input_path):
    # Separating the path into names and extensions
    path = Path(input_path)

    # Adding "_protected" before the extension
    output_path = path.parent / f"{path.stem}_protected{path.suffix}"

    return str(output_path)