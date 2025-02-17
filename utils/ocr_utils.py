# utils/ocr_utils.py
import base64
import google.generativeai as genai
from config import GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)

def perform_ocr(model, image_path, prompt):
    """Perform OCR on a single image."""
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    response = model.generate_content(
        [
            {'mime_type': 'image/png', 'data': encoded_image},
            prompt
        ]
    )

    return response.text