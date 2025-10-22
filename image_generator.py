"""
Image generation module for the Image Styler application.
Handles all Replicate API interactions for image styling.
"""
import replicate
from logger_config import setup_logger

logger = setup_logger()


def generate_styled_image(prompt, uploaded_file, model, api_token):
    """
    Generates a styled image using the Replicate API.
    
    Args:
        prompt: The style prompt for the image
        uploaded_file: The uploaded file object
        model: The Replicate model to use
        api_token: Replicate API token
        
    Returns:
        tuple: (success: bool, result: str or None, error_message: str or None)
    """
    if not uploaded_file:
        logger.warning("Image generation attempted without an uploaded file.")
        return False, None, "Please upload an image first."
    
    if not prompt:
        logger.warning("Image generation attempted without a prompt.")
        return False, None, "Please enter a prompt."
    
    try:
        # Ensure the file pointer is at the beginning
        uploaded_file.seek(0)
        
        # Log model and prompt details
        logger.info(f"Calling Replicate API. Model: {model}, Prompt: '{prompt}'")
        
        api_output = replicate.run(
            model,
            input={
                "prompt": prompt,
                "input_image": uploaded_file,
            }
        )
        
        styled_image_url = None
        if api_output:
            url_candidate = str(api_output)
            if url_candidate.startswith("http"):
                styled_image_url = url_candidate
        
        if styled_image_url:
            logger.info(f"Successfully generated styled image. URL: {styled_image_url}")
            return True, styled_image_url, None
        else:
            logger.warning(f"Image generation did not return a valid URL. API output: {api_output}")
            error_msg = f"Image generation returned an unexpected result. Output: {api_output}" if api_output else "Image generation failed and no output was received."
            return False, None, error_msg
            
    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API call unsuccessful: {e}", exc_info=True)
        return False, None, f"Replicate API Error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during image generation: {e}", exc_info=True)
        return False, None, f"An unexpected error occurred: {e}"
