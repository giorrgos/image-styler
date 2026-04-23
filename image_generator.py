"""
Image generation module for the Image Styler application.
Handles all Replicate API interactions for image styling.
"""
import time
import replicate
from logger_config import setup_logger

logger = setup_logger()

# Import telemetry after logger to avoid circular imports
try:
    from telemetry_config import (
        tracer, 
        api_calls_counter, 
        api_errors_counter,
        api_duration_histogram
    )
    TELEMETRY_ENABLED = True
except ImportError:
    TELEMETRY_ENABLED = False
    logger.warning("Telemetry not available - running without instrumentation")


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
    if TELEMETRY_ENABLED:
        with tracer.start_as_current_span("replicate_api_call") as span:
            span.set_attribute("model", model)
            span.set_attribute("prompt_length", len(prompt) if prompt else 0)
            return _generate_styled_image_impl(prompt, uploaded_file, model, api_token, span)
    else:
        return _generate_styled_image_impl(prompt, uploaded_file, model, api_token, None)


def _generate_styled_image_impl(prompt, uploaded_file, model, api_token, span=None):
    """Implementation of image generation with optional telemetry."""
    start_time = time.time()
    
    if not uploaded_file:
        logger.warning("Image generation attempted without an uploaded file.")
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error_type", "no_file")
        return False, None, "Please upload an image first."
    
    if not prompt:
        logger.warning("Image generation attempted without a prompt.")
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error_type", "no_prompt")
        return False, None, "Please enter a prompt."
    
    try:
        # Ensure the file pointer is at the beginning
        uploaded_file.seek(0)
        
        # Log model and prompt details
        logger.info(f"Calling Replicate API. Model: {model}, Prompt: '{prompt}'")
        
        # Build input payload based on model's expected parameter names
        if model.startswith("qwen/"):
            model_input = {"prompt": prompt, "image": [uploaded_file]}
        else:
            model_input = {"prompt": prompt, "input_image": uploaded_file}

        api_output = replicate.run(model, input=model_input)

        # Calculate duration
        duration = time.time() - start_time
        if TELEMETRY_ENABLED:
            api_duration_histogram.record(duration, {"model": model})

        # flux returns a single URI string; qwen returns a list of URIs
        styled_image_url = None
        if api_output:
            if isinstance(api_output, list):
                url_candidate = str(api_output[0]) if api_output else None
            else:
                url_candidate = str(api_output)
            if url_candidate and url_candidate.startswith("http"):
                styled_image_url = url_candidate
        
        if styled_image_url:
            logger.info(f"Successfully generated styled image. URL: {styled_image_url}")
            
            if span:
                span.set_attribute("success", True)
                span.set_attribute("duration_seconds", duration)
            
            if TELEMETRY_ENABLED:
                api_calls_counter.add(1, {"model": model, "status": "success"})
            
            return True, styled_image_url, None
        else:
            logger.warning(f"Image generation did not return a valid URL. API output: {api_output}")
            error_msg = f"Image generation returned an unexpected result. Output: {api_output}" if api_output else "Image generation failed and no output was received."
            
            if span:
                span.set_attribute("success", False)
                span.set_attribute("error_type", "invalid_output")
            
            if TELEMETRY_ENABLED:
                api_calls_counter.add(1, {"model": model, "status": "failure"})
                api_errors_counter.add(1, {"model": model, "error_type": "invalid_output"})

            return False, None, error_msg

    except replicate.exceptions.ReplicateError as e:
        duration = time.time() - start_time
        logger.error(f"Replicate API call unsuccessful: {e}", exc_info=True)

        if span:
            span.set_attribute("success", False)
            span.set_attribute("error", True)
            span.set_attribute("error_type", "replicate_error")
            span.set_attribute("error_message", str(e))
            span.set_attribute("duration_seconds", duration)

        if TELEMETRY_ENABLED:
            api_calls_counter.add(1, {"model": model, "status": "failure"})
            api_errors_counter.add(1, {"model": model, "error_type": "replicate_error"})
            api_duration_histogram.record(duration, {"model": model})

        return False, None, f"Replicate API Error: {e}"

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"An unexpected error occurred during image generation: {e}", exc_info=True)

        if span:
            span.set_attribute("success", False)
            span.set_attribute("error", True)
            span.set_attribute("error_type", "unexpected_error")
            span.set_attribute("error_message", str(e))
            span.set_attribute("duration_seconds", duration)

        if TELEMETRY_ENABLED:
            api_calls_counter.add(1, {"model": model, "status": "failure"})
            api_errors_counter.add(1, {"model": model, "error_type": "unexpected_error"})
            api_duration_histogram.record(duration, {"model": model})
        
        return False, None, f"An unexpected error occurred: {e}"
