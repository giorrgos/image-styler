"""
Image processing utilities for the Image Styler application.
Handles image manipulation tasks like EXIF orientation fixing.
"""
from PIL import Image, ImageOps
from io import BytesIO
from logger_config import setup_logger

logger = setup_logger()

# Import telemetry after logger to avoid circular imports
try:
    from telemetry_config import tracer
    TELEMETRY_ENABLED = True
except ImportError:
    TELEMETRY_ENABLED = False
    logger.warning("Telemetry not available - running without instrumentation")


def fix_image_orientation(uploaded_file):
    """
    Fix image orientation based on EXIF data.
    Returns a PIL Image with correct orientation.
    
    Args:
        uploaded_file: File-like object (from Streamlit file uploader)
        
    Returns:
        PIL.Image: Image with correct orientation
    """
    if TELEMETRY_ENABLED:
        with tracer.start_as_current_span("fix_orientation") as span:
            return _fix_image_orientation_impl(uploaded_file, span)
    else:
        return _fix_image_orientation_impl(uploaded_file, None)


def _fix_image_orientation_impl(uploaded_file, span=None):
    """Implementation of image orientation fixing with optional telemetry."""
    try:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Check if orientation was changed
        original_size = image.size
        
        # Use ImageOps.exif_transpose to automatically fix orientation based on EXIF data
        # This handles all orientation cases (rotations and flips)
        image = ImageOps.exif_transpose(image)
        
        orientation_changed = (original_size != image.size)
        
        if span:
            span.set_attribute("orientation_fixed", orientation_changed)
            span.set_attribute("image_width", image.width)
            span.set_attribute("image_height", image.height)
        
        logger.debug("Successfully fixed image orientation")
        return image
    except Exception as e:
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error_message", str(e))
        
        logger.warning(f"Could not fix image orientation: {e}")
        # If something goes wrong, just return the original
        uploaded_file.seek(0)
        return Image.open(uploaded_file)


def pil_image_to_bytes(image, format='PNG'):
    """
    Convert a PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        BytesIO: Image as bytes in a file-like object
    """
    img_bytes = BytesIO()
    image.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes

