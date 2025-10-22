import streamlit as st
import replicate
import os
import time
from dotenv import load_dotenv
from logger_config import setup_logger
from image_generator import generate_styled_image
from image_utils import fix_image_orientation, pil_image_to_bytes

# --- Logger Setup ---
logger = setup_logger()

# --- Initialize Telemetry (before everything else) ---
try:
    from telemetry_config import setup_telemetry, tracer, images_processed_counter, request_duration_histogram
    TELEMETRY_ENABLED = setup_telemetry()
    if TELEMETRY_ENABLED:
        logger.info("Telemetry enabled and initialized")
except ImportError:
    TELEMETRY_ENABLED = False
    logger.warning("Telemetry not available - running without instrumentation")

# --- Environment and Model Setup ---
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MODEL = "black-forest-labs/flux-kontext-pro"

# Validate configuration
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN not found in .env file or environment variables.")
    st.error("REPLICATE_API_TOKEN not found. Please set it in your .env file.")
    st.stop()

try:
    replicate.Client(api_token=REPLICATE_API_TOKEN)
    logger.info("Replicate client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Replicate client: {e}", exc_info=True)
    st.error(f"Failed to initialize Replicate client: {e}")
    st.stop()

# --- Streamlit app setup ---
# Log the application starts
logger.info("Application started")
st.set_page_config(page_title="Image Styler", layout="wide")
st.title("Giorgos' Styling Salon")

def _process_image_request(corrected_image, uploaded_file, prompt, model, 
                          api_token, col2, span, request_start_time):
    """Process the image styling request with optional telemetry."""
    with st.spinner("Generating your image..."):
        # Convert corrected PIL image to bytes for API
        if corrected_image:
            image_bytes = pil_image_to_bytes(corrected_image)
            success, image_url, error_message = generate_styled_image(prompt, image_bytes, model, api_token)
        else:
            success, image_url, error_message = generate_styled_image(prompt, uploaded_file, model, api_token)
        
        # Calculate total request duration
        request_duration = time.time() - request_start_time
        
        if success:
            with col2:
                st.image(image_url, caption="Styled Image", use_container_width=True)
            
            # Record success metrics
            if span:
                span.set_attribute("success", True)
                span.set_attribute("duration_seconds", request_duration)
            
            if TELEMETRY_ENABLED:
                images_processed_counter.add(1, {"model": model, "status": "success"})
                request_duration_histogram.record(request_duration, {"model": model, "status": "success"})
        else:
            if error_message and error_message.startswith("Please"):
                st.warning(error_message)
            else:
                st.error(error_message)
            
            # Record failure metrics
            if span:
                span.set_attribute("success", False)
                span.set_attribute("error_message", error_message)
                span.set_attribute("duration_seconds", request_duration)
            
            if TELEMETRY_ENABLED:
                request_duration_histogram.record(request_duration, {"model": model, "status": "failure"})

# --- File uploader at the top ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="uploader")

# Corrected logic for logging image upload/removal
if uploaded_file is not None:
    # A file is currently uploaded
    if 'last_uploaded_filename' not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        # Log the image is uploaded (log only on new upload or change)
        size_in_bytes = uploaded_file.size
        size_in_mb = size_in_bytes / (1024 * 1024) # Convert bytes to MB
        logger.info(f"Image uploaded: {uploaded_file.name}, Type: {uploaded_file.type}, Size: {size_in_mb:.2f} MB")
        st.session_state.last_uploaded_filename = uploaded_file.name
    # If 'last_uploaded_filename' exists and is the same as uploaded_file.name, no new log needed.
elif uploaded_file is None:
    # No file is currently uploaded
    if 'last_uploaded_filename' in st.session_state:
        # This means a file was just removed (or cleared)
        logger.info(f"Uploaded image removed: {st.session_state.last_uploaded_filename}") # Corrected this line
        del st.session_state.last_uploaded_filename


# --- Create two columns for image display ---
col1, col2 = st.columns(2)

# Fix orientation once and reuse for both display and generation
corrected_image = None
if uploaded_file:
    corrected_image = fix_image_orientation(uploaded_file)

# --- Original image display in the first column ---
with col1:
    st.header("Original Image")
    if corrected_image:
        st.image(corrected_image, caption="Uploaded Image", use_container_width=True)

# --- Styled image display area in the second column ---
with col2:
    st.header("Styled Image")

# --- Prompt input below the images ---
prompt = st.text_input("Enter a style prompt (e.g. 'cyberpunk style portrait', 'ghibli studio')")

# --- "Generate Styled Image" button below the prompt ---
if st.button("Generate Styled Image"):
    # Create root span for the entire request
    request_start_time = time.time()
    
    if TELEMETRY_ENABLED:
        with tracer.start_as_current_span("image_styling_request") as span:
            # Add request metadata
            span.set_attribute("has_file", corrected_image is not None)
            span.set_attribute("has_prompt", bool(prompt))
            span.set_attribute("model", MODEL)
            
            if uploaded_file:
                span.set_attribute("filename", uploaded_file.name)
                span.set_attribute("file_size_mb", uploaded_file.size / (1024 * 1024))
                span.set_attribute("file_type", uploaded_file.type)
            
            _process_image_request(
                corrected_image, uploaded_file, prompt, MODEL, 
                REPLICATE_API_TOKEN, col2, span, request_start_time
            )
    else:
        _process_image_request(
            corrected_image, uploaded_file, prompt, MODEL,
            REPLICATE_API_TOKEN, col2, None, request_start_time
        )
