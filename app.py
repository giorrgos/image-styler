import streamlit as st
import os
import logging
from dotenv import load_dotenv
from PIL import Image
import io
import replicate
from langfuse import Langfuse
from langfuse.decorators import observe
import requests
import base64

# Load environment variables
load_dotenv()

# Model constants - Use image-to-image model instead
#FLUX_MODEL = "black-forest-labs/flux-1.1-pro-ultra"  # Better for image-to-image
FLUX_MODEL = "timothybrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f"
#FLUX_MODEL = "timbrooks/instruct-pix2pix"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize clients
@st.cache_resource
def get_replicate_client():
    return replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

@st.cache_resource
def get_langfuse_client():
    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )

def encode_image_to_base64(image):
    """Convert PIL image to base64 data URL."""
    logger.info("Converting image to base64 format")
    buffered = io.BytesIO()
    # Ensure RGB format for consistency
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
        logger.info(f"Converted image from {image.mode} to RGB")
    image.save(buffered, format="JPEG", quality=95)  # Higher quality
    img_str = base64.b64encode(buffered.getvalue()).decode()
    data_url = f"data:image/jpeg;base64,{img_str}"
    logger.info("Successfully converted image to base64 data URL")
    return data_url

def main():
    st.set_page_config(
        page_title="Image Styler",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Yorgos' Styling Salon")
    st.markdown("Upload an image and describe how you want it styled while preserving your characteristics!")
    
    # Add some guidance for better results
    st.info("💡 **Tip**: For best results, use prompts like 'as a watercolor painting', 'in Van Gogh style', 'as an oil painting', etc. The app will preserve your facial features and pose.")
    
    # Only log application start once per session
    if "app_started" not in st.session_state:
        logger.info("Image Styler application started")
        st.session_state.app_started = True
    
    # Initialize clients
    replicate_client = get_replicate_client()
    langfuse_client = get_langfuse_client()
    
    col1, col2 = st.columns([1, 1])
    
    # --- Column 1: Input Image and Controls ---
    with col1:
        st.header("Input image")
        
        # Placeholder for the uploaded image itself
        original_image_display_area = st.empty()

        # File uploader - Placed after the image display area conceptually
        uploaded_file = st.file_uploader(
            "Upload an image to style:",
            type=["jpg", "jpeg", "png"],
            help="Upload an image that you want to style while preserving characteristics"
        )
        
        uploaded_image_pil = None # To store the PIL image object

        if uploaded_file is not None:
            uploaded_image_pil = Image.open(uploaded_file)
            # Display the uploaded image in the placeholder
            original_image_display_area.image(uploaded_image_pil, caption="Original Image", use_container_width=True)
            
            file_size_bytes = uploaded_file.size
            file_size_mb = file_size_bytes / (1024 * 1024)
            logger.info(f"Image uploaded: {uploaded_file.name}, Size: {file_size_mb:.2f} MB")
            
            # Style prompt input with better guidance
            style_prompt = st.text_area(
                "How would you like to style this image?",
                placeholder="e.g., 'as a watercolor painting', 'in the style of Van Gogh', 'as an oil painting', 'with impressionist style'",
                height=120,
                help="Describe the artistic style you want. The app will preserve your facial features and pose."
            )
            
            # Advanced options in an expander
            with st.expander("⚙️ Advanced Settings"):
                strength = st.slider(
                    "Style Strength", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.6, 
                    step=0.1,
                    help="Lower values preserve more of the original image"
                )
            # Process button
            if st.button("🎨 Style This Image", type="primary", use_container_width=True):
                if style_prompt.strip():
                    logger.info(f"Starting styling process with prompt: {style_prompt}, UI Strength: {strength}")
                    
                    with st.spinner("Creating styled version while preserving your characteristics..."):
                        # Pass the PIL image object to the styling function
                        styled_image, error = style_uploaded_image_with_strength(
                            uploaded_image_pil, style_prompt, replicate_client, langfuse_client, file_size_mb, strength
                        )
                        
                        if styled_image:
                            st.session_state.styled_image = styled_image
                            st.session_state.style_prompt_for_caption = style_prompt # Store prompt for caption
                            st.success("Image styled successfully!") # Appears in col1
                            logger.info("Image styling completed and saved to session state")
                        else:
                            st.error(f"Failed to style image: {error}") # Appears in col1
                            logger.error(f"Image styling failed: {error}")
                else:
                    st.warning("Please describe the style you want!") # Appears in col1
                    logger.warning("User attempted to style image without providing style prompt")
        else:
            # Message in the image display area if no image is uploaded
            original_image_display_area.info("👆 Upload an image using the control below to get started.")
    
    # --- Column 2: Styled Result ---
    with col2:
        st.header("Styled Result")
        
        if 'styled_image' in st.session_state and st.session_state.styled_image is not None:
            # Use a specific session state key for the caption prompt
            caption_text = st.session_state.get('style_prompt_for_caption', "Styled image")
            st.image(
                st.session_state.styled_image,
                caption=f"Styled: {caption_text}",
                use_container_width=True
            )
            
            # Download button
            img_buffer = io.BytesIO()
            st.session_state.styled_image.save(img_buffer, format="PNG")
            
            st.download_button(
                label="📥 Download Styled Image",
                data=img_buffer.getvalue(),
                file_name="styled_image.png",
                mime="image/png",
                use_container_width=True
            )
            logger.info("Styled image displayed and download option provided")
        else:
            # Placeholder message for the styled image
            st.info("Your styled image will appear here once processed.")


# Add this new function to handle the strength parameter
@observe(name="style_uploaded_image_with_strength")
def style_uploaded_image_with_strength(uploaded_image, style_prompt, client, langfuse, file_size_mb, strength=0.6): # strength here is the UI slider value
    """Style an uploaded image with custom strength parameter."""
    logger.info("Starting image styling process with custom strength")
    
    try:
        # Resize image to prevent OOM errors, e.g., max 768px on the longest side
        original_dimensions = uploaded_image.size
        max_dimension = 768
        if uploaded_image.width > max_dimension or uploaded_image.height > max_dimension:
            uploaded_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {original_dimensions} to {uploaded_image.size} to fit within {max_dimension}x{max_dimension}")
        else:
            logger.info(f"Image dimensions {original_dimensions} are within limits, no resize needed.")

        # Log image upload
        logger.info(f"Processing uploaded image - Size: {file_size_mb:.2f} MB, Dimensions: {uploaded_image.size}, UI Strength: {strength}")
        
        langfuse.event(
            name="image_uploaded",
            input={
                "image_size_mb": round(file_size_mb, 2),
                "image_mode": uploaded_image.mode,
                "original_image_dimensions": original_dimensions,
                "processed_image_dimensions": uploaded_image.size,
                "style_prompt": style_prompt,
                "ui_strength": strength
            }
        )
        
        # Convert image to base64 data URL for Replicate
        image_data_url = encode_image_to_base64(uploaded_image)
        logger.info("Image converted to data URL for Replicate API")
        
        # For instruct-pix2pix, the prompt is an edit instruction.
        # The detailed preservation instruction might be redundant if image_guidance_scale is used.
        # Using the user's style_prompt directly as the edit instruction.
        edit_instruction_prompt = style_prompt 
        logger.info(f"Created edit instruction prompt: {edit_instruction_prompt}")

        # Map UI strength (lower preserves more) to image_guidance_scale (higher preserves more)
        # UI strength [0.1, 1.0]
        # image_guidance_scale for instruct-pix2pix, e.g., [1.0, 2.5]. Higher means more like input image.
        min_ui_strength = 0.1
        max_ui_strength = 1.0
        min_igs = 1.0  # Corresponds to max_ui_strength (low preservation, more style)
        max_igs = 2.0  # Corresponds to min_ui_strength (high preservation, less style)
        
        # Linear mapping: as ui_strength (strength) goes from 0.1 to 1.0, igs goes from 2.0 to 1.0
        image_guidance_val = max_igs - (strength - min_ui_strength) * (max_igs - min_igs) / (max_ui_strength - min_ui_strength)
        image_guidance_val = max(1.0, min(image_guidance_val, 2.5)) # Clamp to a reasonable range

        logger.info(f"UI Strength: {strength} mapped to image_guidance_scale: {image_guidance_val}")

        # Parameters for timothybrooks/instruct-pix2pix
        input_params = {
            "prompt": edit_instruction_prompt,
            "image": image_data_url,
            "image_guidance_scale": image_guidance_val,
            "text_guidance_scale": 7.5, # This was 'guidance_scale' in previous model attempts
            "num_inference_steps": 25,
            # "negative_prompt": "low quality, blurry", # Optional: can add if needed
        }
        
        logger.info(f"Calling Replicate with model {FLUX_MODEL} and params: {input_params}")

        output = client.run(
            FLUX_MODEL, # This is "timothybrooks/instruct-pix2pix:..."
            input=input_params
        )
        
        # The output is typically a URL to the generated image
        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]
        else:
            # For some models, output might be a direct URL string if num_outputs is 1 (default)
            image_url = str(output) 
            
        logger.info(f"{FLUX_MODEL} generation completed. Image URL: {str(image_url)[:100]}...") # Log more of the URL
        
        # Log image generation usage to Langfuse
        langfuse.generation(
            name="image_generation",
            model=FLUX_MODEL,
            input={
                "prompt": edit_instruction_prompt,
                "image_provided": True,
                "ui_strength": strength, 
                "image_guidance_scale": image_guidance_val,
                "text_guidance_scale": input_params["text_guidance_scale"]
            },
            output={"image_url": str(image_url)},
            usage={"total_tokens": 1},  # Placeholder for Langfuse
            metadata={
                "original_image_size_mb": file_size_mb,
                "processed_image_dimensions": uploaded_image.size,
                "model": FLUX_MODEL
            }
        )
        
        # Download the styled image
        logger.info(f"Downloading generated image from {image_url}")
        img_response = requests.get(image_url)
        img_response.raise_for_status() # Raise an exception for HTTP errors
        styled_image = Image.open(io.BytesIO(img_response.content))
        
        # Log successful completion
        logger.info("Image styling process completed successfully")
        langfuse.event(
            name="image_processing_completed",
            input={"success": True, "style_prompt": style_prompt},
            output={"original_image_size_mb": file_size_mb, "final_image_dimensions": styled_image.size}
        )
        
        return styled_image, None
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error downloading styled image: {str(e)}"
        logger.error(f"Image styling failed: {error_msg}")
        langfuse.event(
            name="image_processing_failed",
            input={"style_prompt": style_prompt, "image_size_mb": file_size_mb, "ui_strength": strength},
            output={"error": error_msg, "success": False, "stage": "download"}
        )
        return None, error_msg
    except Exception as e:
        error_msg = f"Error styling image: {str(e)}"
        logger.error(f"Image styling failed: {error_msg}", exc_info=True) # Log full traceback
        
        # Log error to Langfuse
        langfuse.event(
            name="image_processing_failed",
            input={"style_prompt": style_prompt, "image_size_mb": file_size_mb, "ui_strength": strength},
            output={"error": error_msg, "success": False, "stage": "replicate_run_or_processing"}
        )
        
        return None, error_msg

if __name__ == "__main__":
    main()