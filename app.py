import streamlit as st
import os
import logging
from dotenv import load_dotenv
from PIL import Image
import io
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe
import requests
import base64

# Load environment variables
load_dotenv()

# Model constants
VISION_MODEL = "gpt-4o"  # Vision model for image analysis (always latest version)
IMAGE_GENERATION_MODEL = "dall-e-3"  # Image generation model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize clients
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def get_langfuse_client():
    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )

def encode_image_to_base64(image):
    """Convert PIL image to base64 string."""
    logger.info("Converting image to base64 format")
    buffered = io.BytesIO()
    # Ensure RGB format for consistency
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
        logger.info(f"Converted image from {image.mode} to RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    logger.info("Successfully converted image to base64")
    return img_str

@observe(name="style_uploaded_image")
def style_uploaded_image(uploaded_image, style_prompt, client, langfuse, file_size_mb):
    """Style an uploaded image using vision model + image generation model."""
    logger.info("Starting image styling process")
    
    try:
        # Log image upload
        logger.info(f"Processing uploaded image - Size: {file_size_mb:.2f} MB")
        
        langfuse.event(
            name="image_uploaded",
            input={
                "image_size_mb": round(file_size_mb, 2),
                "image_mode": uploaded_image.mode,
                "style_prompt": style_prompt
            }
        )
        
        # Convert image to base64
        base64_image = encode_image_to_base64(uploaded_image)
        logger.info("Image converted to base64 for API processing")
        
        # Log start of vision analysis
        logger.info(f"Starting {VISION_MODEL} vision analysis")
        langfuse.event(
            name="vision_analysis_started",
            input={"model": VISION_MODEL, "image_processed": True}
        )
        
        # Analyze the image using vision model
        analysis_response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image with extreme detail, focusing on preserving the unique characteristics of any people shown. Include specific details about: facial features, hair color and style, eye color, skin tone, clothing details, accessories, pose, expression, and any distinguishing marks. Also describe the background, lighting, and composition. Be very specific about physical attributes."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        image_description = analysis_response.choices[0].message.content
        logger.info(f"{VISION_MODEL} analysis completed. Description length: {len(image_description)} characters")
        

        # Log vision model usage to Langfuse
        langfuse.generation(
            name="vision_analysis",
            model=VISION_MODEL,
            input={
                "messages": "Image analysis request with vision",
                "style_prompt": style_prompt,
                "max_tokens": 500
            },
            output={"description": image_description},
            usage={
                "prompt_tokens": analysis_response.usage.prompt_tokens,
                "completion_tokens": analysis_response.usage.completion_tokens,
                "total_tokens": analysis_response.usage.total_tokens
            },
            metadata={"image_size_mb": file_size_mb}
        )
        
        # Create styled prompt
        styled_prompt = f"Create an image of: {image_description}. Render this EXACT person/subject {style_prompt}, preserving all their specific physical characteristics, facial features, and identity while applying the requested artistic style. Do not make it generic - keep the specific details of this individual."
        logger.info(f"Created styled prompt: {styled_prompt[:100]}...")

        # Log start of image generation
        logger.info(f"Starting {IMAGE_GENERATION_MODEL} image generation")
        langfuse.event(
            name="image_generation_started",
            input={
                "model": IMAGE_GENERATION_MODEL,
                "prompt": styled_prompt,
                "size": "1024x1024"
            }
        )
        
        # Generate styled image using image generation model
        response = client.images.generate(
            model=IMAGE_GENERATION_MODEL,
            prompt=styled_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        logger.info(f"{IMAGE_GENERATION_MODEL} generation completed. Image URL: {image_url[:50]}...")
        
        # Log image generation usage to Langfuse
        langfuse.generation(
            name="image_generation",
            model=IMAGE_GENERATION_MODEL,
            input={
                "prompt": styled_prompt,
                "size": "1024x1024",
                "quality": "standard",
                "n": 1
            },
            output={"image_url": image_url},
            usage={"total_tokens": 1},  # Image generation doesn't use tokens, placeholder for Langfuse
            metadata={
                "original_image_size_mb": file_size_mb,
                "generated_image_size": "1024x1024"
            }
        )
        
        # Download the styled image
        logger.info("Downloading generated image")
        img_response = requests.get(image_url)
        styled_image = Image.open(io.BytesIO(img_response.content))
        
        # Log successful completion
        logger.info("Image styling process completed successfully")
        langfuse.event(
            name="image_processing_completed",
            input={"success": True, "style_prompt": style_prompt},
            output={"original_image_size_mb": file_size_mb}
        )
        
        return styled_image, None
        
    except Exception as e:
        error_msg = f"Error styling image: {str(e)}"
        logger.error(f"Image styling failed: {error_msg}")
        
        # Log error to Langfuse
        langfuse.event(
            name="image_processing_failed",
            input={"style_prompt": style_prompt, "image_size_mb": file_size_mb},
            output={"error": error_msg, "success": False}
        )
        
        return None, error_msg

def main():
    st.set_page_config(
        page_title="Image Styler",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Yorgos' Styling Salon")
    st.markdown("Upload an image and describe how you want it styled!")
    
    # Only log application start once per session
    if "app_started" not in st.session_state:
        logger.info("Image Styler application started")
        st.session_state.app_started = True
    
    # Initialize clients
    openai_client = get_openai_client()
    langfuse_client = get_langfuse_client()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input image")
        
        # File uploader (required)
        uploaded_file = st.file_uploader(
            "Upload an image to style:",
            type=["jpg", "jpeg", "png"],
            help="Upload an image that you want to recreate in a different style"
        )
        
        uploaded_image = None
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Original Image", use_container_width=True)
            
            # Get file size in MB
            file_size_bytes = uploaded_file.size
            file_size_mb = file_size_bytes / (1024 * 1024)

            logger.info(f"Image uploaded: {uploaded_file.name}, Size: {file_size_mb:.2f} MB")
            
            # Style prompt input
            style_prompt = st.text_area(
                "How would you like to style this image?",
                placeholder="e.g., 'in the style of Van Gogh', 'as a watercolor painting', 'with cyberpunk aesthetics', 'as a pencil sketch'",
                height=120
            )
            
            # Process button
            if st.button("🎨 Style This Image", type="primary", use_container_width=True):
                if style_prompt.strip():
                    logger.info(f"Starting styling process with prompt: {style_prompt}")
                    
                    with st.spinner("Analyzing image and creating styled version..."):
                        styled_image, error = style_uploaded_image(
                            uploaded_image, style_prompt, openai_client, langfuse_client, file_size_mb
                        )
                        
                        if styled_image:
                            st.session_state.styled_image = styled_image
                            st.session_state.style_prompt = style_prompt
                            st.success("Image styled successfully!")
                            logger.info("Image styling completed and saved to session state")
                        else:
                            st.error(f"Failed to style image: {error}")
                            logger.error(f"Image styling failed: {error}")
                else:
                    st.warning("Please describe the style you want!")
                    logger.warning("User attempted to style image without providing style prompt")
        else:
            st.info("👆 Please upload an image to get started")
    
    with col2:
        st.header("Styled Result")
        
        if hasattr(st.session_state, 'styled_image'):
            st.image(
                st.session_state.styled_image,
                caption=f"Styled: {st.session_state.style_prompt}",
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
            st.info("Upload an image and describe the style to see the result here.")

if __name__ == "__main__":
    main()