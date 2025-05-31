# Image Styler

A Streamlit web application that allows you to upload images and apply various artistic styles to transform them.

## Features

- 📸 **Image Upload**: Support for common image formats (JPG, PNG, JPEG)
- 🎨 **Style Transfer**: Apply artistic styles to your images
- 🖼️ **Real-time Preview**: See styled results instantly
- 💾 **Download Results**: Save your styled images locally
- 🌐 **Web Interface**: Easy-to-use browser-based application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/giorrgos/image-styler.git
cd image-styler
```

2. Create a virtual environment:
```bash
uv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
uv sync
```

## Usage

1. Start the Streamlit application:
```bash
uv run streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload an image using the file uploader

4. Select your desired style from the available options

5. Wait for processing and download your styled image

## Requirements

- Python 3.12+
- Streamlit
- PIL/Pillow

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Image processing powered by PIL/Pillow
