# 3D Model Generator

A powerful application that converts 2D images to high-quality 3D models using AI. The application leverages Hunyuan3D-2 models from Tencent and provides a simple API for generating, storing, and retrieving 3D models.

## Features

- Convert 2D images to 3D models with just one API call
- Support for text prompts to guide the 3D generation
- High-quality texture generation
- Vector database integration for similarity search
- FastAPI backend with a Streamlit frontend
- Support for multiple output formats (GLB, OBJ)

## Architecture

- **FastAPI Backend**: Handles image upload, 3D generation, and model serving
- **Hunyuan3D Models**: Powers the 2D to 3D conversion with state-of-the-art quality
- **Qdrant Vector Database**: Stores model vectors for semantic search
- **Streamlit Frontend**: Provides a user-friendly interface

## Setup Instructions

### Prerequisites

- Python 3.10+
- Pip package manager
- Git
- HuggingFace account with API token
- Qdrant account (optional, for vector database)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rishab-max17/3DModelGenerator2.git
   cd 3DModelGenerator2
   ```

2. Create a virtual environment:
   ```bash
   python -m venv fresh_venv
   source fresh_venv/bin/activate  # On Windows: fresh_venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.sample` to `.env`
   - Add your HuggingFace API token
   - (Optional) Add Qdrant configuration

### Running the Application

1. Start the FastAPI backend:
   ```bash
   python app.py
   ```

2. Start the Streamlit frontend:
   ```bash
   streamlit run frontend.py
   ```

3. Open your browser at http://localhost:8501

## API Endpoints

- **POST /generate**: Generate a 3D model from an image
- **POST /generate-with-prompt**: Generate a 3D model with a text prompt
- **POST /generate-simple**: Simplified endpoint with form parameters
- **GET /models/{request_id}/download**: Download a generated 3D model
- **GET /models/search**: Search for similar 3D models

## Vector Database Integration

The application uses Qdrant for storing and retrieving 3D model embeddings:

1. Each generated 3D model is stored with:
   - Visual embeddings from the input image
   - Metadata (resolution, format, generation parameters)
   - Unique UUID identifier

2. Models can be searched by:
   - Text description (using text-to-vector embedding)
   - Visual similarity (using image embeddings)

3. Configuration:
   - Set `QDRANT_URL` and `QDRANT_API_KEY` in your `.env` file
   - Run the test script: `python test_qdrant.py`

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request. 