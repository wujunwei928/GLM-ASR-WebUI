# GLM-ASR-WebUI

**[中文阅读.](./README_zh.md)**

A modern web-based speech recognition service powered by the **[GLM-ASR](https://github.com/zai-org/GLM-ASR)** model, featuring a cyberpunk-themed UI and streaming API support for long audio files.

![Version](https://img.shields.io/badge/version-0.0.1-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Python](https://img.shields.io/badge/python-3.12+-brightgreen)

## ✨ Features

- 🎯 **High Accuracy**: Powered by GLM-ASR-Nano-2512 for state-of-the-art speech recognition
- 🚀 **Streaming API**: Real-time transcription progress for long audio files
- 🎨 **Cyberpunk UI**: Beautiful neon-style interface with particle effects
- 🎙️ **Multiple Input Methods**: File upload, URL download, and real-time recording
- 📦 **Auto Chunking**: Automatically splits long audio into manageable segments
- ⚡ **GPU Acceleration**: CUDA support for faster inference
- 🔌 **REST API**: Clean API design for easy integration

## 📸 Preview

The web interface features a dynamic particle background and neon-styled controls:

- **File Upload Tab**: Drag & drop or click to upload audio files
- **URL Input Tab**: Transcribe audio from direct URLs
- **Recording Tab**: Record directly from your microphone

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- CUDA-capable GPU (optional, for faster inference)
- FFmpeg (system-level dependency, **required only for**:
  - Detecting audio duration
  - Splitting audio files longer than 30 seconds into chunks)

#### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg Official Site](https://ffmpeg.org/download.html) and add to PATH.

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/GLM-ASR-WebUI.git
cd GLM-ASR-WebUI
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Start the Server

**Development mode (with auto-reload):**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

The web interface will be available at `http://localhost:8000`

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Choose an input method:
   - **File Upload**: Click or drag an audio file (WAV, MP3, FLAC, OGG, M4A)
   - **URL**: Enter a direct URL to an audio file
   - **Recording**: Click the microphone button to record
3. Click "Start Recognition" (or equivalent button)
4. Watch real-time progress as audio is transcribed
5. View the complete transcription result

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### Standard Transcription
```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav" \
  -F "language=zh"
```

**Response:**
```json
{
  "success": true,
  "text": "Transcribed text here...",
  "duration": 12.5,
  "error": null
}
```

#### Streaming Transcription (Recommended for Long Audio)
```bash
curl -X POST http://localhost:8000/api/v1/transcribe-stream \
  -F "file=@long_audio.mp3" \
  -F "chunk_duration=30"
```

**Streaming Response (NDJSON):**
```json
{"type": "info", "message": "Audio duration: 125.40s", "duration": 125.4}
{"type": "chunk", "chunk_index": 0, "total_chunks": 5, "text": "First segment...", "progress": 20.0}
{"type": "chunk", "chunk_index": 1, "total_chunks": 5, "text": "Second segment...", "progress": 40.0}
{"type": "complete", "text": "Complete transcription here...", "total_chunks": 5}
```

#### Python Example

```python
import requests

# For standard transcription
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/transcribe',
        files={'file': f},
        data={'language': 'zh'}
    )
    result = response.json()
    print(result['text'])

# For streaming transcription
with open('long_audio.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/transcribe-stream',
        files={'file': f},
        data={'chunk_duration': 30},
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            print(f"Type: {data['type']}, Data: {data}")
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check and model status |
| `/api/v1/transcribe` | POST | Standard transcription (single response) |
| `/api/v1/transcribe-stream` | POST | Streaming transcription (recommended) |
| `/api/v1/model/info` | GET | Model information |
| `/api/info` | GET | Service information |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

## ⚙️ Configuration

### Environment Variables

You can modify these constants in `app.py`:

```python
MODEL_ID = "zai-org/GLM-ASR-Nano-2512"  # Model to use
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Inference device
```

### Chunking Parameters

When using the streaming API, you can specify:

- `chunk_duration`: Duration per chunk in seconds (default: 30)

## 🏗️ Architecture

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Transformers**: Hugging Face library for model loading
- **PyTorch**: Deep learning framework
- **ffmpeg-python**: Audio duration detection and chunking for files >30s

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Canvas API**: Particle system for dynamic background
- **Fetch API**: For streaming responses
- **MediaRecorder API**: For browser-based recording

## 🎯 Technical Details

### Audio Processing

1. **Upload**: Audio files are saved to `/tmp/glm_asr_uploads/`
2. **Duration Detection**: FFmpeg extracts audio duration (required for all files)
3. **Chunking**: Audio longer than 30 seconds is split using FFmpeg (configurable)
4. **Transcription**: Each chunk is processed independently
5. **Aggregation**: Results are combined and returned

> **Note**: FFmpeg is only used for metadata extraction and audio splitting.
> The actual speech recognition is performed by the GLM-ASR model.

### Streaming Protocol

The streaming API uses **NDJSON** (Newline-Delimited JSON) format:

```javascript
// Event types
"info"     // Initial information (duration, etc.)
"chunk"    // Individual chunk result (with progress)
"error"    // Error during chunk processing
"complete" // Final transcription
```

### Model Inference

```python
# Fixed inference parameters
max_new_tokens=256    # Maximum output length
do_sample=False       # Use greedy decoding
dtype=torch.bfloat16  # Precision for efficiency
```

## 🐛 Troubleshooting

### Model Download Issues

The first run will automatically download the model (~500MB). If it fails:

1. Check your internet connection
2. Set the `HF_ENDPOINT` environment variable for a faster mirror:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. The model will be cached in `~/.cache/huggingface/`

### CUDA Not Available

If you see `DEVICE = "cpu"` but have a GPU:

1. Verify CUDA installation:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### FFmpeg Errors

FFmpeg is required for detecting audio duration and splitting long files (>30s).

Ensure FFmpeg is installed and accessible:
```bash
ffmpeg -version
```

If you encounter FFmpeg-related errors:
- Verify that FFmpeg can be executed from the command line
- Check that the audio file format is supported by FFmpeg
- For short audio files (<30s), FFmpeg is only used for duration detection
- For long audio files (>30s), FFmpeg is also used to split the audio into chunks

### Temporary Files Not Cleaned Up

Check permissions for `/tmp/glm_asr_*` directories:
```bash
ls -la /tmp/glm_asr_uploads
ls -la /tmp/glm_asr_chunks
```

## 🔧 Development

### Running Tests

Currently, there are no automated tests. To manually test:

1. Start the server
2. Test health check: `curl http://localhost:8000/health`
3. Upload a short audio file via the web interface
4. Upload a long audio file (> 2 minutes) to test chunking

### Code Style

This project follows PEP 8 guidelines. Contributions should:

- Use Chinese comments for explanations
- Follow existing code structure
- Include error handling with detailed logging
- Clean up temporary files in `finally` blocks

## 📊 Performance

### Benchmarks

Approximate inference times on different hardware:

| Hardware | Duration | Time | Real-time Factor |
|----------|----------|------|------------------|
| RTX 3090 | 1 min | ~6s | 0.1x |
| RTX 3060 | 1 min | ~12s | 0.2x |
| CPU (i7) | 1 min | ~180s | 3x |

*Note: First inference includes model loading time (~10s)*

### Optimization Tips

- Use GPU whenever possible
- Enable CUDA graphs for faster inference
- Adjust `chunk_duration` based on your needs
- Use streaming API for better user experience

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Model**: [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) by ZAI
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **UI Design**: Inspired by cyberpunk aesthetics

## 📮 Support

For issues, questions, or suggestions:

- Open an issue on GitHub
- Check existing documentation in `/docs`
- Review the code comments in `app.py`

## 🔮 Roadmap

- [ ] Add batch transcription for multiple files
- [ ] Support for speaker diarization
- [ ] Export transcriptions to SRT/VTT formats
- [ ] Add authentication and rate limiting
- [ ] Docker container for easy deployment
- [ ] WebSocket support for real-time transcription

---

Made with ❤️ and neon lights
