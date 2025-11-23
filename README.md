# 🎭 Emotion-Aware Storyteller

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-Emotion%20AI-green.svg)
![Llama](https://img.shields.io/badge/Llama%203-LLM-purple.svg)

**Real-time facial emotion detection powered by AI storytelling**

A web application that detects emotions from webcam input and generates personalized, context-aware stories, captions, and affirmations using state-of-the-art AI models. Features optional text-to-speech for an immersive, multimodal experience.

## 🌟 Features

- **🎯 Real-time Emotion Detection**: Uses DeepFace with OpenCV for accurate facial emotion recognition
- **🤖 AI-Powered Story Generation**: Leverages Llama 3 (via Ollama) to create personalized content
- **🔊 Text-to-Speech**: Optional audio output using gTTS or Edge TTS
- **🎨 Multi-Emotion Blending**: Weighted emotion analysis for nuanced content generation
- **📊 Visual Feedback**: Interactive emotion breakdown and confidence scores
- **🎭 Multiple Content Types**: Stories, captions, advice, poems, and affirmations
- **🖥️ Modern Web UI**: Built with Gradio 4.0 for seamless user experience

## 🎬 Demo

```bash
# Quick start
python src/app.py
```

Then open your browser to `http://localhost:7860`

## 🏗️ Architecture

```
emotion_storyteller/
├── src/
│   ├── detector.py      # DeepFace emotion detection
│   ├── generator.py     # LLM story generation
│   ├── tts.py          # Text-to-speech synthesis
│   └── app.py          # Gradio web interface
├── models/             # Model cache (auto-downloaded)
├── assets/             # Static assets
├── requirements.txt
└── README.md
```

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Gradio 4.0+ | Interactive web UI with webcam support |
| **Emotion Detection** | DeepFace + OpenCV | Pre-trained emotion recognition |
| **Story Generation** | Llama 3 (Ollama) | Context-aware content creation |
| **Text-to-Speech** | gTTS / Edge TTS | Audio feedback synthesis |
| **Runtime** | Python 3.10+ | Core application framework |

## 📋 Prerequisites

- Python 3.10 or higher
- Webcam (for real-time emotion detection)
- 4GB+ RAM (8GB recommended for LLM backends)
- (Optional) Ollama installed for local LLM inference

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Emotion-Aware-Storyteller.git
cd Emotion-Aware-Storyteller
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Ollama

For local LLM inference with Llama 3:

```bash
# On Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# On Windows
# Download from https://ollama.com/download

# Pull Llama 3 model
ollama pull llama3
```

## 🎮 Usage

### Basic Usage

```bash
python src/app.py
```

This launches the web interface at `http://localhost:7860`

### Advanced Options

```bash
# Use different LLM backend
python src/app.py --llm-backend huggingface

# Use different TTS backend
python src/app.py --tts-backend edge-tts

# Specify custom model
python src/app.py --model-name meta-llama/Llama-3.2-3B-Instruct

# Create public share link
python src/app.py --share

# Custom port
python src/app.py --port 8080
```

### Available Backends

**LLM Backends:**
- `ollama` (default) - Local inference, fast, no API key needed
- `huggingface` - Transformers library, flexible model choice
- `openai` - OpenAI API (requires API key)
- `template` - Fallback template-based generation

**TTS Backends:**
- `gtts` (default) - Google Text-to-Speech, simple and reliable
- `edge-tts` - Microsoft Edge TTS, high quality voices
- `pyttsx3` - Offline TTS, no internet required

## 💡 Usage Guide

1. **Capture Emotion**: Click to capture a photo from your webcam
2. **Select Content Type**: Choose from Story, Caption, Advice, Poem, or Affirmation
3. **Configure Options**:
   - Enable/disable text-to-speech
   - Toggle multi-emotion blending for nuanced content
4. **Analyze & Generate**: Click the button to process

## 🎯 Use Cases

- **Mental Wellness Apps**: Provide personalized emotional support
- **Education**: Create emotion-responsive learning experiences
- **Gaming**: Adaptive storytelling based on player emotions
- **Therapy Tools**: Interactive emotion recognition exercises
- **Creative Writing**: Emotion-driven story prompts
- **Accessibility**: Audio feedback for visual content

## 🔬 How It Works

### 1. Emotion Detection
```python
from src.detector import EmotionDetector

detector = EmotionDetector()
result = detector.detect_emotion(image)
# Returns: {'dominant_emotion': 'happy', 'confidence': 0.87, ...}
```

### 2. Story Generation
```python
from src.generator import StoryGenerator

generator = StoryGenerator(backend="ollama")
story = generator.generate(emotion="happy", prompt_type="story")
```

### 3. Text-to-Speech
```python
from src.tts import TextToSpeech

tts = TextToSpeech(backend="gtts")
audio_path, _ = tts.text_to_speech(story)
```

## 📊 Supported Emotions

- 😊 **Happy** - Joyful, cheerful, content
- 😢 **Sad** - Melancholic, sorrowful, down
- 😠 **Angry** - Frustrated, irritated, mad
- 😨 **Fear** - Anxious, worried, scared
- 😲 **Surprise** - Amazed, astonished, shocked
- 😐 **Neutral** - Calm, balanced, composed
- 🤢 **Disgust** - Repulsed, averse, displeased

## 🎨 Customization

### Adding Custom Prompts

Edit `src/generator.py`:

```python
prompts = {
    "custom_type": f"Your custom prompt for {emotion}..."
}
```

### Custom Emotion Voices

Edit `src/tts.py`:

```python
emotion_voices = {
    "happy": "en-US-YourPreferredVoice",
    ...
}
```

### UI Theming

Modify `src/app.py`:

```python
gr.Blocks(theme=gr.themes.YourTheme())
```

## 🧪 Testing

Test individual components:

```bash
# Test emotion detection
python src/detector.py

# Test story generation
python src/generator.py

# Test text-to-speech
python src/tts.py
```

## 📈 Performance

- **Emotion Detection**: ~100-200ms per frame
- **Story Generation**:
  - Ollama: ~1-3 seconds
  - HuggingFace: ~2-5 seconds
  - Template: <100ms
- **Text-to-Speech**: ~500ms-1s for short text

## 🐛 Troubleshooting

### No Face Detected
- Ensure adequate lighting
- Center your face in the frame
- Check webcam permissions

### Slow Story Generation
- Use `--llm-backend template` for instant responses
- Try smaller LLM models
- Ensure GPU acceleration is enabled (for HuggingFace)

### Ollama Connection Issues
```bash
# Check Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

### TTS Not Working
```bash
# Try alternative backend
python src/app.py --tts-backend edge-tts
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Multi-face detection and tracking
- [ ] Emotion history visualization
- [ ] Avatar animation matching emotions
- [ ] Background music generation
- [ ] Export conversation logs
- [ ] Mobile app version
- [ ] Multiple language support
- [ ] Fine-tuned emotion-specific LLM models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for emotion detection
- [Ollama](https://ollama.com) for local LLM inference
- [Gradio](https://gradio.app) for the web interface
- [Meta AI](https://ai.meta.com) for Llama 3

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

## ✨ Portfolio Highlight

> "Built a Gradio-based Emotion-Aware Storyteller that uses DeepFace for real-time facial emotion detection and Llama 3 for adaptive caption generation with optional speech output — an example of human-AI emotional co-creation."

**Impact**: Demonstrates multimodal AI reasoning (Computer Vision + NLP + Speech), creative product thinking, and practical application of cutting-edge ML technologies.

---

**Made with ❤️ and AI**
