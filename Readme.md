# 🤖 AI Web Agent System

> **Transform your ideas into fully functional websites using local AI models.**  
> Powered by **Ollama** for intelligent content generation and **Stable Diffusion XL** for stunning visuals.

![Status](https://img.shields.io/badge/status-production%20ready-success)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![VRAM](https://img.shields.io/badge/VRAM-11GB%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| 🧠 **Multi-Agent Architecture** | Planning, Text, Design, and Imaging agents working together |
| 🎨 **AI-Generated Images** | 7 custom images per website (Hero, Gallery, Feature) |
| 📄 **Single HTML Output** | Fully responsive website with embedded CSS/JS |
| 📦 **ZIP Download** | One-click download of complete website package |
| 🔒 **100% Local** | No cloud APIs, all processing on your machine |
| ⚡ **VRAM Optimized** | Runs on 11GB GPUs with intelligent memory management |

---

## 🛠️ Requirements

### Hardware
| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU** | NVIDIA 11GB VRAM | NVIDIA 12GB+ VRAM |
| **RAM** | 16GB | 32GB |
| **Storage** | 20GB free | 50GB+ SSD |

### Software
- Python 3.8+
- CUDA 11.8+
- Ollama (latest version)

---

## 📦 Installation

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd ai-web-agent
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
# or for fish shll users
source venv/bin/activate.fish
pip install -r requirements.txt
```

### 3. Download Ollama Model

```bash
ollama pull ministral-3:8b
```

💡 Recommended Model: ministral-3:8b offers excellent performance with lower VRAM usage compared to larger models.

### 4. Place SDXL Model
Download your preferred SDXL checkpoint and place it in the /models folder:
```
models/
└── your-sdxl-model.safetensors
```


## ⚙️ Configuration
Environment Variables
Create a .env file or configure via the web UI:

```
OLLAMA_MODEL=ministral-3:8b
SDXL_MODEL_PATH=models/your-sdxl-model.safetensors
SDXL_CFG=7.5
SDXL_STEPS=20
SDXL_SAMPLER=Euler a
```

### Critical Environment Variable
Must be set before running to prevent CUDA memory fragmentation:
```bash
# Windows PowerShell
$env:PYTORCH_ALLOC_CONF="expandable_segments:True"
python app.py

# Linux/Mac
export PYTORCH_ALLOC_CONF=expandable_segments:True
python app.py
```

## 🚀 Usage
### 1. Start the Server
```bash
python app.py
```

### 2. Open Web Interface
Navigate to: http://localhost:5000

3. Generate a Website
```
    Enter your website idea (e.g., "A futuristic vertical farming startup")
    Click "Start Agent System"
    Watch the agents work in real-time
    Download the completed website as ZIP
```


## Agent Workflow
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Planning   │ ──► │    Text     │ ──► │   Imaging   │ ──► │   Design    │
│   Agent     │     │   Agent     │     │   (SDXL)    │     │   Agent     │
│  (Ollama)   │     │  (Ollama)   │     │  (7 images) │     │  (Ollama)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 📁 Project Structure
```
ai-web-agent/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                   # Configuration file
├── models/                # SDXL model files
│   └── *.safetensors
├── output/                # Generated websites
│   ├── index.html
│   └── images/
│       ├── gen_0.png      # Hero image
│       ├── gen_1-3.png    # Gallery images
│       └── gen_4-6.png    # Feature images
└── templates/
    └── index.html         # Web UI
```

## 🎯 Tips for Best Results

```
    Be Specific with Ideas:
    ✅ "A modern coffee shop with warm lighting and minimalist design"
    ❌ "A website about coffee"
    Use Recommended Model:
    ministral-3:8b provides the best balance of quality and VRAM efficiency.
    Monitor VRAM:
    Watch the terminal logs for VRAM usage during generation.
    Save Configurations:
    Use the "Save All Settings" button to persist your preferred settings.
```

## 📄 License
```
MIT License - Feel free to use, modify, and distribute.
```

## 🙏 Acknowledgments

```
    Ollama - Local LLM inference
    Stable Diffusion XL - Image generation
    Diffusers - Hugging Face pipeline library
    PyTorch - Deep learning framework
```

Built with ❤️ for local AI enthusiasts
