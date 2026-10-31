# MindfulAI — Multimodal AI Chatbot

A high-performance, resilient multimodal chatbot built with **Python**, **Streamlit**, **Google Gemini API**, and **Ollama Local AI**.

MindfulAI features a **dual-provider failover architecture**:
- **Primary Provider**: Google Gemini API (Ultra-fast multimodal reasoning, image understanding, audio input processing, and Google Imagen 3 text-to-image).
- **Secondary Provider**: Local Ollama AI (Completely private, 100% free, automatic local fallback when Gemini daily quota is reached).

---

## 1. Resolved Issues

### Issue 1: Gemini Audio Input Modality Error (400)
- **Problem**: Incoming audio (from microphone or file upload) was sent to a Text-to-Speech preview model (`gemini-2.5-flash-preview-tts`), producing `(InvalidArgument): 400 Audio input modality is not enabled for models/gemini-2.5-flash-preview-tts`.
- **Solution**:
  - Model capabilities are strictly separated in `config.py` and `services/gemini_service.py`.
  - Incoming audio is **exclusively routed to `AUDIO_INPUT_MODEL`** (`gemini-1.5-flash` or `gemini-2.0-flash`).
  - Strict guards in `gemini_discovery.py` classify any model with `-tts` as a speech output model and prevent it from receiving audio input.

### Issue 2: Gemini Daily Free-Tier Quota (429) & Silent Lockout
- **Problem**: When Google Gemini's daily free-tier quota (20 requests/day) was exhausted, users were locked out with no recourse.
- **Solution**:
  - Implemented `services/quota_manager.py` to detect HTTP 429 daily quota exhaustion vs. per-minute RPM rate limits.
  - Automatically activates the local Ollama AI assistant (`gemma2:2b`) without dropping conversation history.
  - Displays a clean status notification and live countdown to quota recovery.
  - Automatically resumes Gemini availability when the quota reset window elapses.

---

## 2. Project Architecture

```
chat bot/
├── app.py                      # Main Streamlit application
├── streamlit_app.py            # Streamlit entrypoint alias
├── config.py                   # Centralized configuration & model identifiers
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables & secrets (ignored in git)
├── .env.example                # Template configuration
├── README.md                   # Documentation & setup guide
│
├── services/
│   ├── gemini_service.py       # Google Gemini API client (text, vision, audio input, Imagen)
│   ├── ollama_service.py       # Local Ollama AI client (text, vision fallback)
│   ├── ai_router.py            # Centralized provider router & automatic failover engine
│   ├── quota_manager.py        # Dedicated 429 quota tracking, cooldown & recovery
│   ├── text_service.py         # Prompt engineering & conversation history
│   ├── audio_input_service.py  # Audio validation & speech-to-text processing
│   ├── tts_service.py          # Speech synthesis & text-to-speech output
│   └── image_service.py        # Multimodal vision & Google Imagen 3 generation
│
├── utils/
│   ├── audio_utils.py          # Audio validation & WAV inspections
│   ├── file_validator.py       # File upload constraints (docs, images, audio)
│   ├── session_manager.py      # Streamlit session state management
│   └── error_handler.py        # Professional banners & status indicators
│
└── tests/
    ├── test_master_v2.py       # Comprehensive Master Prompt V2.0 test suite
    ├── test_ollama_provider.py # Ollama integration tests
    ├── test_gemini_discovery.py# Dynamic model discovery & capability tests
    └── test_rate_limiting.py   # Rate limiting & quota tests
```

---

## 3. Quick Start Guide (Windows PowerShell)

### Step 1: Virtual Environment & Dependencies
```powershell
cd "c:\Users\Lenovo\OneDrive\Documents\chat bot"

# Activate Python virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (`.env`)
Create or edit `.env` in the root folder:
```ini
# Google Gemini API (Primary Provider)
GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
GEMINI_TEXT_MODEL="gemini-1.5-flash"
GEMINI_IMAGE_MODEL="gemini-1.5-flash"
GEMINI_AUDIO_INPUT_MODEL="gemini-1.5-flash"
GEMINI_TTS_MODEL="gemini-2.5-flash-preview-tts"
GEMINI_RPM_LIMIT=4
GEMINI_MAX_RETRIES=3

# Ollama Local AI (Automatic Offline Fallback)
OLLAMA_ENABLED=true
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_TEXT_MODEL="gemma2:2b"
OLLAMA_VISION_MODEL="moondream"
OLLAMA_TIMEOUT=120.0
OLLAMA_FALLBACK_ON_DAILY_QUOTA=true
```

### Step 3: Set Up Ollama Local AI (Optional but Recommended)
1. Download Ollama for Windows from [ollama.com/download](https://ollama.com/download).
2. Start the Ollama server:
   ```powershell
   ollama serve
   ```
3. In a second PowerShell terminal, pull the recommended lightweight model:
   ```powershell
   ollama pull gemma2:2b
   ```
   *(Optional for local vision capability)*:
   ```powershell
   ollama pull moondream
   ```

### Step 4: Launch MindfulAI
```powershell
streamlit run app.py
```
*Open in your browser:* `http://localhost:8501`

*(Alternative launch command)*:
```powershell
streamlit run streamlit_app.py
```

---

## 4. Running the Automated Test Suite

All 67 unit and integration tests execute with zero external quota consumption:
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_master_v2.py tests/test_ollama_provider.py tests/test_gemini_discovery.py tests/test_rate_limiting.py tests/test_database.py -v
```

---

## 5. Multimodal Features Supported

| Feature | Primary Engine (Gemini) | Local Fallback (Ollama) |
| :--- | :--- | :--- |
| **Text Chat** | `gemini-1.5-flash` / `gemini-2.0-flash` | `gemma2:2b` |
| **Document/Code Editing** | Full context reasoning | Local Ollama reasoning |
| **Image Understanding** | Multimodal Vision | `moondream` / `llava:7b` (when pulled) |
| **Audio Input (Speech)** | `gemini-1.5-flash` (strictly non-TTS) | Local SpeechRecognition + Ollama text |
| **Voice Output (TTS)** | Browser Web Speech API | Browser Web Speech API |
| **Image Generation** | Google Imagen 3 | Not supported (honest advisory displayed) |

