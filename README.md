# AI Smart Recruiter & Interview Assessment System

An AI-powered hiring platform that automates **CV screening**, conducts **voice-based technical interviews**, and analyzes candidate responses for both **technical relevance** and **speaking confidence**.

Built with **Streamlit**, **PyTorch**, **Transformers**, and **Groq**.

## 🚀 Features

* **📄 CV Screening:** * Upload PDF CVs and Job Descriptions.
    * Calculates a semantic match score using `SentenceTransformer`.
    * Auto-rejects or accepts candidates based on a match threshold.
* **🎙️ AI Audio Interviewer:**
    * Generates dynamic or static interview questions using **Edge TTS** (Text-to-Speech).
    * Records candidate answers directly in the browser.
    * Uses **Groq (Llama 3)** to generate follow-up questions based on context.
* **🧠 Advanced Analysis:**
    * **Technical Score:** Compares the semantic meaning of the candidate's answer against the question context.
    * **Confidence Score:** Uses a custom-trained **Wav2Vec2** model to detect confidence levels in the speaker's voice.
    * **Emotion/Tone Detection:** Analyzes audio features to determine speaker sentiment.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `finalapp.py` | The main Streamlit dashboard application. Handles the UI and interview flow. |
| `utils.py` | Contains core logic: Audio processing, TTS generation, PDF text extraction, and model loading. |
| `Confidence_Notebook.ipynb` | Jupyter Notebook used to train/fine-tune the Wav2Vec2 model for confidence detection. |
| `Transcirbtion and assessment.ipynb` | Research notebook for testing Whisper ASR and semantic similarity scoring. |
| `Skills matching.py` | Standalone script for extracting skills from PDFs using Spacy phrase matching. |
| `requirements.txt` | List of Python dependencies. |

## 🛠️ Installation

### 2. Install System Dependencies (FFmpeg)
This project requires **FFmpeg** for audio processing (`librosa` and `pypdf` dependency).

* **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
* **Mac (Homebrew):** `brew install ffmpeg`
* **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/) and add to your System PATH.

### 3. Install Python Packages
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt

### 4. Download Spacy Model
Required for the text preprocessing logic.

```bash
python -m spacy download en_core_web_sm
