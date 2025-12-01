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

### 1. Install System Dependencies (FFmpeg)
This project requires **FFmpeg** for audio processing (`librosa` and `pypdf` dependency).

* **Ubuntu/Debian:** `sudo apt-get install ffmpeg`
* **Mac (Homebrew):** `brew install ffmpeg`
* **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/) and add to your System PATH.

### 2. Install Python Packages
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```
### 3. Download Spacy Model
Required for the text preprocessing logic.

```bash
python -m spacy download en_core_web_sm
```

## ⚙️ Model Configuration

### Confidence Model (.pth)
The application (`utils.py`) looks for a specific PyTorch model checkpoint to calculate confidence scores.

1.  If you have trained the model using `Confidence_Notebook.ipynb`, locate the file named `confidence_model.pth`.
2.  Place this file in the root directory of the project.
3.  **Important:** Open `utils.py` and ensure the `load_emotion_model` function points to the correct path. If the file is in the root directory, update the path in `utils.py` to:

```python
# In utils.py
local_model_path = "confidence_model.pth"
```

## 🚀 Usage

### Run the Streamlit App:
```bash
streamlit run finalapp.py
```

## 🔑 API Key Configuration

### Enter API Key (Optional but Recommended):
* On the sidebar, enter your **Groq API Key**.
* This enables **dynamic follow-up questions**. If skipped, the application will default to a static list of interview questions.

## 🚀 Application Workflow
* **Phase 1: CV Screening** 📄
    * **Action:** Paste a Job Description and upload a Candidate CV (PDF). Click "**Analyze**".
* **Phase 2: Interview Start** 🎙️
    * **Action:** If the match score is **greater than 50%**, the live interview phase begins.
* **Phase 3: Live Interview Assessment** 📊
    * **Action:** Listen to the AI question, record your audio answer, and then receive real-time analysis.
    * **Analysis:** The results include scores for **Technical Relevance** and **Confidence**.

## 🧪 Notebooks for Research
The following files are included for development, training, and testing the core features:

* **`Confidence_Notebook.ipynb`**: Run this in Google Colab or Kaggle to train your own **Audio Confidence Classifier** using the RAVDESS or LibriSpeech datasets.
* **`Skills matching.py`**: Run this locally to test specific **keyword extraction logic** against resume PDFs.

## 🛡️ Troubleshooting Guide
| Issue | Recommended Action | Source |
| :--- | :--- | :--- |
| **Audio Error** (e.g., `PySoundFile failed`) | Ensure **FFmpeg** is installed correctly on your system. | N/A (General audio dependency) |
| **Model Not Found** (`confidence_model.pth`) | Ensure `confidence_model.pth` is in the same directory as `finalapp.py` or update the path in `utils.py`. | `utils.py`, `finalapp.py` |
| **Groq Connection** (Dynamic questions fail) | Check your **internet connection** and ensure your **API key validity**. | `finalapp.py` |
