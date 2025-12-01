import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pypdf
import io
import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
import re
import edge_tts
import asyncio

# --- 1. MODEL LOADING (Cached) ---

@st.cache_resource
def load_cv_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_asr_pipeline():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

@st.cache_resource
def load_emotion_model():
    """
    Loads Wav2Vec2 model. Supports local .pth and public HF models.
    """
    local_model_path = r"C:\Users\ONE TOUCH\Downloads\confidence_model.pth"
    base_model_template = "facebook/wav2vec2-base"
    large_model_template = "facebook/wav2vec2-large-xlsr-53"
    fallback_model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    feature_extractor = None
    model = None
    
    # Attempt local load
    if os.path.isfile(local_model_path):
        try:
            print(f"Attempting local load: {local_model_path}")
            state_dict = torch.load(local_model_path, map_location=torch.device('cpu'))
            
            # Detect size
            hidden_size = 768
            config_template = base_model_template
            if 'wav2vec2.masked_spec_embed' in state_dict:
                if state_dict['wav2vec2.masked_spec_embed'].shape[0] == 1024:
                    hidden_size = 1024
                    config_template = large_model_template
            
            config = Wav2Vec2Config.from_pretrained(config_template)
            
            # Detect labels
            if 'classifier.weight' in state_dict:
                config.num_labels = state_dict['classifier.weight'].shape[0]
                config.classifier_proj_size = state_dict['classifier.weight'].shape[1]

            model = Wav2Vec2ForSequenceClassification(config)
            model.load_state_dict(state_dict, strict=False)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config_template)
            
            print("✅ Loaded local model.")
            return feature_extractor, model
        except Exception as e:
            print(f"⚠️ Local load failed: {e}")

    # Fallback
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(fallback_model_id)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(fallback_model_id)
        return feature_extractor, model
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None, None

# --- 2. TEXT PROCESSING ---

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_match_score(cv_text, job_text, model):
    embeddings = model.encode([cv_text, job_text])
    score = util.cos_sim(embeddings[0], embeddings[1])
    return score.item() * 100

# --- 3. TTS ---

async def _generate_audio_async(text, output_file):
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)

def generate_question_audio(text, output_path="question.mp3"):
    try:
        asyncio.run(_generate_audio_async(text, output_path))
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

# --- 4. SCORING ALGORITHMS ---

def analyze_audio_features(audio_path):
    """
    Confidence Score using exact logic from your notebook snippet.
    Manual tensor creation to avoid FeatureExtractor's hidden normalization.
    """
    results = {
        "final_score": 0, "emotion_label": "N/A", "emotion_confidence": 0,
        "avg_energy": 0, "silence_ratio": 0
    }
    
    try:
        feature_extractor, model = load_emotion_model()
        if not model:
            return results

        # 1. Load Audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Calculate heuristics just for display (optional)
        results["avg_energy"] = np.mean(librosa.feature.rms(y=y))
        
        # 2. Preprocessing (Pad or Slice)
        target_length = 48000  # 3 seconds * 16000 Hz
        
        chunks = []
        if len(y) > target_length:
            num_chunks = len(y) // target_length
            chunks = [y[i*target_length : (i+1)*target_length] for i in range(num_chunks)]
        else:
            padding = target_length - len(y)
            chunks = [np.pad(y, (0, padding), 'constant')]
        
        scores = []
        
        # 3. Prediction Loop
        model.eval()
        
        for chunk in chunks:
            # Normalize (EXACTLY as per your snippet)
            # This is Peak Normalization, NOT the Zero-Mean normalization that HF's extractor does.
            if np.max(np.abs(chunk)) > 0:
                chunk = chunk / np.max(np.abs(chunk))
            
            # Manual Tensor Creation (Bypassing Feature Extractor to prevent double-normalization)
            input_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(input_tensor).logits
                
                # Extract prediction
                if model.config.num_labels == 1:
                    # FIX: Apply Sigmoid with Gain to spread the distribution
                    # Raw logits near 0 give 0.5. Multiplying by 10 makes small deviations significant.
                    logit_val = logits[0][0].item()
                    prediction = torch.sigmoid(torch.tensor(logit_val * 10.0)).item()
                else:
                    # Fallback for multi-class
                    probs = F.softmax(logits, dim=-1)
                    prediction = torch.max(probs).item()
                
                scores.append(prediction)

        # 4. Final Result
        if scores:
            avg_score = sum(scores) / len(scores)
            
            # Map 0-1 to 0-100 for the UI
            results["final_score"] = avg_score * 100
            
            # Set label for display
            results["emotion_label"] = "Confidence" if avg_score > 0.5 else "Uncertain"
            results["emotion_confidence"] = avg_score
        
        return results

    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"final_score": 0, "error": str(e)}

def assess_interview_response(audio_path, question_text, asr_pipe, embedding_model, context_text=None):
    # 1. Transcribe
    transcription_result = asr_pipe(audio_path)
    transcript = transcription_result["text"]

    # 2. Technical Score (Scaled Cosine Similarity)
    emb_question = embedding_model.encode(question_text)
    emb_answer = embedding_model.encode(transcript)
    
    # Raw cosine similarity (typically 0.1 to 0.7 for sentences)
    raw_sim = util.cos_sim(emb_question, emb_answer).item()
    
    # FIX: Boost scaling. Map 0.5 to 100% instead of 0.75.
    # A score of 0.25 (weak match) now becomes 50%.
    tech_score = (raw_sim / 0.50) * 100
    
    # Clamp 0-100
    tech_score = min(max(tech_score, 0), 100)
    
    # 3. Confidence
    audio_metrics = analyze_audio_features(audio_path)

    return {
        "transcript": transcript,
        "technical_relevance": tech_score,
        "confidence_score": audio_metrics.get("final_score", 0),
        "details": audio_metrics
    }