import streamlit as st
import os
import tempfile
import time
import glob
import hashlib
from groq import Groq
import utils

# --- CONFIGURATION ---
MAX_QUESTIONS = 5 # Limit for interview questions
st.set_page_config(page_title="AI Recruiter", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "phase" not in st.session_state:
    st.session_state.phase = "cv_screening"  # Options: 'cv_screening', 'interview', 'results', 'rejected'

if "interview_scores" not in st.session_state:
    st.session_state.interview_scores = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = "Tell me about yourself and your technical background."

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API CONFIG (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    
    st.info(f"Interview Limit: {MAX_QUESTIONS} questions.")
    if st.button("Reset App"):
        # Cleanup audio files on reset
        for f in glob.glob("q_*.mp3"):
            try: os.remove(f)
            except: pass
        st.session_state.clear()
        st.rerun()

# =========================================================
# PHASE 1: CV SCREENING
# =========================================================
if st.session_state.phase == "cv_screening":
    st.title("📄 CV & Job Matching Portal")
    st.markdown("Upload a CV and Job Description to begin.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Job Description")
        job_desc_input = st.text_area("Paste Job Description here:", height=300)
    with col2:
        st.subheader("2. Applicant CV")
        uploaded_cv = st.file_uploader("Upload CV (PDF)", type=["pdf"])

    if st.button("Analyze Application"):
        if job_desc_input and uploaded_cv:
            with st.spinner("Parsing and Matching..."):
                cv_model = utils.load_cv_model()
                cv_text = utils.extract_text_from_pdf(uploaded_cv)
                cleaned_cv = utils.clean_text(cv_text)
                cleaned_job = utils.clean_text(job_desc_input)
                
                # Store text for Interview Context
                st.session_state.cv_text = cleaned_cv
                st.session_state.job_text = cleaned_job

                match_score = utils.calculate_match_score(cleaned_cv, cleaned_job, cv_model)
                st.metric("Match Score", f"{match_score:.2f}%")
                
                THRESHOLD = 50.0 
                if match_score >= THRESHOLD:
                    st.success("✅ Application Accepted! Starting Interview...")
                    time.sleep(2)
                    st.session_state.phase = "interview"
                    st.rerun()
                else:
                    st.error("❌ Application Rejected. Low Match Score.")
                    st.session_state.phase = "rejected"
        else:
            st.warning("Please provide both Job Description and CV.")

elif st.session_state.phase == "rejected":
    st.title("Application Status")
    st.error("Thank you. Unfortunately, your profile does not match our requirements.")

# =========================================================
# PHASE 2: LIVE INTERVIEW (Voice + Limits)
# =========================================================
elif st.session_state.phase == "interview":
    # 1. Check Limits
    if st.session_state.question_count >= MAX_QUESTIONS:
        st.session_state.phase = "results"
        st.rerun()

    st.title(f"🎙️ Interview (Question {st.session_state.question_count + 1}/{MAX_QUESTIONS})")
    
    # 2. Display & Speak Question
    current_q = st.session_state.current_question
    st.markdown(f"### 🤖 AI: {current_q}")
    
    # Audio Container - Use st.empty() to force refresh the player element
    audio_player_container = st.empty()
    
    # Generate Audio Logic
    # FIX: Use Hash of text in filename to ensure unique audio for every unique question
    q_hash = hashlib.md5(current_q.encode()).hexdigest()[:8]
    audio_file = f"q_{st.session_state.question_count}_{q_hash}.mp3"
    
    should_generate = True
    
    # Check if file exists and is valid (not empty)
    if os.path.exists(audio_file):
        if os.path.getsize(audio_file) > 500: # Check for reasonable size (>500 bytes)
            should_generate = False
        else:
            try: os.remove(audio_file)
            except: pass

    if should_generate:
        with st.spinner("Generating voice..."):
            try:
                utils.generate_question_audio(current_q, audio_file)
                time.sleep(0.2) # Short pause to ensure file write buffer closes
            except Exception as e:
                print(f"⚠️ TTS Generation Failed: {e}")
    
    # Play Audio
    if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
        # Open as bytes to force Streamlit to see it as "new" data
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            # Render audio in the empty container
            audio_player_container.audio(audio_bytes, format="audio/mp3", autoplay=True)

    # 3. Audio Input
    st.write("---")
    st.info("Click 'Record' to answer. To skip this question (or if no mic), click 'Skip'.")
    
    col_input, col_skip = st.columns([3, 1])
    
    with col_input:
        # Unique key for every question to reset the widget state
        audio_answer = st.audio_input("Record your answer", key=f"audio_q{st.session_state.question_count}")

    with col_skip:
        st.write("") # Spacer
        st.write("") 
        if st.button("Skip / No Answer ⏭️"):
            # Mark as skipped
            st.session_state.interview_scores.append({
                "question": current_q,
                "transcript": "SKIPPED",
                "technical_relevance": 0,
                "confidence_score": 0,
                "details": {}
            })
            st.session_state.messages.append({"role": "assistant", "content": current_q})
            st.session_state.messages.append({"role": "user", "content": "No answer provided."})
            
            # --- NEXT QUESTION LOGIC (SKIP) ---
            static_questions = [
                "Tell me about yourself and your technical background.",
                "Describe a challenging technical problem you solved.",
                "What are your strengths and weaknesses?",
                "Where do you see yourself in 5 years?",
                "Do you have any questions for us?"
            ]
            
            # Default to static
            next_idx = st.session_state.question_count + 1
            if next_idx < len(static_questions):
                 next_q_static = static_questions[next_idx]
            else:
                 next_q_static = "Thank you. We are done."

            # Try Dynamic if Key exists
            next_q = next_q_static
            if groq_api_key and groq_api_key.strip():
                try:
                    client = Groq(api_key=groq_api_key)
                    msgs = [{"role": "system", "content": f"Context: Job is {st.session_state.job_text[:300]}. User skipped last question. Ask another technical question."}]
                    msgs.extend(st.session_state.messages[-4:]) 
                    completion = client.chat.completions.create(messages=msgs, model="llama-3.1-8b-instant")
                    next_q = completion.choices[0].message.content
                except:
                    pass # Keep static fallback
            
            st.session_state.current_question = next_q
            st.session_state.question_count += 1
            st.rerun()

    # 4. Processing Answer (If Recorded)
    if audio_answer:
        with st.spinner("Analyzing your response..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_answer.read())
                tmp_path = tmp_audio.name
            
            try:
                # Run AI Analysis
                asr_pipe = utils.load_asr_pipeline()
                cv_model = utils.load_cv_model()
                result = utils.assess_interview_response(tmp_path, current_q, asr_pipe, cv_model)
                
                # Feedback to user (Debug info)
                st.subheader("Analysis of your response:")
                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{result['confidence_score']:.1f}/100")
                c2.metric("Tech Relevance", f"{result['technical_relevance']:.1f}/100")
                emotion = result['details'].get('emotion_label', 'N/A')
                c3.metric("Detected Tone", f"{emotion}")
                
                with st.expander("Show Transcript"):
                    st.write(result['transcript'])
                
                # Save Score
                st.session_state.interview_scores.append({
                    "question": current_q,
                    "transcript": result["transcript"],
                    "technical_relevance": result["technical_relevance"],
                    "confidence_score": result["confidence_score"],
                    "details": result['details']
                })
                
                # Update Chat History
                st.session_state.messages.append({"role": "assistant", "content": current_q})
                st.session_state.messages.append({"role": "user", "content": result["transcript"]})
                
                # --- NEXT QUESTION LOGIC (ANSWERED) ---
                static_questions = [
                    "Tell me about yourself and your technical background.",
                    "Describe a challenging technical problem you solved.",
                    "What are your strengths and weaknesses?",
                    "Where do you see yourself in 5 years?",
                    "Do you have any questions for us?"
                ]
                
                # Default to static
                next_idx = st.session_state.question_count + 1
                if next_idx < len(static_questions):
                     next_q_static = static_questions[next_idx]
                else:
                     next_q_static = "Thank you. We are done."
                
                # Try Dynamic if Key exists
                next_q = next_q_static
                if groq_api_key and groq_api_key.strip():
                    try:
                        client = Groq(api_key=groq_api_key)
                        sys_prompt = f"""You are a technical interviewer. 
                        Job Description: {st.session_state.job_text[:500]}
                        Last Answer: {result['transcript']}
                        Ask a follow-up or new technical question. Keep it short (1 sentence)."""
                        
                        completion = client.chat.completions.create(
                            messages=[{"role": "system", "content": sys_prompt}],
                            model="llama-3.1-8b-instant"
                        )
                        next_q = completion.choices[0].message.content
                    except Exception as e:
                        st.error(f"Groq Error: {e}")
                
                time.sleep(3) # Give user time to see scores
                
                # Update State Variables BEFORE Rerun
                st.session_state.current_question = next_q
                st.session_state.question_count += 1
                
                # Explicit rerun ensures the UI updates to the next question
                st.rerun()
            
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                # We do NOT rerun here so the user can see the error
            
            finally:
                # Cleanup temp file
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

# =========================================================
# PHASE 3: RESULTS
# =========================================================
elif st.session_state.phase == "results":
    st.title("📋 Interview Results Summary")
    st.balloons()
    
    scores = st.session_state.interview_scores
    if scores:
        avg_tech = sum(s['technical_relevance'] for s in scores) / len(scores)
        avg_conf = sum(s['confidence_score'] for s in scores) / len(scores)
        
        c1, c2 = st.columns(2)
        c1.metric("Overall Technical Score", f"{avg_tech:.1f}/100")
        c2.metric("Overall Confidence Score", f"{avg_conf:.1f}/100")
        
        st.subheader("Detailed Breakdown")
        for i, item in enumerate(scores):
            with st.expander(f"Q{i+1}: {item['question']}"):
                st.write(f"**Answer:** {item['transcript']}")
                st.write(f"**Tech Score:** {item['technical_relevance']:.1f}")
                st.write(f"**Confidence:** {item['confidence_score']:.1f}")
                
                # Show detected emotion if available
                if 'details' in item and 'emotion_label' in item['details']:
                    st.info(f"Detected Tone: {item['details']['emotion_label']}")
    else:
        st.write("No questions answered.")

    if st.button("Start New Interview"):
        st.session_state.clear()
        st.rerun()