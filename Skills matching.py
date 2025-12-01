
import pandas as pd
import pdfplumber
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

"""# Skills Matching

def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()


def preprocess_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())
    email=[token for token in doc if token.like_email]
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.like_email and not token.like_url  and not token.like_num and not token.is_space]
    return tokens
def skill_matching(cv_text, skills):
    nlp = spacy.load('en_core_web_sm')
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp(skill) for skill in skills]
    matcher.add("TechSkills", patterns)


    doc = nlp(" ".join(cv_text))
    matches = matcher(doc)
    cv_skills = list(set([doc[start:end].text for match_id, start, end in matches]))

    overlap = set(cv_skills) & set(skills)
    acceptance = len(list(overlap)) / len(skills)

    return cv_skills, overlap, acceptance

"""# Embedding"""

def vectorization(cv_text, job_description,acceptance):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    cv_vec = model.encode(cv_text, convert_to_tensor=True)
    job_vec = model.encode(job_description, convert_to_tensor=True)
    return job_vec,cv_vec
def similarity_calculation(cv_vec, job_vec,acceptance, Eweight=0.7,SKweight=0.3):
    similarity = util.cos_sim(cv_vec, job_vec).item()
    final= ((Eweight * similarity) + (SKweight * acceptance) )*100
    return final
