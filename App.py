import streamlit as st
import requests
import pandas as pd
import pdfplumber
import docx2txt
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer, util
import googleapiclient.discovery
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Predefined list of technical skills for extraction
TECH_SKILLS = {"machine learning", "deep learning", "data science", "python", "sql", "tensorflow", "keras", "nlp", "power bi", "tableau", "pandas", "numpy", "big data", "AI", "cloud computing", "docker", "kubernetes", "flask", "django", "fastapi", "data visualization"}

# Load AI Model
st_model = SentenceTransformer("all-MiniLM-L6-v2")

# YouTube API Key (Use a secure way to store API keys)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Use environment variables
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Function to extract text from files
def extract_text(uploaded_file):
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif ext in ["docx", "doc"]:
            return docx2txt.process(uploaded_file)
        elif ext == "txt":
            return uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format! Please upload PDF, DOCX, or TXT.")
    return ""

# Alternative Skill Extraction Function using spaCy
def extract_skills(text):
    if not text:
        return []
    
    doc = nlp(text)
    extracted_skills = set()
    
    # Extract entities using spaCy's NER
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:  # Common labels for skills
            extracted_skills.add(ent.text.lower())
    
    # Match words from predefined skills database
    for token in doc:
        if token.text.lower() in TECH_SKILLS:
            extracted_skills.add(token.text.lower())
    
    return list(extracted_skills)

# Function to calculate similarity score
def calculate_matching_score(resume_text, job_text):
    if not resume_text or not job_text:
        return 0.0
    embeddings = st_model.encode([resume_text, job_text], convert_to_tensor=True)
    return round(float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0]), 2) * 100

# Function to plot skill comparison
def plot_skill_comparison(resume_skills, job_skills):
    if not resume_skills and not job_skills:
        return
    
    all_skills = list(set(resume_skills + job_skills))
    resume_counts = [1 if skill in resume_skills else 0 for skill in all_skills]
    job_counts = [1 if skill in job_skills else 0 for skill in all_skills]

    df = pd.DataFrame({"Skills": all_skills, "Resume": resume_counts, "Job Requirements": job_counts})
    
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", x="Skills", ax=ax, color=["blue", "red"], alpha=0.7)
    ax.set_title("Resume vs. Job Skills Comparison")
    ax.set_xticklabels(df["Skills"], rotation=45, ha="right")
    ax.set_ylabel("Presence (1 = Present, 0 = Missing)")
    st.pyplot(fig)

# Function to fetch YouTube course recommendations
def fetch_youtube_courses(skill):
    if not YOUTUBE_API_KEY:
        st.error("YouTube API Key not found!")
        return []

    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    
    request = youtube.search().list(
        q=f"{skill} tutorial",
        part="snippet",
        type="video",
        maxResults=3
    )
    
    response = request.execute()
    
    courses = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        courses.append({"Skill": skill, "Course Title": title, "Link": url})
    
    return courses

# Streamlit UI
st.title("📄 AI Resume Analyzer & Skill Enhancer")
st.write("Upload your Resume and Job Description to analyze missing skills and get YouTube course recommendations!")

# File Uploaders
resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    st.subheader("📌 Resume Summary")
    st.write(resume_text[:500] + "...")

    st.subheader("📌 Job Description Summary")
    st.write(job_text[:500] + "...")

    if st.button("Analyze Skills & Matching Score"):
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        missing_skills = list(set(job_skills) - set(resume_skills))

        st.subheader("🔍 Extracted Skills")
        st.write(f"**Resume Skills:** {', '.join(resume_skills)}")
        st.write(f"**Job Required Skills:** {', '.join(job_skills)}")

        st.subheader("📊 Resume Matching Score")
        match_score = calculate_matching_score(resume_text, job_text)
        st.write(f"Your resume matches **{match_score}%** of the job requirements.")

        st.subheader("⚠️ Missing Skills")
        if missing_skills:
            st.write(f"You are missing: {', '.join(missing_skills)}")
        else:
            st.success("You have all the required skills!")

        plot_skill_comparison(resume_skills, job_skills)

if st.button("📚 Get Recommended Courses"):
    all_courses = []
    for skill in missing_skills:
        all_courses.extend(fetch_youtube_courses(skill))
    
    if all_courses:
        df = pd.DataFrame(all_courses)
        st.table(df)
    else:
        st.error("No courses found.")
