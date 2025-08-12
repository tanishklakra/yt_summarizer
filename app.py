import streamlit as st
from utils.downloader import download_video_audio
from utils.audio import extract_audio
from utils.transcribe import transcribe_with_assemblyai
from utils.frames import extract_frames
from utils.embeddings import get_clip_embedding, get_text_embedding
from utils.fusion import fuse_embeddings
from utils.storage import save_embedding_to_json
from utils.vectorstore import get_or_create_collection, add_video_to_collection
from utils.pdf import generate_pdf_summary, generate_detailed_pdf, generate_breakdown_pdf
import hashlib
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_answer_groq(context_text, chat_history, question):
    messages = chat_history + [
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
    ]
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"Groq API error {response.status_code}: {response.text}")
        return "Failed to generate answer."

def generate_summary_groq(transcript_text):
    prompt = f"""
You are an assistant that summarizes YouTube video transcripts.

Please provide a concise and informative summary of the following transcript:

{transcript_text}

Summary:
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"Groq API error {response.status_code}: {response.text}")
        return "Failed to generate summary."

def generate_detailed_explanation(transcript_text):
    prompt = f"""
You are an AI assistant. Read the transcript of a YouTube video below and generate a detailed explanation that expands on all important points, examples, and logic used.

Transcript:
{transcript_text}

Detailed Explanation:
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"Groq API error {response.status_code}: {response.text}")
        return "Failed to generate detailed explanation."

def generate_time_aligned_breakdown(transcript_text):
    prompt = f"""
Read the following YouTube video transcript and generate a time-aligned breakdown of key sections. Format it as:

[00:00] Introduction: ...
[01:15] Key Concept 1: ...
[03:45] Example and Use Case: ...
...

Transcript:
{transcript_text}
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"Groq API error {response.status_code}: {response.text}")
        return "Failed to generate time-aligned breakdown."

def ingest_video(url, summary_type):
    video_id = hashlib.md5(url.encode()).hexdigest()
    video_dir = os.path.join("data/videos", video_id)
    os.makedirs(video_dir, exist_ok=True)

    with st.spinner("Downloading video and extracting audio..."):
        video_path, _ = download_video_audio(url)
        audio_path = extract_audio(video_path, video_id)

    with st.spinner("Transcribing audio with AssemblyAI..."):
        transcription_result = transcribe_with_assemblyai(audio_path)
        if not transcription_result:
            st.error("Transcription failed.")
            return None, None, None, None, None
        text = transcription_result

    with st.spinner("Extracting frames and computing embeddings..."):
        frame_paths = extract_frames(video_path, video_id, interval_sec=5)
        frame_embeddings = [get_clip_embedding(p) for p in frame_paths]
        text_embedding = get_text_embedding(text)
        video_embedding = fuse_embeddings(text_embedding, frame_embeddings)

    save_embedding_to_json(video_id, video_embedding, {
        "url": url,
        "text": text,
        "title": "Unknown Title"
    })

    client, collection = get_or_create_collection()
    add_video_to_collection(client, collection, video_id, video_embedding, {
        "url": url,
        "text": text,
        "title": "Unknown Title"
    })

    if summary_type == "Brief Summary":
        summary = generate_summary_groq(text)
        pdf_path = generate_pdf_summary(video_id, url, summary)
    elif summary_type == "Detailed Explanation":
        summary = generate_detailed_explanation(text)
        pdf_path = generate_detailed_pdf(video_id, summary)
    elif summary_type == "Time-Aligned Breakdown":
        raw_summary = generate_time_aligned_breakdown(text)
        summary = '\n\n'.join(line.strip() for line in raw_summary.split('\n') if line.strip())
        pdf_path = generate_breakdown_pdf(video_id, summary)
    else:
        summary = "Invalid summary type."
        pdf_path = None

    return text, video_embedding, summary, pdf_path, video_id

def answer_question(collection, question, chat_history):
    q_emb = get_text_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )
    context = "\n\n".join(doc[0] if isinstance(doc, list) else doc for doc in results['documents']) if results['documents'] else "No relevant context found."
    return generate_answer_groq(context, chat_history, question)

# Streamlit UI
st.title("YouTube Video Summarizer & Chat with Groq LLaMA")

_, collection = get_or_create_collection()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "transcription" not in st.session_state:
    st.session_state.transcription = None

if "embedding" not in st.session_state:
    st.session_state.embedding = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

tab1, tab2 = st.tabs(["Ingest Video", "Chat"])

with tab1:
    st.header("Ingest a YouTube Video")
    summary_option = st.radio("Choose summary type:", ["Brief Summary", "Detailed Explanation", "Time-Aligned Breakdown"])
    video_url = st.text_input("Enter YouTube video URL")
    if st.button("Ingest Video"):
        if not video_url.strip():
            st.error("Please enter a valid URL.")
        else:
            transcription, embedding, summary, pdf_path, video_id = ingest_video(video_url.strip(), summary_option)
            if transcription:
                st.success("Video ingested successfully!")
                st.session_state.transcription = transcription
                st.session_state.embedding = embedding
                st.session_state.summary = summary
                st.session_state.chat_history = []
                st.session_state.video_id = video_id
                st.write(summary)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("📄 Download Summary PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

with tab2:
    st.header("Chat with the Video")
    if st.session_state.embedding is None:
        st.info("Please ingest a video first in Tab 1.")
    else:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Groq LLaMA:** {msg['content']}")

        user_input = st.text_input("Ask a question")
        if st.button("Send") and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                answer = answer_question(collection, user_input.strip(), st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
