# 📺 YouTube Video Summarizer & Chat with Groq LLaMA

---

## 📌 Overview

This is a **Streamlit application** that:
- 📥 Downloads YouTube videos
- 🎙 Extracts audio and transcribes it with **AssemblyAI**
- 🧠 Summarizes the video (Brief, Detailed, or Time-Aligned)
- 🔍 Creates embeddings for semantic search using CLIP & text models
- 💬 Lets you **chat with the video** using **Groq LLaMA**
- 📄 Exports summaries as downloadable PDFs

---

## ✨ Features

✅ YouTube video ingestion with transcription  
✅ CLIP & text embeddings for semantic search  
✅ Three types of summaries:
- **Brief Summary**
- **Detailed Explanation**
- **Time-Aligned Breakdown**  
✅ Chat with the video content  
✅ PDF download of summaries  

---

🖥 Application Flow
Tab 1 – Ingest Video
Enter YouTube video URL

Select summary type:

Brief Summary

Detailed Explanation

Time-Aligned Breakdown

Click "Ingest Video"

View summary and download PDF

Tab 2 – Chat
Type a question about the ingested video

Get context-aware answers from Groq LLaMA


---


A[YouTube URL] --> B[Download Video & Audio]
    B --> C[Extract Audio]
    C --> D[Transcribe with AssemblyAI]
    D --> E[Extract Frames]
    E --> F[Generate CLIP Embeddings]
    D --> G[Generate Text Embeddings]
    F & G --> H[Fuse Embeddings]
    H --> I[Save to JSON & Vector Store]
    D --> J[Summarization (Groq LLaMA)]
    J --> K[Generate PDF]
    I --> L[Chat Interface]
    L --> M[Groq LLaMA Q&A]
