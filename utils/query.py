import os
import openai
from utils.vectorstore import get_or_create_collection
from app import generate_answer_groq
import streamlit as st

openai.api_key = os.getenv("OPENAI_API_KEY")

def query_chroma(collection, query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return results

def generate_answer(context_text, question):
    prompt = f"""
You are a helpful assistant. Based on the following context, answer the question:

Context:
{context_text}

Question:
{question}

Answer:
"""
    response = openai.Completion.create(
        engine="llama-3.3-70b-versatile",  # replace with your desired engine or Groq llama integration
        prompt=prompt,
        max_tokens=256,
        temperature=0.3,
        n=1,
    )
    return response.choices[0].text.strip()


def answer_question(collection, question, chat_history):
    from utils.embeddings import get_text_embedding

    q_emb = get_text_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=10,
        include=['documents', 'metadatas', 'distances']
    )

    current_vid = st.session_state.video_id
    filtered_docs = []

    for doc_list, meta in zip(results['documents'], results['metadatas']):
        if meta.get("video_id") == current_vid:
            if isinstance(doc_list, list):
                filtered_docs.extend(doc_list)
            else:
                filtered_docs.append(doc_list)

    context = "\n\n".join(filtered_docs) if filtered_docs else "No relevant context found."
    return generate_answer_groq(context, chat_history, question)

if __name__ == "__main__":
    collection = get_or_create_collection()
    print("Enter 'exit' to quit.")
    while True:
        q = input("\n❓ Ask a question: ").strip()
        if q.lower() == "exit":
            break
        ans = answer_question(collection, q)
        print("\n🤖 Answer:\n", ans)
