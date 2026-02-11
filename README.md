# 📚 Semantic Book Recommender 

AI-powered book recommendation system that finds books based on:
- Semantic similarity (description search using OpenAI embeddings & Chroma vector DB)
- Genre filtering
- Emotional tone/mood (joy, surprise, anger, suspense, sadness)

Built with **Gradio** (beautiful UI), **LangChain + Chroma** (vector search), and **Docker** (containerized).

Live demo (if deployed): https://huggingface.co/spaces/YOURUSERNAME/semantic-book-recommender  
(or Render/Fly.io link when you add it)

## Features
- Natural language search: "A story about forgiveness in a small town"
- Genre dropdown: All / Fiction / Nonfiction
- Mood selection: Happy / Surprise / Angry / Suspenseful / Sad
- Beautiful book gallery with covers, titles & short descriptions
- Persistent Chroma vector database (data survives restarts)
- Secure API key handling with Docker secrets

## Tech Stack
- **Frontend/UI**: Gradio
- **Backend**: Python 3.11
- **Vector Search**: LangChain + ChromaDB + OpenAI Embeddings
- **Containerization**: Docker + Docker Compose
- **Data**: Google Books-derived dataset with emotion analysis

## Future Improvements (Ideas)
-GPU acceleration for embeddings
-Better error messages & loading statescd semantic-book-recommender
