# You_Tube-chat-


This project builds a YouTube Video Question-Answering system using LangChain, FAISS, and Google Gemini.
It automatically fetches a video’s transcript, splits it into chunks, generates embeddings, and stores them in a vector database.
Using a retriever + LLM pipeline, the system can answer questions about the video, summarize it, or extract specific information.

 #Features

Fetches YouTube transcripts programmatically

Creates text chunks using LangChain’s RecursiveCharacterTextSplitter

Generates embeddings with Google Generative AI

Stores vectors locally with FAISS

Builds a RAG (Retrieval-Augmented Generation) pipeline

Answers user queries about the video using Gemini 2.5 Pro
