# LLM Document Search & Analysis System

This project implements a Retrieval-Augmented Generation (RAG) pipeline for large document collections.
It ingests raw text, chunks it into overlapping segments, embeds each chunk, and stores the vectors in a FAISS index.
The system exposes APIs for semantic search and context-grounded answer generation.
