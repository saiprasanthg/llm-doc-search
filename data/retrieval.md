# Retrieval + Generation Workflow

The API retrieves the most relevant chunks using vector similarity.
Those chunks are used as context for a large language model to generate an answer.
If the answer cannot be found in the retrieved context, the system responds that it lacks enough information.
