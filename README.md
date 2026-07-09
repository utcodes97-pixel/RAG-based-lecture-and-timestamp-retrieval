# RAG-based Lecture and Timestamp Retrieval

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions over lecture/video content and get answers grounded in the actual spoken content — along with the relevant chunks it was pulled from.

The pipeline converts videos into transcribed, chunked, embedded text, stores it locally, and uses a local LLM (via Ollama) to answer natural language questions using only the most relevant retrieved chunks.

## Tech Stack

- **Video → Audio**: `moviepy` / `ffmpeg`
- **Speech-to-Text**: OpenAI Whisper
- **Embeddings**: `bge-m3` (base model) via Ollama
- **LLM**: LLaMA 3.2 / DeepSeek-R1 (1.5B, quantized) via Ollama
- **Storage**: Pandas DataFrame, persisted with `joblib`

## Pipeline Overview

This project is built step-by-step as a series of notebooks, each handling one stage of the RAG pipeline — from raw video to a working Q&A system.

### Step 1: Videos to MP3
Extracts audio from input video files and converts it into `.mp3` format using `moviepy`/`ffmpeg`, preparing it for transcription.

### Step 2: Audio to Text (Overview)
Transcribes the extracted audio into text using OpenAI Whisper. Produces the raw text output that will later be chunked and processed.

### Step 3: Chunking of Audio Files to Text Data with Metadata
Splits the full transcripts into smaller, manageable chunks and attaches metadata to each chunk (e.g. source file, timestamp) so that answers can later be traced back to their original location in the video.

### Step 4: Creating Embeddings and Storing in a DataFrame
Generates vector embeddings for each text chunk using the `bge-m3` embedding model (via Ollama) and stores them alongside the chunk text and metadata in a Pandas DataFrame.

### Step 5: Pulling Top Matching Chunks
Given a user query, computes similarity between the query embedding and stored chunk embeddings to retrieve the most relevant chunks.

### Step 6: Saving Embeddings using Joblib (Persisting Embeddings)
Persists the embeddings DataFrame to disk using `joblib`, so the embedding step doesn't need to be re-run every time the pipeline is used.

### Step 7: Loading DataFrame using Joblib and Viewing Top Matching Chunks
Loads the persisted embeddings DataFrame back into memory and retrieves the top matching chunks for a given user query — validating that persistence and retrieval work correctly end-to-end.

### Step 8: Creating a Prompt for LLM from User Query
Constructs a well-structured prompt that combines the user's query with the retrieved top-matching chunks, preparing the final input to be sent to the LLM.

### Step 9: Getting Response from LLM
Sends the constructed prompt to a local LLM (LLaMA 3.2 or DeepSeek-R1 via Ollama) and returns a grounded, context-aware answer based on the retrieved lecture content.

## How It Works (End-to-End Flow)

```
Video → MP3 → Transcript → Chunks + Metadata → Embeddings → Stored DataFrame (joblib)
                                                                     │
                                                     User Query ─────┤
                                                                     ▼
                                                        Top Matching Chunks
                                                                     │
                                                                     ▼
                                                       Prompt Construction
                                                                     │
                                                                     ▼
                                                          LLM Response (Ollama)
```

## Notes

- Embedding and LLM inference are run locally via **Ollama**, using lightweight models (`bge-m3` base, and a 1.5B-parameter LLM) to accommodate limited hardware specs.
- Each step is implemented as a standalone notebook, making the pipeline easy to follow, debug, and extend stage by stage.

## Status

Actively being built — Steps 1 through 9 are complete, covering the full pipeline from raw video input to LLM-generated answers.
