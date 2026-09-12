# RAG Profile Matching Engine

A RAG-based resume matching system that ranks candidates against job descriptions using semantic search, keyword matching, and must-have requirement filtering.

## Features

- PDF resume text extraction
- Intelligent resume chunking
- Resume metadata extraction
- OpenAI embeddings through OpenRouter
- ChromaDB vector database
- Semantic search
- Hybrid semantic + keyword search
- Must-have requirement filtering
- Candidate scoring from 0-100
- Match reasoning and relevant excerpts
- Top-10 candidate ranking
- JSON result generation

## Dataset

- 35 resumes
- 5 job descriptions
- 328 resume chunks

Job descriptions:

1. Python Developer
2. ML Engineer
3. .NET Backend Developer
4. Data Engineer
5. DevOps Engineer

## Architecture

```text
Resume PDFs
    |
    v
Text Extraction
    |
    v
Document Chunking
    |
    v
Embeddings
    |
    v
ChromaDB
    |
    | <--- Job Description
    v
Semantic Retrieval
    |
    v
Keyword Matching
    |
    v
Must-Have Filtering
    |
    v
Candidate Scoring
    |
    v
Top 10 Matches

## LangGraph Agent

The project also includes a conversational resume matching agent implemented using LangGraph.

The agent maintains an `AgentState` containing:

- Conversation history
- Job description
- Extracted requirements
- Candidate shortlist
- Candidate reasoning
- Final report
- Human feedback

### Agent Workflow

```text
START
  |
  v
Parse JD
  |
  v
Extract Requirements
  |
  v
Search Resumes
(RAG + ChromaDB)
  |
  v
Rank Candidates
  |
  v
Generate Report
  |
  v
Human Feedback / Refinement
  |
  v
Deep Candidate Analysis
  |
  v
Final Screening Recommendation
  |
  v
END