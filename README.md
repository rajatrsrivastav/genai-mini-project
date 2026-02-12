# Cold Case Detective 🔍

A RAG (Retrieval Augmented Generation) system that analyzes cold case evidence and answers questions with proper source citations.

## Features

- 📁 Source tracking with explicit citations
- 🔍 Local vector embeddings (Xenova/all-MiniLM-L6-v2)
- 🤖 LLM integration (Gemini/OpenAI)

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd cold-case-detective
npm install
```

### 2. Set Up API Key

Get a free Gemini API key: https://makersuite.google.com/app/apikey

```bash
cp .env.example .env
```

Edit `.env` and add your key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Build

```bash
npm run build
```

### 4. Use the App

```bash
npm run ask "What color was the car?"
npm run ask "Who witnessed the crime?"
npm run ask "What evidence connects the suspect to the crime scene?"
```

## Usage

```bash
# Ask questions about the case
npm run ask "What color was the car?"
npm run ask "Who witnessed the crime?"
npm run ask "What evidence connects the suspect to the crime scene?"
```

## Example Output

```
Question: What color was the car?

Answer:
================================================================================
The car was red.

According to vehicle_report.txt, the red paint chips were analyzed...
According to witness_statement_1.txt, the witness saw a red sedan...
According to forensics_report.txt, red paint chips were found...
================================================================================
```

## Project Structure

```
cold-case-detective/
├── evidence/         # 5 case evidence text files
├── src/
│   ├── index.ts     # Main RAG implementation
│   └── test-query.ts # Interactive query script
└── package.json
```

## How It Works

1. **Data Ingestion**: Loads evidence files with source tracking
2. **Vector Search**: Creates embeddings and finds relevant chunks
3. **Citation Pipeline**: LLM generates answers with "According to [filename]..." citations

## Tech Stack

- TypeScript
- Xenova/all-MiniLM-L6-v2 (embeddings)
- Gemini 2.5 Flash / OpenAI GPT-3.5 (LLM)

## Troubleshooting

**Module not found?** Run `npm run build`

**API errors?** Check your API key at https://makersuite.google.com/app/apikey

## Requirements

- Node.js 18+
- Gemini or OpenAI API key

## License

MIT
