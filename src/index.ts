// Cold Case Detective - Main entry point

import { readdir, readFile } from 'fs/promises';
import { join } from 'path';
import { pipeline } from '@xenova/transformers';

/**
 * Data object structure for evidence files
 */
export interface EvidenceData {
  content: string;
  source: string;
}

/**
 * Vector store entry containing embedding and metadata
 */
export interface VectorStoreEntry {
  embedding: number[];
  content: string;
  source: string;
}

/**
 * Simple in-memory vector store
 */
export interface VectorStore {
  entries: VectorStoreEntry[];
}

/**
 * Loads all .txt evidence files from a directory
 * @param directoryPath - Path to the directory containing evidence files
 * @returns Array of evidence data objects with content and source
 */
export async function loadEvidenceFiles(directoryPath: string): Promise<EvidenceData[]> {
  try {
    // Read all files in the directory
    const files = await readdir(directoryPath);
    
    // Filter for .txt files only
    const txtFiles = files.filter(file => file.endsWith('.txt'));
    
    if (txtFiles.length === 0) {
      throw new Error(`No .txt files found in directory: ${directoryPath}`);
    }
    
    // Load each file and create data objects
    const evidenceData: EvidenceData[] = [];
    
    for (const file of txtFiles) {
      try {
        const filePath = join(directoryPath, file);
        const content = await readFile(filePath, 'utf-8');
        
        evidenceData.push({
          content,
          source: file
        });
      } catch (error) {
        // Log warning and continue with other files
        console.warn(`Warning: Failed to read file ${file}:`, error instanceof Error ? error.message : 'Unknown error');
      }
    }
    
    return evidenceData;
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      throw new Error(`Directory not found: ${directoryPath}`);
    }
    throw error;
  }
}

/**
 * Creates a vector store from evidence data objects
 * Converts content to embeddings and stores them with source metadata
 * @param dataObjects - Array of evidence data objects
 * @returns Vector store containing embeddings with source tracking
 */
export async function createVectorStore(dataObjects: EvidenceData[]): Promise<VectorStore> {
  // Initialize the embedding model (using a small, efficient model)
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  
  const entries: VectorStoreEntry[] = [];
  
  for (const dataObject of dataObjects) {
    // Generate embedding for the content
    const output = await embedder(dataObject.content, { pooling: 'mean', normalize: true });
    
    // Convert tensor to array
    const embedding = Array.from(output.data as Float32Array);
    
    // Store embedding with content and source metadata
    entries.push({
      embedding,
      content: dataObject.content,
      source: dataObject.source
    });
  }
  
  return { entries };
}
/**
 * Search result containing retrieved content with source and similarity score
 */
export interface SearchResult {
  content: string;
  source: string;
  score: number;
}

/**
 * Calculate cosine similarity between two vectors
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine similarity score between -1 and 1
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}

/**
 * Performs similarity search on the vector store
 * @param query - Search query string
 * @param vectorStore - Vector store to search
 * @param k - Number of top results to return
 * @returns Array of top-k search results with content, source, and similarity score
 */
export async function searchSimilar(
  query: string,
  vectorStore: VectorStore,
  k: number
): Promise<SearchResult[]> {
  // Handle empty query
  if (!query || query.trim().length === 0) {
    return [];
  }

  // Handle empty vector store
  if (vectorStore.entries.length === 0) {
    return [];
  }

  // Generate embedding for the query
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  const output = await embedder(query, { pooling: 'mean', normalize: true });
  const queryEmbedding = Array.from(output.data as Float32Array);

  // Calculate similarity scores for all entries
  const resultsWithScores = vectorStore.entries.map(entry => ({
    content: entry.content,
    source: entry.source,
    score: cosineSimilarity(queryEmbedding, entry.embedding)
  }));

  // Sort by similarity score (highest first) and return top-k
  resultsWithScores.sort((a, b) => b.score - a.score);

  return resultsWithScores.slice(0, k);
}
/**
 * Builds a prompt with source-labeled context for LLM citation
 * @param query - User's question
 * @param searchResults - Retrieved chunks from vector search
 * @returns Formatted prompt with source labels and citation instructions
 */
export function buildPromptWithSources(query: string, searchResults: SearchResult[]): string {
  // Build context section with source labels
  const contextParts = searchResults.map(result =>
    `[From ${result.source}]:\n${result.content}`
  );

  const context = contextParts.join('\n\n');

  // Construct the full prompt with instructions
  const prompt = `Context:
${context}

Question: ${query}

Instructions: Answer the question based only on the provided context. Cite your sources using the format "According to [filename]..." for each fact you reference. If the context doesn't contain enough information to answer the question, say so.`;

  return prompt;
}
/**
 * Answers a query with citations using LLM and vector search
 * @param query - User's question
 * @param vectorStore - Vector store containing evidence embeddings
 * @returns LLM-generated answer with source citations
 */
export async function answerWithCitations(
  query: string,
  vectorStore: VectorStore
): Promise<string> {
  // Retrieve relevant chunks using vector search (top 3 results)
  const searchResults = await searchSimilar(query, vectorStore, 3);

  // Handle case where no relevant context is found
  if (searchResults.length === 0) {
    return 'I could not find any relevant evidence to answer your question.';
  }

  // Build prompt with source-labeled context
  const prompt = buildPromptWithSources(query, searchResults);

  // Check which API key is available
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const openaiApiKey = process.env.OPENAI_API_KEY;

  try {
    if (geminiApiKey) {
      // Use Gemini API
      const { GoogleGenerativeAI } = await import('@google/generative-ai');
      const genAI = new GoogleGenerativeAI(geminiApiKey);
      // Use Gemini 2.5 Flash (available with your API key)
      const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
      
      const result = await model.generateContent(prompt);
      const response = result.response;
      const answer = response.text();
      return answer;
    } else if (openaiApiKey) {
      // Use OpenAI API
      const { OpenAI } = await import('openai');
      const openai = new OpenAI({ apiKey: openaiApiKey });
      
      const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
        max_tokens: 500
      });

      const answer = response.choices[0]?.message?.content || 'No response generated.';
      return answer;
    } else {
      return 'Error: No API key found. Please set either GEMINI_API_KEY or OPENAI_API_KEY environment variable.';
    }
  } catch (error) {
    // Handle LLM API errors
    if (error instanceof Error) {
      return `Error calling LLM API: ${error.message}`;
    }
    return 'An unknown error occurred while calling the LLM API.';
  }
}
