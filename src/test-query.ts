#!/usr/bin/env node
/**
 * Interactive test script - ask your own questions!
 */

// Load environment variables from .env file
import { config } from 'dotenv';
config();

import { loadEvidenceFiles, createVectorStore, answerWithCitations } from './index.js';
import { join } from 'path';

async function askQuestion(question: string) {
  console.log('Loading evidence and creating vector store...\n');
  
  const evidenceDir = join(process.cwd(), 'evidence');
  const evidenceData = await loadEvidenceFiles(evidenceDir);
  const vectorStore = await createVectorStore(evidenceData);
  
  console.log(`Question: ${question}\n`);
  console.log('Generating answer...\n');
  
  const answer = await answerWithCitations(question, vectorStore);
  
  console.log('Answer:');
  console.log('='.repeat(80));
  console.log(answer);
  console.log('='.repeat(80));
}

// Get question from command line or use default
const question = process.argv[2] || 'What color was the car?';
askQuestion(question);
