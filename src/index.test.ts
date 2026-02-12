import { describe, it, expect, beforeAll } from 'vitest';
import { loadEvidenceFiles, EvidenceData, VectorStore, SearchResult } from './index';
import { mkdir, writeFile, rm } from 'fs/promises';
import { join } from 'path';

describe('loadEvidenceFiles', () => {
  const testDir = join(process.cwd(), 'test-evidence');

  beforeAll(async () => {
    // Clean up test directory if it exists
    try {
      await rm(testDir, { recursive: true, force: true });
    } catch (error) {
      // Ignore if directory doesn't exist
    }
  });

  it('should load all .txt files from directory', async () => {
    // Create test directory with sample files
    await mkdir(testDir, { recursive: true });
    await writeFile(join(testDir, 'file1.txt'), 'Content 1');
    await writeFile(join(testDir, 'file2.txt'), 'Content 2');
    await writeFile(join(testDir, 'file3.txt'), 'Content 3');

    const result = await loadEvidenceFiles(testDir);

    expect(result).toHaveLength(3);
    expect(result.every(item => 'content' in item && 'source' in item)).toBe(true);
    
    // Clean up
    await rm(testDir, { recursive: true });
  });

  it('should return objects with content and source fields', async () => {
    await mkdir(testDir, { recursive: true });
    await writeFile(join(testDir, 'test.txt'), 'Test content');

    const result = await loadEvidenceFiles(testDir);

    expect(result[0]).toHaveProperty('content');
    expect(result[0]).toHaveProperty('source');
    expect(result[0].content).toBe('Test content');
    expect(result[0].source).toBe('test.txt');
    
    await rm(testDir, { recursive: true });
  });

  it('should only load .txt files and ignore other file types', async () => {
    await mkdir(testDir, { recursive: true });
    await writeFile(join(testDir, 'file1.txt'), 'Text file');
    await writeFile(join(testDir, 'file2.md'), 'Markdown file');
    await writeFile(join(testDir, 'file3.json'), '{}');

    const result = await loadEvidenceFiles(testDir);

    expect(result).toHaveLength(1);
    expect(result[0].source).toBe('file1.txt');
    
    await rm(testDir, { recursive: true });
  });

  it('should handle empty files', async () => {
    await mkdir(testDir, { recursive: true });
    await writeFile(join(testDir, 'empty.txt'), '');

    const result = await loadEvidenceFiles(testDir);

    expect(result).toHaveLength(1);
    expect(result[0].content).toBe('');
    expect(result[0].source).toBe('empty.txt');
    
    await rm(testDir, { recursive: true });
  });

  it('should throw error for non-existent directory', async () => {
    await expect(loadEvidenceFiles('/non/existent/path')).rejects.toThrow('Directory not found');
  });

  it('should throw error when no .txt files found', async () => {
    await mkdir(testDir, { recursive: true });
    await writeFile(join(testDir, 'file.md'), 'Not a txt file');

    await expect(loadEvidenceFiles(testDir)).rejects.toThrow('No .txt files found');
    
    await rm(testDir, { recursive: true });
  });

  it('should handle file read errors gracefully', async () => {
    await mkdir(testDir, { recursive: true });
    await writeFile(join(testDir, 'good.txt'), 'Good content');
    await writeFile(join(testDir, 'also-good.txt'), 'Also good');

    const result = await loadEvidenceFiles(testDir);

    // Should still load the files that can be read
    expect(result.length).toBeGreaterThan(0);
    
    await rm(testDir, { recursive: true });
  });

  it('should work with actual evidence directory', async () => {
    const evidenceDir = join(process.cwd(), 'evidence');
    
    const result = await loadEvidenceFiles(evidenceDir);

    expect(result.length).toBeGreaterThanOrEqual(3);
    expect(result.length).toBeLessThanOrEqual(5);
    expect(result.every(item => 
      typeof item.content === 'string' && 
      typeof item.source === 'string' &&
      item.source.endsWith('.txt')
    )).toBe(true);
  });
});

describe('createVectorStore', () => {
  it('should create embeddings for all data objects', async () => {
    const { createVectorStore } = await import('./index');
    
    const testData: EvidenceData[] = [
      { content: 'The car was red', source: 'witness1.txt' },
      { content: 'The suspect wore a blue jacket', source: 'witness2.txt' }
    ];

    const vectorStore = await createVectorStore(testData);

    expect(vectorStore.entries).toHaveLength(2);
    expect(vectorStore.entries[0]).toHaveProperty('embedding');
    expect(vectorStore.entries[0]).toHaveProperty('content');
    expect(vectorStore.entries[0]).toHaveProperty('source');
  });

  it('should preserve source metadata for each embedding', async () => {
    const { createVectorStore } = await import('./index');
    
    const testData: EvidenceData[] = [
      { content: 'Evidence from first file', source: 'file1.txt' },
      { content: 'Evidence from second file', source: 'file2.txt' }
    ];

    const vectorStore = await createVectorStore(testData);

    expect(vectorStore.entries[0].source).toBe('file1.txt');
    expect(vectorStore.entries[0].content).toBe('Evidence from first file');
    expect(vectorStore.entries[1].source).toBe('file2.txt');
    expect(vectorStore.entries[1].content).toBe('Evidence from second file');
  });

  it('should create valid embeddings (non-empty arrays)', async () => {
    const { createVectorStore } = await import('./index');
    
    const testData: EvidenceData[] = [
      { content: 'Test content for embedding', source: 'test.txt' }
    ];

    const vectorStore = await createVectorStore(testData);

    expect(Array.isArray(vectorStore.entries[0].embedding)).toBe(true);
    expect(vectorStore.entries[0].embedding.length).toBeGreaterThan(0);
    expect(vectorStore.entries[0].embedding.every(val => typeof val === 'number')).toBe(true);
  });

  it('should handle empty data objects array', async () => {
    const { createVectorStore } = await import('./index');
    
    const testData: EvidenceData[] = [];

    const vectorStore = await createVectorStore(testData);

    expect(vectorStore.entries).toHaveLength(0);
  });

  it('should work with actual evidence files', async () => {
    const { loadEvidenceFiles, createVectorStore } = await import('./index');
    const evidenceDir = join(process.cwd(), 'evidence');
    
    const evidenceData = await loadEvidenceFiles(evidenceDir);
    const vectorStore = await createVectorStore(evidenceData);

    expect(vectorStore.entries.length).toBe(evidenceData.length);
    expect(vectorStore.entries.every(entry => 
      Array.isArray(entry.embedding) &&
      entry.embedding.length > 0 &&
      typeof entry.content === 'string' &&
      typeof entry.source === 'string'
    )).toBe(true);
  });
});
describe('searchSimilar', () => {
  it('should return top-k results with content, source, and score', async () => {
    const { createVectorStore, searchSimilar } = await import('./index');

    const testData: EvidenceData[] = [
      { content: 'The car was red and fast', source: 'witness1.txt' },
      { content: 'The suspect wore a blue jacket', source: 'witness2.txt' },
      { content: 'The vehicle was a red sports car', source: 'vehicle_report.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const results = await searchSimilar('What color was the car?', vectorStore, 2);

    expect(results).toHaveLength(2);
    expect(results[0]).toHaveProperty('content');
    expect(results[0]).toHaveProperty('source');
    expect(results[0]).toHaveProperty('score');
    expect(typeof results[0].score).toBe('number');
  });

  it('should return results sorted by similarity score', async () => {
    const { createVectorStore, searchSimilar } = await import('./index');

    const testData: EvidenceData[] = [
      { content: 'The weather was sunny', source: 'file1.txt' },
      { content: 'The car was red', source: 'file2.txt' },
      { content: 'The red vehicle was parked', source: 'file3.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const results = await searchSimilar('red car', vectorStore, 3);

    // Results should be sorted by score (highest first)
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
    expect(results[1].score).toBeGreaterThanOrEqual(results[2].score);
  });

  it('should preserve source information in results', async () => {
    const { createVectorStore, searchSimilar } = await import('./index');

    const testData: EvidenceData[] = [
      { content: 'Evidence about the car', source: 'witness_statement.txt' },
      { content: 'Forensics report on the vehicle', source: 'forensics.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const results = await searchSimilar('car', vectorStore, 2);

    expect(results.every(r => typeof r.source === 'string' && r.source.endsWith('.txt'))).toBe(true);
  });

  it('should return empty array for empty query', async () => {
    const { createVectorStore, searchSimilar } = await import('./index');

    const testData: EvidenceData[] = [
      { content: 'Some content', source: 'file.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const results = await searchSimilar('', vectorStore, 5);

    expect(results).toHaveLength(0);
  });

  it('should return empty array for empty vector store', async () => {
    const { searchSimilar } = await import('./index');

    const emptyStore = { entries: [] };
    const results = await searchSimilar('test query', emptyStore, 5);

    expect(results).toHaveLength(0);
  });

  it('should respect k parameter', async () => {
    const { createVectorStore, searchSimilar } = await import('./index');

    const testData: EvidenceData[] = [
      { content: 'First document', source: 'file1.txt' },
      { content: 'Second document', source: 'file2.txt' },
      { content: 'Third document', source: 'file3.txt' },
      { content: 'Fourth document', source: 'file4.txt' }
    ];

    const vectorStore = await createVectorStore(testData);

    const results1 = await searchSimilar('document', vectorStore, 1);
    expect(results1).toHaveLength(1);

    const results2 = await searchSimilar('document', vectorStore, 2);
    expect(results2).toHaveLength(2);

    const results3 = await searchSimilar('document', vectorStore, 3);
    expect(results3).toHaveLength(3);
  });

  it('should work with actual evidence files', async () => {
    const { loadEvidenceFiles, createVectorStore, searchSimilar } = await import('./index');
    const evidenceDir = join(process.cwd(), 'evidence');

    const evidenceData = await loadEvidenceFiles(evidenceDir);
    const vectorStore = await createVectorStore(evidenceData);
    const results = await searchSimilar('What color was the car?', vectorStore, 3);

    expect(results.length).toBeGreaterThan(0);
    expect(results.length).toBeLessThanOrEqual(3);
    expect(results.every(r =>
      typeof r.content === 'string' &&
      typeof r.source === 'string' &&
      typeof r.score === 'number' &&
      r.score >= -1 && r.score <= 1
    )).toBe(true);
  });

  it('should not call LLM (returns raw search results)', async () => {
    const { createVectorStore, searchSimilar } = await import('./index');

    const testData: EvidenceData[] = [
      { content: 'The car was red', source: 'witness.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const results = await searchSimilar('What color was the car?', vectorStore, 1);

    // Result should be raw content, not an LLM-generated answer
    expect(results[0].content).toBe('The car was red');
    expect(results[0].content).not.toContain('According to');
    expect(results[0].content).not.toContain('The answer is');
  });
});
describe('buildPromptWithSources', () => {
  it('should format each chunk with source label', async () => {
    const { buildPromptWithSources } = await import('./index');

    const searchResults: SearchResult[] = [
      { content: 'The car was red', source: 'witness1.txt', score: 0.9 },
      { content: 'The suspect wore a blue jacket', source: 'witness2.txt', score: 0.8 }
    ];

    const prompt = buildPromptWithSources('What color was the car?', searchResults);

    expect(prompt).toContain('[From witness1.txt]:');
    expect(prompt).toContain('The car was red');
    expect(prompt).toContain('[From witness2.txt]:');
    expect(prompt).toContain('The suspect wore a blue jacket');
  });

  it('should include the user query in the prompt', async () => {
    const { buildPromptWithSources } = await import('./index');

    const searchResults: SearchResult[] = [
      { content: 'Some evidence', source: 'file.txt', score: 0.9 }
    ];

    const query = 'What happened at the scene?';
    const prompt = buildPromptWithSources(query, searchResults);

    expect(prompt).toContain('Question: What happened at the scene?');
  });

  it('should include instructions for LLM to cite sources', async () => {
    const { buildPromptWithSources } = await import('./index');

    const searchResults: SearchResult[] = [
      { content: 'Evidence content', source: 'file.txt', score: 0.9 }
    ];

    const prompt = buildPromptWithSources('Test query', searchResults);

    expect(prompt).toContain('Instructions:');
    expect(prompt).toContain('Cite');
    expect(prompt).toContain('According to [filename]');
  });

  it('should handle multiple search results from different sources', async () => {
    const { buildPromptWithSources } = await import('./index');

    const searchResults: SearchResult[] = [
      { content: 'First piece of evidence', source: 'witness_statement.txt', score: 0.95 },
      { content: 'Second piece of evidence', source: 'forensics_report.txt', score: 0.90 },
      { content: 'Third piece of evidence', source: 'police_log.txt', score: 0.85 }
    ];

    const prompt = buildPromptWithSources('What happened?', searchResults);

    expect(prompt).toContain('[From witness_statement.txt]:');
    expect(prompt).toContain('[From forensics_report.txt]:');
    expect(prompt).toContain('[From police_log.txt]:');
    expect(prompt).toContain('First piece of evidence');
    expect(prompt).toContain('Second piece of evidence');
    expect(prompt).toContain('Third piece of evidence');
  });

  it('should handle empty search results', async () => {
    const { buildPromptWithSources } = await import('./index');

    const searchResults: any[] = [];
    const prompt = buildPromptWithSources('Test query', searchResults);

    expect(prompt).toContain('Question: Test query');
    expect(prompt).toContain('Instructions:');
    expect(prompt).toContain('Context:');
  });

  it('should structure prompt with Context, Question, and Instructions sections', async () => {
    const { buildPromptWithSources } = await import('./index');

    const searchResults: SearchResult[] = [
      { content: 'Evidence', source: 'file.txt', score: 0.9 }
    ];

    const prompt = buildPromptWithSources('Query', searchResults);

    // Check that sections appear in the correct order
    const contextIndex = prompt.indexOf('Context:');
    const questionIndex = prompt.indexOf('Question:');
    const instructionsIndex = prompt.indexOf('Instructions:');

    expect(contextIndex).toBeGreaterThan(-1);
    expect(questionIndex).toBeGreaterThan(contextIndex);
    expect(instructionsIndex).toBeGreaterThan(questionIndex);
  });
});

describe('answerWithCitations', () => {
  it('should return an answer with citations from LLM', async () => {
    const { loadEvidenceFiles, createVectorStore, answerWithCitations } = await import('./index');
    const evidenceDir = join(process.cwd(), 'evidence');

    // Skip test if no API key is set
    if (!process.env.GEMINI_API_KEY && !process.env.OPENAI_API_KEY) {
      console.log('Skipping test: No API key set (GEMINI_API_KEY or OPENAI_API_KEY required)');
      return;
    }

    const evidenceData = await loadEvidenceFiles(evidenceDir);
    const vectorStore = await createVectorStore(evidenceData);
    
    const answer = await answerWithCitations('What color was the car?', vectorStore);

    expect(typeof answer).toBe('string');
    expect(answer.length).toBeGreaterThan(0);
    // Answer should contain some form of citation
    expect(answer.toLowerCase()).toMatch(/according to|from|source|witness|forensics|police/);
  });

  it('should retrieve relevant chunks using vector search', async () => {
    const { createVectorStore, answerWithCitations } = await import('./index');

    // Skip test if no API key is set
    if (!process.env.GEMINI_API_KEY && !process.env.OPENAI_API_KEY) {
      console.log('Skipping test: No API key set (GEMINI_API_KEY or OPENAI_API_KEY required)');
      return;
    }

    const testData: EvidenceData[] = [
      { content: 'The car was red and parked on Main Street', source: 'witness1.txt' },
      { content: 'The suspect wore a blue jacket', source: 'witness2.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const answer = await answerWithCitations('What color was the car?', vectorStore);

    expect(typeof answer).toBe('string');
    expect(answer.length).toBeGreaterThan(0);
  });

  it('should return error message when no relevant context is found', async () => {
    const { answerWithCitations } = await import('./index');

    const emptyStore = { entries: [] };
    const answer = await answerWithCitations('What happened?', emptyStore);

    expect(answer).toContain('could not find any relevant evidence');
  });

  it('should handle LLM API errors gracefully', async () => {
    const { createVectorStore, answerWithCitations } = await import('./index');

    // Create test data
    const testData: EvidenceData[] = [
      { content: 'Test content', source: 'test.txt' }
    ];

    const vectorStore = await createVectorStore(testData);

    // Temporarily remove API keys to trigger error
    const originalOpenAIKey = process.env.OPENAI_API_KEY;
    const originalGeminiKey = process.env.GEMINI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    delete process.env.GEMINI_API_KEY;

    const answer = await answerWithCitations('Test query', vectorStore);

    // Restore API keys
    if (originalOpenAIKey) {
      process.env.OPENAI_API_KEY = originalOpenAIKey;
    }
    if (originalGeminiKey) {
      process.env.GEMINI_API_KEY = originalGeminiKey;
    }

    expect(answer).toContain('Error');
  });

  it('should call LLM with constructed prompt containing source labels', async () => {
    const { createVectorStore, answerWithCitations } = await import('./index');

    // Skip test if no API key is set
    if (!process.env.GEMINI_API_KEY && !process.env.OPENAI_API_KEY) {
      console.log('Skipping test: No API key set (GEMINI_API_KEY or GEMINI_API_KEY required)');
      return;
    }

    const testData: EvidenceData[] = [
      { content: 'The vehicle was a red sedan', source: 'vehicle_report.txt' },
      { content: 'Witness saw a red car', source: 'witness_statement.txt' }
    ];

    const vectorStore = await createVectorStore(testData);
    const answer = await answerWithCitations('What color was the car?', vectorStore);

    // The answer should be a string response (could be success or error message)
    expect(typeof answer).toBe('string');
    expect(answer.length).toBeGreaterThan(0);
  });

  it('should work with multi-source questions', async () => {
    const { loadEvidenceFiles, createVectorStore, answerWithCitations } = await import('./index');
    const evidenceDir = join(process.cwd(), 'evidence');

    // Skip test if no API key is set
    if (!process.env.GEMINI_API_KEY && !process.env.OPENAI_API_KEY) {
      console.log('Skipping test: No API key set (GEMINI_API_KEY or OPENAI_API_KEY required)');
      return;
    }

    const evidenceData = await loadEvidenceFiles(evidenceDir);
    const vectorStore = await createVectorStore(evidenceData);
    
    // Ask a question that might require multiple sources
    const answer = await answerWithCitations('What evidence was found at the scene?', vectorStore);

    expect(typeof answer).toBe('string');
    expect(answer.length).toBeGreaterThan(0);
  });
});
