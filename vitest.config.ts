import { defineConfig } from 'vitest/config';
import { config } from 'dotenv';

// Load environment variables from .env file
config();

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 15000, // Increase default timeout to 15 seconds for LLM API calls
  },
});
