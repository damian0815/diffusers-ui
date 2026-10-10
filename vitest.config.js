import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['frontend/test-setup.js'],
    include: ['frontend/**/*.test.js'],
  },
});
