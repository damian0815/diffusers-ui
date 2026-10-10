import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import './persist.js';

describe('Persist', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('round-trips get/set', () => {
    const obj = { a: 1, b: [1, 2, 3], c: { d: true }, e: null };
    window.Persist.set('k', obj);
    expect(window.Persist.get('k')).toEqual(obj);
  });

  it('returns the fallback on a missing key', () => {
    expect(window.Persist.get('nope', 'fb')).toBe('fb');
    expect(window.Persist.get('nope')).toBeNull();
  });

  it('returns the fallback on corrupt JSON', () => {
    localStorage.setItem('k', '{bad');
    expect(window.Persist.get('k', { def: 1 })).toEqual({ def: 1 });
  });

  it('setDebounced writes after the debounce window', () => {
    vi.useFakeTimers();
    window.Persist.setDebounced('k', { a: 1 });
    expect(localStorage.getItem('k')).toBeNull();
    vi.advanceTimersByTime(250);
    expect(window.Persist.get('k')).toEqual({ a: 1 });
  });

  it('setDebounced coalesces rapid writes', () => {
    vi.useFakeTimers();
    window.Persist.setDebounced('k', { n: 1 });
    vi.advanceTimersByTime(100);
    window.Persist.setDebounced('k', { n: 2 });
    vi.advanceTimersByTime(100);
    expect(localStorage.getItem('k')).toBeNull();
    vi.advanceTimersByTime(150);
    expect(window.Persist.get('k')).toEqual({ n: 2 });
  });

  it('flush writes immediately and cancels any pending debounced write', () => {
    vi.useFakeTimers();
    window.Persist.setDebounced('k', { n: 1 });
    window.Persist.flush('k', { n: 2 });
    expect(window.Persist.get('k')).toEqual({ n: 2 });
    vi.advanceTimersByTime(1000);
    expect(window.Persist.get('k')).toEqual({ n: 2 });
  });

  it('swallows storage quota/security errors', () => {
    const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });
    expect(() => window.Persist.set('k', { x: 1 })).not.toThrow();
    expect(() => window.Persist.setDebounced('k', { x: 1 })).not.toThrow();
    expect(() => window.Persist.flush('k', { x: 1 })).not.toThrow();
    spy.mockRestore();
  });
});
