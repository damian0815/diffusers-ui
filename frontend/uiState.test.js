import { describe, it, expect, beforeEach } from 'vitest';
import './persist.js';
import './app.js';
import './gallery.js';

const APP_STATE_KEY = 'diffusers-ui:app-state';
const GALLERY_STATE_KEY = 'diffusers-ui:gallery-state';

beforeEach(() => {
  localStorage.clear();
});

describe('app _restoreUiState', () => {
  it('applies scalars and restores composite state', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({
      prompt: 'a cat', negativePrompt: 'blur', seed: 42, width: 1024, height: 768,
      steps: 30, cfg: 8, cfgRescale: 0.3, sampler: 'euler', predictionType: 'epsilon',
      batchCount: 3, autoSelectLatest: false, showPreview: false, showInfoOverlay: true,
      lockAspectRatio: true, selectedAspectRatio: '4:3', sourceStrength: 0.5,
      activeTab: 'models', localScanPath: '/tmp/models', modelKey: 'm1', targetBoardId: 'b1',
      activeLoras: [{ filename: 'x.safetensors', strength: 1.0 }],
      sourceImage: { name: 'src.png', url: '/u', width: 512, height: 512 },
      cropRect: { left: 10, top: 20, width: 200, height: 300 },
      hfInstall: { repoId: 'o/r', name: 'My', base: 'sana', quantization: 'int8', apiKey: 'SECRET' },
    }));

    const c = window.app();
    c._restoreUiState();

    expect(c.prompt).toBe('a cat');
    expect(c.negativePrompt).toBe('blur');
    expect(c.seed).toBe(42);
    expect(c.width).toBe(1024);
    expect(c.height).toBe(768);
    expect(c.steps).toBe(30);
    expect(c.cfg).toBe(8);
    expect(c.cfgRescale).toBe(0.3);
    expect(c.sampler).toBe('euler');
    expect(c.predictionType).toBe('epsilon');
    expect(c.batchCount).toBe(3);
    expect(c.autoSelectLatest).toBe(false);
    expect(c.showPreview).toBe(false);
    expect(c.showInfoOverlay).toBe(true);
    expect(c.lockAspectRatio).toBe(true);
    expect(c.sourceStrength).toBe(0.5);
    expect(c.activeTab).toBe('models');
    expect(c.localScanPath).toBe('/tmp/models');
    expect(c.modelKey).toBe('m1');
    expect(c.targetBoardId).toBe('b1');

    expect(c.activeLoras).toEqual([{ filename: 'x.safetensors', strength: 1.0 }]);
    expect(c.sourceImage).toEqual({ name: 'src.png', url: '/u', width: 512, height: 512 });
    expect(c.cropRect).toEqual({ left: 10, top: 20, width: 200, height: 300 });

    expect(c.hfInstall.repoId).toBe('o/r');
    expect(c.hfInstall.name).toBe('My');
    expect(c.hfInstall.base).toBe('sana');
    expect(c.hfInstall.quantization).toBe('int8');
    expect(c.hfInstall.apiKey).toBe('');
    expect(c.hfInstall._nameAutoFilled).toBe(false);
  });

  it('clamps batchCount to at least 1', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({ batchCount: 0, width: 512, height: 512 }));
    const c = window.app();
    c._restoreUiState();
    expect(c.batchCount).toBe(1);
  });

  it('clamps cropRect to source bounds', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({
      sourceImage: { name: 's', url: '/u', width: 100, height: 80 },
      cropRect: { left: 500, top: 500, width: 300, height: 50 },
    }));
    const c = window.app();
    c._restoreUiState();
    expect(c.sourceImage).toEqual({ name: 's', url: '/u', width: 100, height: 80 });
    expect(c.cropRect.left).toBe(36);
    expect(c.cropRect.top).toBe(16);
    expect(c.cropRect.width).toBe(100);
    expect(c.cropRect.height).toBe(64);
  });

  it('no-ops on corrupt or foreign saved state', () => {
    localStorage.setItem(APP_STATE_KEY, 'not json');
    const c = window.app();
    expect(() => c._restoreUiState()).not.toThrow();
    expect(c.prompt).toBe('');
    expect(c.batchCount).toBe(1);
  });
});

describe('app _persistUiState', () => {
  it('round-trips through localStorage without persisting the HF API key', () => {
    const c = window.app();
    c.prompt = 'hello world';
    c.seed = 7;
    c.modelKey = 'm1';
    c.hfInstall.repoId = 'o/r';
    c.hfInstall.name = 'Repo';
    c.hfInstall.base = 'sdxl';
    c.hfInstall.quantization = 'int8';
    c.hfInstall.apiKey = 'SECRET';
    c._persistUiState(true);

    const raw = JSON.parse(localStorage.getItem(APP_STATE_KEY));
    expect(raw.prompt).toBe('hello world');
    expect(raw.seed).toBe(7);
    expect(raw.modelKey).toBe('m1');
    expect(raw.hfInstall).toEqual({ repoId: 'o/r', name: 'Repo', base: 'sdxl', quantization: 'int8', vaeTiling: false, useCompel: true });
    expect(raw.hfInstall.apiKey).toBeUndefined();

    const c2 = window.app();
    c2._restoreUiState();
    expect(c2.prompt).toBe('hello world');
    expect(c2.seed).toBe(7);
    expect(c2.modelKey).toBe('m1');
    expect(c2.hfInstall.apiKey).toBe('');
  });
});

describe('app _validateRestoredUiState', () => {
  it('falls back to the first model when the restored modelKey is stale', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({ modelKey: 'ghost', width: 512, height: 512 }));
    const c = window.app();
    c.models = [{ key: 'm1', base: 'sd-1', name: 'M1' }, { key: 'm2', base: 'sd-1', name: 'M2' }];
    c._restoreUiState();
    c._validateRestoredUiState();
    expect(c.modelKey).toBe('m1');
  });

  it('keeps a valid restored modelKey', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({ modelKey: 'm2', width: 512, height: 512 }));
    const c = window.app();
    c.models = [{ key: 'm1', base: 'sd-1', name: 'M1' }, { key: 'm2', base: 'sd-1', name: 'M2' }];
    c._restoreUiState();
    c._validateRestoredUiState();
    expect(c.modelKey).toBe('m2');
  });

  it('drops restored LoRAs whose filename no longer exists', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({
      activeLoras: [
        { filename: 'keep.safetensors', strength: 1.0 },
        { filename: 'gone.safetensors', strength: 0.5 },
      ],
    }));
    const c = window.app();
    c.loraEntries = [{ filename: 'keep.safetensors', name: 'Keep' }];
    c._restoreUiState();
    c._validateRestoredUiState();
    expect(c.activeLoras).toEqual([{ filename: 'keep.safetensors', strength: 1.0 }]);
  });

  it('snaps resolution to the restored model step (sana → 32)', () => {
    localStorage.setItem(APP_STATE_KEY, JSON.stringify({ modelKey: 'sana1', width: 500, height: 600 }));
    const c = window.app();
    c.models = [{ key: 'sana1', base: 'sana', name: 'S' }];
    c._restoreUiState();
    c._validateRestoredUiState();
    expect(c.resolutionStep).toBe(32);
    expect(c.width).toBe(512);
    expect(c.height).toBe(608);
  });
});

describe('app _validateTargetBoard', () => {
  it('clears a stale targetBoardId', () => {
    const c = window.app();
    c.targetBoardId = 'ghost';
    c._galleries = [{ board_id: 'b1', board_name: 'B1' }];
    c._validateTargetBoard();
    expect(c.targetBoardId).toBeNull();
  });

  it('keeps a valid targetBoardId', () => {
    const c = window.app();
    c.targetBoardId = 'b1';
    c._galleries = [{ board_id: 'b1', board_name: 'B1' }];
    c._validateTargetBoard();
    expect(c.targetBoardId).toBe('b1');
  });
});

describe('gallery state', () => {
  it('round-trips _persistGalleryUiState / _restoreGalleryUiState', () => {
    const g = window.gallery();
    g.selectedBoardId = 'b2';
    g.search = 'cat';
    g.sortBy = 'created_at';
    g.sortDir = 'asc';
    g.starredFirst = true;
    g.thumbsPerRow = 5;
    g._persistGalleryUiState(true);

    const g2 = window.gallery();
    g2._restoreGalleryUiState();
    expect(g2.selectedBoardId).toBe('b2');
    expect(g2.search).toBe('cat');
    expect(g2.sortBy).toBe('created_at');
    expect(g2.sortDir).toBe('asc');
    expect(g2.starredFirst).toBe(true);
    expect(g2.thumbsPerRow).toBe(5);
  });

  it('ignores an out-of-range thumbsPerRow and non-string fields', () => {
    localStorage.setItem(GALLERY_STATE_KEY, JSON.stringify({
      selectedBoardId: 42, search: 'ok', sortBy: 'created_at', sortDir: 'asc',
      starredFirst: 1, thumbsPerRow: 99,
    }));
    const g = window.gallery();
    g._restoreGalleryUiState();
    expect(g.selectedBoardId).toBeNull();
    expect(g.search).toBe('ok');
    expect(g.sortBy).toBe('created_at');
    expect(g.sortDir).toBe('asc');
    expect(g.starredFirst).toBe(false);
    expect(g.thumbsPerRow).toBe(3);
  });
});
