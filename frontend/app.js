/* frontend/app.js — Control panel Alpine.js component */

function app() {
  return {
    // ---- Form state ----
    prompt: '',
    negativePrompt: '',
    seed: -1,
    width: 512,
    height: 512,
    steps: 20,
    cfg: 7.5,
    cfgRescale: 0.0,
    sampler: 'dpmpp_2m',
    modelKey: '',

    // ---- UI state ----
    models: [],
    selectedImageName: null,
    selectedImageMeta: null,
    previewSrc: null,
    previewProgress: 0,
    queueItems: [],
    errorMessage: null,
    localScanPath: '',
    candidates: [],

    get displaySrc() {
      // Show preview during generation, otherwise show selected image
      if (this.previewProgress > 0 && this.previewProgress < 1) return this.previewSrc;
      if (this.selectedImageName) return `/api/v1/images/i/${this.selectedImageName}/full`;
      return this.previewSrc;
    },

    async init() {
      await this.loadModels();
      this.setupKeyboardShortcuts();
      connectPreview(({ image_b64, step, total_steps }) => {
        this.previewSrc = `data:image/jpeg;base64,${image_b64}`;
        this.previewProgress = total_steps > 0 ? step / total_steps : 0;
      });
    },

    // ----------------------------------------------------------------
    // Generation
    // ----------------------------------------------------------------

    async generate(prepend = false) {
      if (!this.modelKey) {
        this.errorMessage = 'Please select a model first.';
        return;
      }
      try {
        const body = this._buildEnqueueBody(prepend);
        const resp = await fetch('/api/v1/queue/default/enqueue_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        this.queueItems.push(...data.item_ids);
        // Poll for completion
        for (const id of data.item_ids) {
          this._pollItem(id);
        }
      } catch (err) {
        this.errorMessage = `Generation failed: ${err.message}`;
      }
    },

    _buildEnqueueBody(prepend) {
      return {
        prepend,
        batch: {
          prompt: this.prompt,
          negative_prompt: this.negativePrompt,
          seed: this.seed,
          width: this.width,
          height: this.height,
          steps: this.steps,
          cfg: this.cfg,
          cfg_rescale_multiplier: this.cfgRescale,
          sampler: this.sampler,
          model_key: this.modelKey,
        },
      };
    },

    async _pollItem(itemId) {
      const POLL_MS = 1000;
      while (true) {
        await new Promise(r => setTimeout(r, POLL_MS));
        try {
          const resp = await fetch(`/api/v1/queue/default/i/${itemId}`);
          if (!resp.ok) break;
          const data = await resp.json();
          if (data.status === 'completed') {
            this.previewProgress = 0;
            this.queueItems = this.queueItems.filter(id => id !== itemId);
            // Dispatch event so gallery reloads
            this.$dispatch('image-completed', { imageName: data.output_image_name });
            break;
          }
          if (data.status === 'failed' || data.status === 'cancelled') {
            this.queueItems = this.queueItems.filter(id => id !== itemId);
            if (data.status === 'failed') this.errorMessage = `Job failed: ${data.error}`;
            break;
          }
        } catch (e) {
          break;
        }
      }
    },

    // ----------------------------------------------------------------
    // Adopt settings from gallery selection
    // ----------------------------------------------------------------

    adoptSettings(what) {
      const meta = this.selectedImageMeta;
      if (!meta) return;
      if (what === 'all' || what === 'prompt') {
        this.prompt = meta.prompt || '';
        this.negativePrompt = meta.negative_prompt || '';
      }
      if (what === 'all' || what === 'size') {
        this.width = meta.width;
        this.height = meta.height;
      }
      if (what === 'all' || what === 'model') {
        this.modelKey = meta.model_key || this.modelKey;
      }
      if (what === 'all' || what === 'seed') {
        this.seed = meta.seed;
      }
      if (what === 'all' || what === 'steps') {
        this.steps = meta.steps;
        this.cfg = meta.cfg;
        this.cfgRescale = meta.cfg_rescale_multiplier;
        this.sampler = meta.sampler;
      }
    },

    selectImageFromEvent(event) {
      this.selectedImageName = event.detail.imageName;
      this._loadImageMeta(event.detail.imageName);
    },

    async _loadImageMeta(imageName) {
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) this.selectedImageMeta = await resp.json();
      } catch (e) {
        // non-fatal
      }
    },

    async toggleStar() {
      if (!this.selectedImageMeta) return;
      const newStarred = !this.selectedImageMeta.starred;
      try {
        const resp = await fetch(
          `/api/v1/images/i/${this.selectedImageName}/star?starred=${newStarred}`,
          { method: 'PATCH' }
        );
        if (resp.ok) this.selectedImageMeta = await resp.json();
      } catch (e) {
        this.errorMessage = 'Could not update star.';
      }
    },

    // ----------------------------------------------------------------
    // Queue management
    // ----------------------------------------------------------------

    async clearQueue() {
      try {
        await fetch('/api/v1/queue/default/clear', { method: 'PUT' });
        this.queueItems = [];
      } catch (e) {
        this.errorMessage = 'Could not clear queue.';
      }
    },

    // ----------------------------------------------------------------
    // Models management
    // ----------------------------------------------------------------

    async loadModels() {
      try {
        const resp = await fetch('/api/v2/models/');
        if (!resp.ok) return;
        const data = await resp.json();
        this.models = data.models || [];
        if (this.models.length > 0 && !this.modelKey) {
          this.modelKey = this.models[0].key;
        }
      } catch (e) {
        // non-fatal at init
      }
    },

    async scanHfCache() {
      try {
        const resp = await fetch('/api/v2/models/scan_hf_cache');
        const data = await resp.json();
        this.candidates = data.candidates || [];
        if (this.candidates.length === 0) this.errorMessage = 'No HF cache candidates found.';
      } catch (e) {
        this.errorMessage = `Scan failed: ${e.message}`;
      }
    },

    async scanLocalDir() {
      if (!this.localScanPath) return;
      try {
        const resp = await fetch(
          `/api/v2/models/scan_local_dir?path=${encodeURIComponent(this.localScanPath)}`
        );
        const data = await resp.json();
        this.candidates = data.candidates || [];
        if (this.candidates.length === 0) this.errorMessage = 'No pipeline directories found.';
      } catch (e) {
        this.errorMessage = `Scan failed: ${e.message}`;
      }
    },

    async registerCandidate(candidate) {
      try {
        const params = new URLSearchParams({ source: candidate.source });
        const resp = await fetch(`/api/v2/models/install?${params}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: candidate.name,
            base: candidate.base,
            source_type: candidate.source_type,
          }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
        this.candidates = this.candidates.filter(c => c.source !== candidate.source);
      } catch (e) {
        this.errorMessage = `Register failed: ${e.message}`;
      }
    },

    async deleteModel(key) {
      if (!confirm('Delete this model from the registry?')) return;
      try {
        await fetch(`/api/v2/models/${key}`, { method: 'DELETE' });
        await this.loadModels();
        if (this.modelKey === key) this.modelKey = this.models[0]?.key || '';
      } catch (e) {
        this.errorMessage = 'Could not delete model.';
      }
    },

    // ----------------------------------------------------------------
    // Keyboard shortcuts
    // ----------------------------------------------------------------

    setupKeyboardShortcuts() {
      const ADOPT_KEYS = ['1', '2', '3', '4', '5', '6'];
      const ADOPT_WHAT = ['prompt', 'size', 'model', 'seed', 'steps', 'all'];
      document.addEventListener('keydown', (e) => {
        const meta = e.metaKey || e.ctrlKey;
        if (meta && e.key === 'Enter') {
          e.preventDefault();
          this.generate(e.shiftKey); // shift = prepend (priority)
        }
        if (meta && ADOPT_KEYS.includes(e.key)) {
          e.preventDefault();
          this.adoptSettings(ADOPT_WHAT[ADOPT_KEYS.indexOf(e.key)]);
        }
      });
    },
  };
}

