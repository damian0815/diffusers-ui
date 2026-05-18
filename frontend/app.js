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
    displaySrc: null,       // what is shown in the image panel right now
    imageLoading: false,    // true while the full-res image is fetching
    previewProgress: 0,
    queueItems: [],
    errorMessage: null,
    localScanPath: '',
    candidates: [],

    // ---- Preview & aspect-ratio state ----
    showPreview: true,
    lockAspectRatio: false,
    selectedAspectRatio: 'custom',
    _aspectRatioValue: null,
    _updatingAR: false,

    async init() {
      await this.loadModels();
      this.setupKeyboardShortcuts();

      // Keep height in sync when width changes (and vice-versa) while AR is locked.
      // The _updatingAR flag prevents the two watchers from cascading into each other.
      this.$watch('width', (val) => {
        if (this._updatingAR || !this.lockAspectRatio || !this._aspectRatioValue) return;
        this._updatingAR = true;
        this.height = this._roundTo64(val / this._aspectRatioValue);
        this._updatingAR = false;
      });
      this.$watch('height', (val) => {
        if (this._updatingAR || !this.lockAspectRatio || !this._aspectRatioValue) return;
        this._updatingAR = true;
        this.width = this._roundTo64(val * this._aspectRatioValue);
        this._updatingAR = false;
      });

      // Preview frames write directly into displaySrc (only when preview is visible)
      connectPreview(({ image_b64, step, total_steps }) => {
        if (this.showPreview) {
          this.displaySrc = `data:image/jpeg;base64,${image_b64}`;
        }
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
      const { imageName } = event.detail;
      this.selectedImageName = imageName;
      this._loadImageMeta(imageName);

      // (a) Immediately show thumbnail magnified to fill the frame
      this.imageLoading = true;
      this.displaySrc = `/api/v1/images/i/${imageName}/thumbnail`;

      // Prefetch the full image off-screen; swap in once decoded
      const fullUrl = `/api/v1/images/i/${imageName}/full`;
      const img = new Image();
      img.onload = () => {
        // Guard: user may have clicked a different image while this was loading
        if (this.selectedImageName === imageName) {
          this.displaySrc = fullUrl;
          this.imageLoading = false;
        }
      };
      img.onerror = () => {
        this.imageLoading = false;
      };
      img.src = fullUrl;
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
    // Preview toggle
    // ----------------------------------------------------------------

    togglePreview() {
      this.showPreview = !this.showPreview;
    },

    // ----------------------------------------------------------------
    // Aspect ratio
    // ----------------------------------------------------------------

    toggleLockAspectRatio() {
      this.lockAspectRatio = !this.lockAspectRatio;
      if (this.lockAspectRatio) {
        // Capture the current ratio from the live inputs
        this._aspectRatioValue = this.width / this.height;
      } else {
        this.selectedAspectRatio = 'custom';
        this._aspectRatioValue = null;
      }
    },

    onAspectRatioSelect(ratio) {
      if (ratio === 'custom') {
        this.lockAspectRatio = false;
        this._aspectRatioValue = null;
        return;
      }
      const [w, h] = ratio.split(':').map(Number);
      this._aspectRatioValue = w / h;
      // Recalculate height from current width, snapped to multiples of 64
      this.height = this._roundTo64(this.width / this._aspectRatioValue);
      this.lockAspectRatio = true;
    },

    /** Round n to the nearest multiple of 64, minimum 256. */
    _roundTo64(n) {
      return Math.max(256, Math.round(n / 64) * 64);
    },

    // ----------------------------------------------------------------
    // Keyboard shortcuts
    // ----------------------------------------------------------------

    setupKeyboardShortcuts() {
      document.addEventListener('keydown', (e) => {
        const meta = e.metaKey || e.ctrlKey;
        const isTyping = ['INPUT', 'TEXTAREA', 'SELECT'].includes(
          document.activeElement?.tagName
        );

        // ⌘↩  Generate  /  ⌘⇧↩  Front of queue
        if (meta && e.key === 'Enter') {
          e.preventDefault();
          this.generate(e.shiftKey);
          return;
        }

        // Single-letter shortcuts — disabled while an input has focus
        if (!isTyping && !meta && !e.altKey && !e.shiftKey) {
          switch (e.key.toUpperCase()) {
            case 'P': e.preventDefault(); this.adoptSettings('prompt'); break;
            case 'S': e.preventDefault(); this.adoptSettings('seed');   break;
            case 'D': e.preventDefault(); this.adoptSettings('size');   break;
            case 'A': e.preventDefault(); this.adoptSettings('all');    break;
            case 'V': e.preventDefault(); this.togglePreview();         break;
          }
        }
      });
    },
  };
}




