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
    activeTab: 'generate',
    models: [],
    selectedImageName: null,
    selectedImageMeta: null,
    displaySrc: null,
    imageLoading: false,
    previewProgress: 0,
    queueItems: [],
    errorMessage: null,
    localScanPath: '',
    candidates: [],
    autoSelectLatest: true,

    // ---- Preview & aspect-ratio state ----
    showPreview: true,
    showInfoOverlay: false,
    lockAspectRatio: false,
    selectedAspectRatio: 'custom',
    _aspectRatioValue: null,
    _updatingAR: false,

    async init() {
      Layout.init();
      await this.loadModels();
      this.setupKeyboardShortcuts();

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
      if (!this.modelKey) { this.errorMessage = 'Please select a model first.'; return; }
      try {
        const resp = await fetch('/api/v1/queue/default/enqueue_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this._buildEnqueueBody(prepend)),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        this.queueItems.push(...data.item_ids);
        for (const id of data.item_ids) this._pollItem(id);
      } catch (err) { this.errorMessage = `Generation failed: ${err.message}`; }
    },

    _buildEnqueueBody(prepend) {
      return {
        prepend,
        batch: {
          prompt: this.prompt, negative_prompt: this.negativePrompt,
          seed: this.seed, width: this.width, height: this.height,
          steps: this.steps, cfg: this.cfg,
          cfg_rescale_multiplier: this.cfgRescale,
          sampler: this.sampler, model_key: this.modelKey,
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
            this.$dispatch('image-completed', { imageName: data.output_image_name });
            if (this.autoSelectLatest && data.output_image_name) {
              this.selectImageFromEvent({ detail: { imageName: data.output_image_name } });
            }
            break;
          }
          if (data.status === 'failed' || data.status === 'cancelled') {
            this.queueItems = this.queueItems.filter(id => id !== itemId);
            if (data.status === 'failed') this.errorMessage = `Job failed: ${data.error}`;
            break;
          }
        } catch (e) { break; }
      }
    },

    // ----------------------------------------------------------------
    // Adopt settings
    // ----------------------------------------------------------------

    adoptSettings(what) {
      const meta = this.selectedImageMeta;
      if (!meta) return;
      if (what === 'all' || what === 'prompt')  { this.prompt = meta.prompt || ''; this.negativePrompt = meta.negative_prompt || ''; }
      if (what === 'all' || what === 'size')    { this.width = meta.width; this.height = meta.height; }
      if (what === 'all' || what === 'model')   { this.modelKey = meta.model_key || this.modelKey; }
      if (what === 'all' || what === 'seed')    { this.seed = meta.seed; }
      if (what === 'all' || what === 'steps')   { this.steps = meta.steps; this.cfg = meta.cfg; this.cfgRescale = meta.cfg_rescale_multiplier; this.sampler = meta.sampler; }
    },

    selectImageFromEvent(event) {
      const { imageName } = event.detail;
      this.selectedImageName = imageName;
      this._loadImageMeta(imageName);
      this.imageLoading = true;
      this.displaySrc = `/api/v1/images/i/${imageName}/thumbnail`;
      const fullUrl = `/api/v1/images/i/${imageName}/full`;
      const img = new Image();
      img.onload = () => { if (this.selectedImageName === imageName) { this.displaySrc = fullUrl; this.imageLoading = false; } };
      img.onerror = () => { this.imageLoading = false; };
      img.src = fullUrl;
    },

    async _loadImageMeta(imageName) {
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) this.selectedImageMeta = await resp.json();
      } catch (e) {}
    },

    async toggleStar() {
      if (!this.selectedImageMeta) return;
      const newStarred = !this.selectedImageMeta.starred;
      try {
        const resp = await fetch(`/api/v1/images/i/${this.selectedImageName}/star?starred=${newStarred}`, { method: 'PATCH' });
        if (resp.ok) this.selectedImageMeta = await resp.json();
      } catch (e) { this.errorMessage = 'Could not update star.'; }
    },

    // ----------------------------------------------------------------
    // Queue
    // ----------------------------------------------------------------

    async clearQueue() {
      try { await fetch('/api/v1/queue/default/clear', { method: 'PUT' }); this.queueItems = []; }
      catch (e) { this.errorMessage = 'Could not clear queue.'; }
    },

    // ----------------------------------------------------------------
    // Models
    // ----------------------------------------------------------------

    async loadModels() {
      try {
        const resp = await fetch('/api/v2/models/');
        if (!resp.ok) return;
        const data = await resp.json();
        this.models = data.models || [];
        if (this.models.length > 0 && !this.modelKey) this.modelKey = this.models[0].key;
      } catch (e) {}
    },

    async scanHfCache() {
      try {
        const data = await (await fetch('/api/v2/models/scan_hf_cache')).json();
        this.candidates = data.candidates || [];
        if (!this.candidates.length) this.errorMessage = 'No HF cache candidates found.';
      } catch (e) { this.errorMessage = `Scan failed: ${e.message}`; }
    },

    async scanLocalDir() {
      if (!this.localScanPath) return;
      try {
        const data = await (await fetch(`/api/v2/models/scan_local_dir?path=${encodeURIComponent(this.localScanPath)}`)).json();
        this.candidates = data.candidates || [];
        if (!this.candidates.length) this.errorMessage = 'No pipeline directories found.';
      } catch (e) { this.errorMessage = `Scan failed: ${e.message}`; }
    },

    async registerCandidate(candidate) {
      try {
        const resp = await fetch(`/api/v2/models/install?${new URLSearchParams({ source: candidate.source })}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: candidate.name, base: candidate.base, source_type: candidate.source_type }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
        this.candidates = this.candidates.filter(c => c.source !== candidate.source);
      } catch (e) { this.errorMessage = `Register failed: ${e.message}`; }
    },

    async deleteModel(key) {
      if (!confirm('Delete this model?')) return;
      try {
        await fetch(`/api/v2/models/${key}`, { method: 'DELETE' });
        await this.loadModels();
        if (this.modelKey === key) this.modelKey = this.models[0]?.key || '';
      } catch (e) { this.errorMessage = 'Could not delete model.'; }
    },

    // ----------------------------------------------------------------
    // Preview + aspect ratio
    // ----------------------------------------------------------------

    togglePreview() { this.showPreview = !this.showPreview; },

    toggleInfoOverlay() { this.showInfoOverlay = !this.showInfoOverlay; },

    adoptField(field) {
      const meta = this.selectedImageMeta;
      if (!meta) return;
      switch (field) {
        case 'prompt':        this.prompt = meta.prompt || ''; break;
        case 'negPrompt':     this.negativePrompt = meta.negative_prompt || ''; break;
        case 'model':         if (meta.model_key) this.modelKey = meta.model_key; break;
        case 'seed':          this.seed = meta.seed; break;
        case 'size':          this.width = meta.width; this.height = meta.height; break;
        case 'steps':         this.steps = meta.steps; break;
        case 'cfg':           this.cfg = meta.cfg; break;
        case 'cfgRescale':    this.cfgRescale = meta.cfg_rescale_multiplier; break;
        case 'sampler':       this.sampler = meta.sampler; break;
      }
    },

    toggleLockAspectRatio() {
      this.lockAspectRatio = !this.lockAspectRatio;
      if (this.lockAspectRatio) { this._aspectRatioValue = this.width / this.height; }
      else { this.selectedAspectRatio = 'custom'; this._aspectRatioValue = null; }
    },

    onAspectRatioSelect(ratio) {
      if (ratio === 'custom') { this.lockAspectRatio = false; this._aspectRatioValue = null; return; }
      const [w, h] = ratio.split(':').map(Number);
      this._aspectRatioValue = w / h;
      this.height = this._roundTo64(this.width / this._aspectRatioValue);
      this.lockAspectRatio = true;
    },

    _roundTo64(n) { return Math.max(256, Math.round(n / 64) * 64); },

    // ----------------------------------------------------------------
    // Keyboard shortcuts
    // ----------------------------------------------------------------

    setupKeyboardShortcuts() {
      document.addEventListener('keydown', (e) => {
        const meta     = e.metaKey || e.ctrlKey;
        const isTyping = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);

        if (meta && e.key === 'Enter') { e.preventDefault(); this.generate(e.shiftKey); return; }

        if (!isTyping && !meta && !e.altKey && !e.shiftKey) {
          switch (e.key.toUpperCase()) {
            case '1': e.preventDefault(); this.activeTab = 'generate'; break;
            case '5': e.preventDefault(); this.activeTab = 'models';   break;
            case 'G': e.preventDefault(); Layout.toggleRight();        break;
            case 'T': e.preventDefault(); Layout.toggleLeft();         break;
            case 'P': e.preventDefault(); this.adoptSettings('prompt'); break;
            case 'S': e.preventDefault(); this.adoptSettings('seed');   break;
            case 'D': e.preventDefault(); this.adoptSettings('size');   break;
            case 'A': e.preventDefault(); this.adoptSettings('all');    break;
            case 'V': e.preventDefault(); this.togglePreview();         break;
            case 'I': e.preventDefault(); this.toggleInfoOverlay();      break;
          }
        }
      });
    },
  };
}



