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
    isGenerating: false,     // true while at least one job is being polled
    isEnqueuing: false,      // true during the enqueue network request
    _activePolls: 0,         // count of in-flight _pollItem loops
    _isPreviewFrame: false,  // displaySrc is currently a live preview frame
    queueItems: [],
    queueDetails: [],
    _queueTimer: null,
    errorMessage: null,
    localScanPath: '',
    candidates: [],
    scanDone: false,
    modelEditingKey: null,
    modelEditName: '',
    autoSelectLatest: true,

    // ---- Preview & aspect-ratio state ----
    showPreview: true,
    showInfoOverlay: false,
    lockAspectRatio: false,
    selectedAspectRatio: 'custom',
    _aspectRatioValue: null,
    _updatingAR: false,

    // Known aspect ratios for fuzzy matching
    _KNOWN_AR: [
      { v: '1:1',  rw: 1,  rh: 1  },
      { v: '5:4',  rw: 5,  rh: 4  },
      { v: '4:3',  rw: 4,  rh: 3  },
      { v: '3:2',  rw: 3,  rh: 2  },
      { v: '16:9', rw: 16, rh: 9  },
      { v: '4:5',  rw: 4,  rh: 5  },
      { v: '3:4',  rw: 3,  rh: 4  },
      { v: '2:3',  rw: 2,  rh: 3  },
      { v: '9:16', rw: 9,  rh: 16 },
    ],

    async init() {
      Layout.init();
      await this.loadModels();
      this.setupKeyboardShortcuts();

      // Auto-refresh queue details when on the queue tab
      this.$watch('activeTab', (tab) => {
        clearInterval(this._queueTimer);
        this._queueTimer = null;
        if (tab === 'queue') {
          this.loadQueueDetails();
          this._queueTimer = setInterval(() => {
            if (this.activeTab === 'queue') this.loadQueueDetails();
          }, 2000);
        }
      });

      this.$watch('width', (val) => {
        if (this._updatingAR) return;
        if (this.lockAspectRatio && this._aspectRatioValue) {
          this._updatingAR = true;
          this.height = this._roundTo64(val / this._aspectRatioValue);
          this._updatingAR = false;
        }
        this.selectedAspectRatio = this._detectAspectRatio(this.width, this.height);
      });
      this.$watch('height', (val) => {
        if (this._updatingAR) return;
        if (this.lockAspectRatio && this._aspectRatioValue) {
          this._updatingAR = true;
          this.width = this._roundTo64(val * this._aspectRatioValue);
          this._updatingAR = false;
        }
        this.selectedAspectRatio = this._detectAspectRatio(this.width, this.height);
      });

      // Keep the browser tab title in sync with pending job count.
      this.$watch('queueItems', (items) => {
        document.title = items.length > 0 ? `(${items.length}) diffusers-ui` : 'diffusers-ui';
      });

      connectPreview(({ image_b64, step, total_steps }) => {
        if (this.showPreview) {
          this.displaySrc = `data:image/jpeg;base64,${image_b64}`;
          this._isPreviewFrame = true;
        }
        // Only update progress while actively generating; ignore stale frames
        // that arrive after job completion.
        if (this.isGenerating) {
          this.previewProgress = total_steps > 0 ? step / total_steps : 0;
        }
      });

      // Sync starred state when gallery updates it via hover action
      window.addEventListener('image-star-updated', (e) => {
        if (this.selectedImageMeta && this.selectedImageMeta.image_name === e.detail.imageName) {
          this.selectedImageMeta = { ...this.selectedImageMeta, starred: e.detail.starred };
        }
      });
    },

    // ----------------------------------------------------------------
    // Generation
    // ----------------------------------------------------------------

    async generate(prepend = false) {
      if (!this.modelKey) { this.errorMessage = 'Please select a model first.'; return; }
      if (this.isEnqueuing) return;

      this.isEnqueuing = true;

      // ── Optimistic update: bump the counter immediately so the badge and
      //    title react before the server responds.
      const tempId = `_opt_${Date.now()}`;
      this.queueItems = [...this.queueItems, tempId];

      try {
        const resp = await fetch('/api/v1/queue/default/enqueue_batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this._buildEnqueueBody(prepend)),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        // Replace the optimistic placeholder with the real item IDs.
        const realIds = data.items.map(i => i.item_id);
        this.queueItems = [
          ...this.queueItems.filter(id => id !== tempId),
          ...realIds,
        ];
        for (const id of realIds) this._pollItem(id);
      } catch (err) {
        this.queueItems = this.queueItems.filter(id => id !== tempId);
        this.errorMessage = `Generation failed: ${err.message}`;
      } finally {
        this.isEnqueuing = false;
      }
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
      this._activePolls++;
      this.isGenerating = true;

      const POLL_MS = 1000;
      while (true) {
        await new Promise(r => setTimeout(r, POLL_MS));
        try {
          const resp = await fetch(`/api/v1/queue/default/i/${itemId}`);
          if (!resp.ok) break;
          const data = await resp.json();
          if (data.status === 'completed') {
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

      this._activePolls = Math.max(0, this._activePolls - 1);
      if (this._activePolls === 0) {
        this.isGenerating = false;
        this.previewProgress = 0;
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
      const { imageName, fromGallery = false } = event.detail;

      // Null imageName → clear the panel (e.g. when a deletion removes the selected image)
      if (!imageName) {
        this.selectedImageName = null;
        this.selectedImageMeta = null;
        if (!this._isPreviewFrame) this.displaySrc = null;
        this.imageLoading = false;
        return;
      }

      this.selectedImageName = imageName;
      this._loadImageMeta(imageName);
      this.imageLoading = true;

      const fullUrl = `/api/v1/images/i/${imageName}/full`;

      // For gallery clicks (or when there's no current image), show the
      // thumbnail immediately for responsive feel.
      // For auto-select on generation completion, keep the live preview frame
      // visible — it's already the best approximation of the result — and
      // atomically swap to the full image once it's ready.
      if (fromGallery || !this.displaySrc) {
        this.displaySrc = `/api/v1/images/i/${imageName}/thumbnail`;
        this._isPreviewFrame = false;
      }

      const img = new Image();
      img.onload = () => {
        if (this.selectedImageName === imageName) {
          this.displaySrc = fullUrl;
          this.imageLoading = false;
          this._isPreviewFrame = false;
        }
      };
      img.onerror = () => {
        if (!this.displaySrc || this._isPreviewFrame) {
          this.displaySrc = `/api/v1/images/i/${imageName}/thumbnail`;
          this._isPreviewFrame = false;
        }
        this.imageLoading = false;
      };
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
      try { await fetch('/api/v1/queue/default/clear', { method: 'PUT' }); this.queueItems = []; await this.loadQueueDetails(); }
      catch (e) { this.errorMessage = 'Could not clear queue.'; }
    },

    async loadQueueDetails() {
      try {
        const resp = await fetch('/api/v1/queue/default/');
        if (!resp.ok) return;
        const data = await resp.json();
        this.queueDetails = data.items || [];
      } catch (e) {}
    },

    async cancelQueueItem(itemId) {
      try {
        await fetch(`/api/v1/queue/default/i/${itemId}`, { method: 'DELETE' });
        await this.loadQueueDetails();
      } catch (e) { this.errorMessage = 'Could not cancel item.'; }
    },

    async moveToFront(itemId) {
      try {
        const resp = await fetch(`/api/v1/queue/default/i/${itemId}/move_front`, { method: 'PUT' });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadQueueDetails();
      } catch (e) { this.errorMessage = `Could not move item to front: ${e.message}`; }
    },

    async resubmitItem(itemId) {
      try {
        const resp = await fetch(`/api/v1/queue/default/i/${itemId}/resubmit?prepend=true`, { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        this.queueItems.push(data.item_id);
        this._pollItem(data.item_id);
        await this.loadQueueDetails();
      } catch (e) { this.errorMessage = `Could not resubmit: ${e.message}`; }
    },

    async pruneQueue() {
      try {
        await fetch('/api/v1/queue/default/prune', { method: 'PUT' });
        await this.loadQueueDetails();
        this.queueItems = this.queueItems.filter(id =>
          this.queueDetails.some(d => d.item_id === id && (d.status === 'pending' || d.status === 'processing'))
        );
      } catch (e) { this.errorMessage = 'Could not prune queue.'; }
    },

    viewQueueImage(imageName) {
      this.activeTab = 'generate';
      this.$nextTick(() => {
        this.selectImageFromEvent({ detail: { imageName } });
        window.dispatchEvent(new CustomEvent('select-image', { detail: { imageName } }));
      });
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
        this.scanDone = true;
      } catch (e) { this.errorMessage = `Scan failed: ${e.message}`; }
    },

    async scanLocalDir() {
      if (!this.localScanPath) return;
      try {
        const data = await (await fetch(`/api/v2/models/scan_local_dir?path=${encodeURIComponent(this.localScanPath)}`)).json();
        this.candidates = data.candidates || [];
        this.scanDone = true;
      } catch (e) { this.errorMessage = `Scan failed: ${e.message}`; }
    },

    isInstalled(candidate) {
      return this.models.some(m => m.source === candidate.source);
    },

    async registerCandidate(candidate) {
      try {
        const resp = await fetch(`/api/v2/models/install?${new URLSearchParams({ source: candidate.source })}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: candidate.name, base: candidate.base, source_type: candidate.source_type }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
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

    startEdit(m) {
      this.modelEditingKey = m.key;
      this.modelEditName = m.name;
      this.$nextTick(() => {
        document.getElementById(`name-input-${m.key}`)?.focus();
      });
    },

    async commitRename(key) {
      const name = this.modelEditName.trim();
      this.modelEditingKey = null;
      if (!name) return;
      const current = this.models.find(m => m.key === key);
      if (current && current.name === name) return;
      try {
        const resp = await fetch(`/api/v2/models/${key}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
      } catch (e) { this.errorMessage = `Rename failed: ${e.message}`; }
    },

    // ----------------------------------------------------------------
    // Preview + aspect ratio
    // ----------------------------------------------------------------

    togglePreview() {
      this.showPreview = !this.showPreview;
      if (!this.showPreview) {
        // Preview turned off: show the selected gallery image or clear the panel.
        this._isPreviewFrame = false;
        if (this.selectedImageName) {
          this.selectImageFromEvent({ detail: { imageName: this.selectedImageName, fromGallery: true } });
        } else {
          this.displaySrc = null;
        }
      }
    },

    toggleInfoOverlay() { this.showInfoOverlay = !this.showInfoOverlay; },

    swapDimensions() {
      [this.width, this.height] = [this.height, this.width];
    },

    _detectAspectRatio(w, h) {
      const TOL = 64;
      for (const { v, rw, rh } of this._KNOWN_AR) {
        if (Math.abs(h - Math.round(w * rh / rw)) <= TOL) return v;
        if (Math.abs(w - Math.round(h * rw / rh)) <= TOL) return v;
      }
      return 'custom';
    },

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
            case '2': e.preventDefault(); this.activeTab = 'queue';    break;
            case '5': e.preventDefault(); this.activeTab = 'models';   break;
            case 'G': e.preventDefault(); Layout.toggleRight();        break;
            case 'T': e.preventDefault(); Layout.toggleLeft();         break;
            case 'P': e.preventDefault(); this.adoptSettings('prompt'); break;
            case 'S': e.preventDefault(); this.adoptSettings('seed');   break;
            case 'D': e.preventDefault(); this.adoptSettings('size');   break;
            case 'A': e.preventDefault(); this.adoptSettings('all');    break;
            case 'V': e.preventDefault(); this.togglePreview();         break;
            case 'I': e.preventDefault(); this.toggleInfoOverlay();     break;
          }
        }
      });
    },
  };
}
