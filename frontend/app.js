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
    predictionType: 'default',
    modelKey: '',
    batchCount: 1,
    targetBoardId: null,

    // ---- UI state ----
    activeTab: 'generate',
    models: [],
    selectedImageName: null,
    selectedImageMeta: null,
    displaySrc: null,
    imageLoading: false,
    previewProgress: 0,
    isGenerating: false,       // true while at least one job is in-flight
    isEnqueuing: false,        // true during the enqueue network request
    _previewDataUrl: null,     // latest live-preview data URL (null = no current frame)
    _currentDisplayName: null, // image_name currently shown (null when showing preview/nothing)
    _galleryFirstImage: null,  // first image_name in the current gallery (null = empty gallery)
    _currentBoardId: null,     // board_id of the currently-selected gallery (set by board-changed event)
    queueItems: [],
    queueDetails: [],
    _queueTimer: null,
    errorMessage: null,
    _errorExpanded: false,     // true when the nav-error toast is expanded to show the full message
    _errorTimer: null,         // setTimeout handle for auto-hiding the toast
    errorLog: [],              // history of all errors: [{message, timeStr}]
    localScanPath: '',
    candidates: [],
    scanDone: false,
    // ---- HF direct-install form ----
    hfInstall: {
      repoId: '',    // raw input — may be a full URL or bare repo id
      name: '',
      base: 'sd-1',
      apiKey: '',
      quantization: '',
      pending: false,
    },
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

      // Initialise session-scoped state that can't be serialised as plain data
      this._finishedItemIds = new Set();

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

      connectPreview(({ item_id, image_b64, step, total_steps }) => {
        // If we haven't started tracking this item yet (e.g. the tab opened after
        // the job was enqueued, or the WS enqueue event was missed), start tracking now.
        if (item_id !== undefined && !this.queueItems.includes(item_id)) {
          this.queueItems = [...this.queueItems, item_id];
          this.isGenerating = true;
          this._pollItem(item_id); // safety-net timeout only
        }
        if (this.isGenerating) {
          this.previewProgress = total_steps > 0 ? step / total_steps : 0;
        }
        // Update live-preview display directly (bypass _refreshDisplay for performance)
        if (this.showPreview) {
          this._previewDataUrl = `data:image/jpeg;base64,${image_b64}`;
          this.displaySrc = this._previewDataUrl;
          this._currentDisplayName = null; // we are NOT showing a gallery image right now
        }
      });

      // Sync starred state when gallery updates it via hover action
      window.addEventListener('image-star-updated', (e) => {
        if (this.selectedImageMeta && this.selectedImageMeta.image_name === e.detail.imageName) {
          this.selectedImageMeta = { ...this.selectedImageMeta, starred: e.detail.starred };
        }
      });

      // Track items enqueued by other clients (other tabs, scripts, etc.)
      // and handle job-completion / failure events pushed from the server.
      window.addEventListener('ws-gallery-event', (e) => {
        const { event, data } = e.detail;
        if (event === 'queue_item_enqueued') {
          const id = data.item_id;
          if (!this.queueItems.includes(id)) {
            this.queueItems = [...this.queueItems, id];
            this.isGenerating = true;
            this._pollItem(id); // safety-net timeout only
          }
        } else if (event === 'queue_item_completed') {
          this._onItemCompleted(data.item_id, data.image_name);
        } else if (event === 'queue_item_failed') {
          this._onItemFailed(data.item_id, data.error);
        } else if (event === 'queue_item_cancelled') {
          this._onItemFinished(data.item_id);
        }
      });

      // Keep _currentBoardId in sync with whatever gallery has selected.
      window.addEventListener('board-changed', (e) => {
        this._currentBoardId = e.detail.boardId;
      });

      // Keep a local copy of the gallery list for the target-gallery picker.
      window.addEventListener('galleries-updated', (e) => {
        this._galleries = e.detail.galleries || [];
      });

      // Keep targetBoardId in sync with the gallery panel's selectedBoardId
      window.addEventListener('board-changed', (e) => {
        this.targetBoardId = e.detail.boardId;
      });

      // React to gallery reporting its first visible image.
      // The gallery fires this whenever its images array changes (board switch,
      // search, load, add, delete), so we always know what rule 3 should display.
      window.addEventListener('gallery-first-image-changed', (e) => {
        this._galleryFirstImage = e.detail.imageName;
        this._refreshDisplay();
      });

      // When the WebSocket reconnects after a dropout, re-reconcile queue state
      // so we pick up anything that happened while the connection was down.
      window.addEventListener('ws-reconnected', () => {
        this._reconcileQueue();
      });

      // Reconcile any pending/processing items that were in flight before this
      // page session started (e.g. server restarted, client reconnected).
      await this._reconcileQueue();
      await this.checkLoadedModel();
    },

    // ----------------------------------------------------------------
    // Queue startup reconciliation
    // ----------------------------------------------------------------

    // On page load, fetch current queue state and start tracking any items
    // that are still pending or actively processing.  This ensures the queue
    // badge, progress indicator, and completion callbacks all work correctly
    // even when the client connects (or reconnects) after jobs were enqueued.
    async _reconcileQueue() {
      try {
        const resp = await fetch('/api/v1/queue/default/');
        if (!resp.ok) return;
        const data = await resp.json();
        const active = (data.items || []).filter(
          i => i.status === 'pending' || i.status === 'processing'
        );
        if (active.length > 0) {
          const ids = active.map(i => i.item_id);
          // Merge without duplicating ids already present (e.g. from WS events
          // that arrived between page-load and this fetch completing).
          // Also skip items that already finished THIS session (race-condition
          // guard: queue_item_completed may have arrived while the fetch was in
          // flight, so the server response can contain a stale "processing" row).
          const existingSet = new Set(this.queueItems);
          const newIds = ids.filter(
            id => !existingSet.has(id) && !this._finishedItemIds?.has(id)
          );
          if (newIds.length > 0) {
            this.queueItems = [...this.queueItems, ...newIds];
            this.isGenerating = true;
            // Safety-net timeout for each — completion is normally driven by WS events.
            for (const id of newIds) this._pollItem(id);
          }
        }
      } catch (e) {}
    },

    // ----------------------------------------------------------------
    // Generation
    // ----------------------------------------------------------------

    async generate(prepend = false) {
      if (!this.modelKey) { this._setError('Please select a model first.'); return; }
      const effectiveBoardId = this.targetBoardId || this._currentBoardId;
      if (!effectiveBoardId) { this._setError('No gallery selected — please wait for the gallery to load or create one on the right panel.'); return; }
      if (this.isEnqueuing) return;

      this.isEnqueuing = true;

      // Expand {a|b|c} brace syntax into a list of prompt variants.
      const variants = this._expandBraces(this.prompt);

      // Optimistic placeholders — one per variant so the badge reacts immediately.
      const tempIds = variants.map((_, i) => `_opt_${Date.now()}_${i}`);
      this.queueItems = [...this.queueItems, ...tempIds];

      try {
        const allRealIds = [];
        for (const variant of variants) {
          const resp = await fetch('/api/v1/queue/default/enqueue_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(this._buildEnqueueBody(prepend, variant)),
          });
          if (!resp.ok) throw new Error(await resp.text());
          const data = await resp.json();
          allRealIds.push(...data.items.map(i => i.item_id));
        }
        // Replace all optimistic placeholders with the real IDs.
        // Use a Set to prevent duplicates in case the ws-gallery-event for
        // queue_item_enqueued arrived before this HTTP response (it would have
        // already added the real ID while the optimistic placeholder was still
        // present, causing a duplicate on the naive spread approach).
        const withoutTemps = this.queueItems.filter(id => !tempIds.includes(id));
        const merged = new Set(withoutTemps);
        for (const id of allRealIds) merged.add(id);
        this.queueItems = [...merged];
        this.isGenerating = true;
      } catch (err) {
        this.queueItems = this.queueItems.filter(id => !tempIds.includes(id));
        this._setError(`Generation failed: ${err.message}`);
      } finally {
        this.isEnqueuing = false;
      }
    },

    // Expand InvokeAI-style {a|b|c} brace syntax into all prompt variants.
    // Multiple brace groups create the cartesian product:
    //   "a {photo|sketch} of a {cat|dog}" → 4 variants.
    // If the prompt has no brace groups, returns [prompt].
    _expandBraces(prompt) {
      const re = /\{([^{}]*\|[^{}]*)\}/;   // matches {x|y…} (at least one pipe, no nested {})
      const match = re.exec(prompt);
      if (!match) return [prompt];
      const alternatives = match[1].split('|');
      return alternatives.flatMap(alt => {
        const replaced = prompt.slice(0, match.index) + alt + prompt.slice(match.index + match[0].length);
        return this._expandBraces(replaced);
      });
    },

    // Total number of items that will be enqueued given current prompt + batchCount.
    _totalGenCount() {
      return this._expandBraces(this.prompt).length * this.batchCount;
    },

    _buildEnqueueBody(prepend, promptOverride = null) {
      return {
        prepend,
        batch: {
          prompt: promptOverride ?? this.prompt,
          negative_prompt: this.negativePrompt,
          seed: this.seed, width: this.width, height: this.height,
          steps: this.steps, cfg: this.cfg,
          cfg_rescale_multiplier: this.cfgRescale,
          sampler: this.sampler,
          prediction_type: this.predictionType,
          model_key: this.modelKey,
          board_id: this.targetBoardId || this._currentBoardId,
          count: this.batchCount,
        },
      };
    },

    async _pollItem(itemId) {
      // DEPRECATED — completion is now driven by WS events (queue_item_completed /
      // queue_item_failed / queue_item_cancelled).  This method is kept as a
      // safety-net fallback: if a WS event is missed for any reason the item
      // will be cleared after a 5-minute timeout so isGenerating doesn't get stuck.
      const TIMEOUT_MS = 5 * 60 * 1000;
      await new Promise(r => setTimeout(r, TIMEOUT_MS));
      // If still tracked after the timeout, force-clear it.
      if (this.queueItems.includes(itemId)) {
        this._onItemFinished(itemId);
      }
    },

    // Called via WS event when a job completes successfully.
    _onItemCompleted(itemId, imageName) {
      if (this.autoSelectLatest && imageName) {
        this.selectedImageName = imageName;
        this._loadImageMeta(imageName);
      }
      this._onItemFinished(itemId);
    },

    // Called via WS event when a job fails.
    _onItemFailed(itemId, error) {
      if (error) this._setError(`Job failed: ${error}`);
      this._onItemFinished(itemId);
    },

    // Shared cleanup: remove item from tracking and, when nothing is left in
    // flight, reset generating state and refresh the display.
    _onItemFinished(itemId) {
      // Record as finished so _reconcileQueue won't re-add it if a stale server
      // response races with this completion event.
      this._finishedItemIds?.add(itemId);
      this.queueItems = this.queueItems.filter(id => id !== itemId);
      if (this.queueItems.length === 0) {
        this.isGenerating = false;
        this.previewProgress = 0;
        this._previewDataUrl = null;
        this._currentDisplayName = null;
        this._refreshDisplay();
      }
    },

    // ----------------------------------------------------------------
    // Error helpers
    // ----------------------------------------------------------------

    // Set an error message: shows the toast, auto-hides after 10 s, logs it.
    _setError(msg) {
      if (!msg) return;
      const timeStr = new Date().toLocaleTimeString();
      this.errorMessage = msg;
      this._errorExpanded = false;
      this.errorLog.unshift({ message: msg, timeStr });
      clearTimeout(this._errorTimer);
      this._errorTimer = setTimeout(() => { this.errorMessage = null; }, 10_000);
    },

    // Dismiss the toast immediately (e.g. user clicks ×).
    _dismissError() {
      clearTimeout(this._errorTimer);
      this._errorTimer = null;
      this.errorMessage = null;
      this._errorExpanded = false;
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
      if (what === 'all' || what === 'steps')   { this.steps = meta.steps; this.cfg = meta.cfg; this.cfgRescale = meta.cfg_rescale_multiplier; this.sampler = meta.sampler; this.predictionType = meta.prediction_type || 'default'; }
    },

    selectImageFromEvent(event) {
      const { imageName } = event.detail;
      this.selectedImageName = imageName || null;
      if (imageName) {
        this._loadImageMeta(imageName);
      } else {
        this.selectedImageMeta = null;
        this.imageLoading = false;
      }
      this._refreshDisplay();
    },

    // ----------------------------------------------------------------
    // Display priority logic
    // ----------------------------------------------------------------

    // Implements the canonical display priority:
    //   1. Generation in progress + showPreview + preview frame available → keep preview
    //   2. Image selected in gallery → show it
    //   3. Gallery has images → show first one
    //   4. Nothing → blank
    // Call this whenever any of the above inputs change.
    _refreshDisplay() {
      // Rule 1 — live preview takes over during generation (handled directly by
      // connectPreview for performance; we just guard against overriding it here)
      if (this.isGenerating && this.showPreview && this._previewDataUrl) return;

      // Rules 2 & 3 — show the selected image, or fall back to the first gallery image
      const imageName = this.selectedImageName || this._galleryFirstImage;
      if (imageName) {
        if (this._currentDisplayName !== imageName) {
          this._loadAndShowImage(imageName);
        }
        return;
      }

      // Rule 4 — nothing to show
      this.displaySrc = null;
      this._currentDisplayName = null;
    },

    _loadAndShowImage(imageName) {
      this._currentDisplayName = imageName;
      this.imageLoading = true;
      // Show thumbnail immediately for responsive feel
      this.displaySrc = `/api/v1/images/i/${imageName}/thumbnail`;

      const fullUrl = `/api/v1/images/i/${imageName}/full`;
      const img = new Image();
      img.onload = () => {
        if (this._currentDisplayName === imageName) {
          this.displaySrc = fullUrl;
          this.imageLoading = false;
        }
      };
      img.onerror = () => {
        if (this._currentDisplayName === imageName) this.imageLoading = false;
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
      } catch (e) { this._setError('Could not update star.'); }
    },

    // ----------------------------------------------------------------
    // Queue
    // ----------------------------------------------------------------

    async clearQueue() {
      try { await fetch('/api/v1/queue/default/clear', { method: 'PUT' }); this.queueItems = []; await this.loadQueueDetails(); }
      catch (e) { this._setError('Could not clear queue.'); }
    },

    async loadQueueDetails() {
      try {
        const resp = await fetch('/api/v1/queue/default/');
        if (!resp.ok) return;
        const data = await resp.json();
        this.queueDetails = data.items || [];
      } catch (e) {}
      // Keep loaded-model indicator in sync while on the queue tab.
      await this.checkLoadedModel();
    },

    async cancelQueueItem(itemId) {
      try {
        await fetch(`/api/v1/queue/default/i/${itemId}`, { method: 'DELETE' });
        await this.loadQueueDetails();
      } catch (e) { this._setError('Could not cancel item.'); }
    },

    async moveToFront(itemId) {
      try {
        const resp = await fetch(`/api/v1/queue/default/i/${itemId}/move_front`, { method: 'PUT' });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadQueueDetails();
      } catch (e) { this._setError(`Could not move item to front: ${e.message}`); }
    },

    async resubmitItem(itemId) {
      try {
        const resp = await fetch(`/api/v1/queue/default/i/${itemId}/resubmit?prepend=true`, { method: 'POST' });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        this.queueItems.push(data.item_id);
        this.isGenerating = true;
        this._pollItem(data.item_id); // safety-net timeout only
        await this.loadQueueDetails();
      } catch (e) { this._setError(`Could not resubmit: ${e.message}`); }
    },

    async pruneQueue() {
      try {
        await fetch('/api/v1/queue/default/prune', { method: 'PUT' });
        await this.loadQueueDetails();
        this.queueItems = this.queueItems.filter(id =>
          this.queueDetails.some(d => d.item_id === id && (d.status === 'pending' || d.status === 'processing'))
        );
      } catch (e) { this._setError('Could not prune queue.'); }
    },

    viewQueueImage(imageName) {
      this.activeTab = 'generate';
      this.$nextTick(() => {
        this.selectImageFromEvent({ detail: { imageName } });
        window.dispatchEvent(new CustomEvent('select-image', { detail: { imageName } }));
      });
    },

    // ---- Model cache management ----

    async checkLoadedModel() {
      try {
        const resp = await fetch('/api/v2/models/loaded');
        if (resp.ok) {
          const data = await resp.json();
          this.loadedModelKey = data.loaded_key ?? null;
        }
      } catch (e) {}
    },

    async clearModelCache() {
      if (this.clearingCache || this.isGenerating) return;
      this.clearingCache = true;
      try {
        const resp = await fetch('/api/v2/models/unload_cache', { method: 'POST' });
        if (resp.ok) {
          this.loadedModelKey = null;
        } else {
          const err = await resp.json().catch(() => ({}));
          this._setError(`Could not unload model: ${err.detail || resp.statusText}`);
        }
      } catch (e) {
        this._setError(`Could not unload model: ${e.message}`);
      } finally {
        this.clearingCache = false;
      }
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
      } catch (e) { this._setError(`Scan failed: ${e.message}`); }
    },

    async scanLocalDir() {
      if (!this.localScanPath) return;
      try {
        const data = await (await fetch(`/api/v2/models/scan_local_dir?path=${encodeURIComponent(this.localScanPath)}`)).json();
        this.candidates = data.candidates || [];
        this.scanDone = true;
      } catch (e) { this._setError(`Scan failed: ${e.message}`); }
    },

    isInstalled(candidate) {
      return this.models.some(m => m.source === candidate.source);
    },

    async registerCandidate(candidate) {
      try {
        const quant = candidate.quantization || null;
        const body = {
          name: candidate.name,
          base: candidate.base,
          source_type: candidate.source_type,
          ...(quant ? { quantization: quant } : {}),
        };
        const resp = await fetch(`/api/v2/models/install?${new URLSearchParams({ source: candidate.source })}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
      } catch (e) { this._setError(`Register failed: ${e.message}`); }
    },

    // Extract "owner/repo" from a bare id or full HF URL.
    _normaliseHfRepoId(raw) {
      const s = raw.trim();
      // https://huggingface.co/owner/repo  or  hf.co/owner/repo
      const m = s.match(/(?:huggingface\.co|hf\.co)\/([^/]+\/[^/?#]+)/);
      if (m) return m[1].replace(/\/$/, '');
      return s;   // assume it's already "owner/repo"
    },

    // Mirror the backend scanner heuristic so the base auto-fills correctly.
    _guessBaseFromId(id) {
      const l = id.toLowerCase();
      if (l.includes('sana'))                              return 'sana';
      if (l.includes('xl') || l.includes('sdxl'))         return 'sdxl';
      if (l.includes('2.0') || l.includes('2-') || l.includes('sd2')) return 'sd-2';
      return 'sd-1';
    },

    // Called whenever the repoId input changes — keeps name and base in sync.
    onHfRepoIdInput() {
      const id = this._normaliseHfRepoId(this.hfInstall.repoId);
      const short = id.split('/').pop() || '';
      if (!this.hfInstall.name || this.hfInstall._nameAutoFilled) {
        this.hfInstall.name = short;
        this.hfInstall._nameAutoFilled = true;
      }
      this.hfInstall.base = this._guessBaseFromId(id);
    },

    async installFromHf() {
      const repoId = this._normaliseHfRepoId(this.hfInstall.repoId);
      if (!repoId || !repoId.includes('/')) {
        this._setError('Enter a valid HuggingFace repo ID (owner/repo) or URL.');
        return;
      }
      const name = this.hfInstall.name.trim() || repoId.split('/').pop();
      this.hfInstall.pending = true;
      try {
        const body = {
          name,
          base: this.hfInstall.base,
          source_type: 'hf_repo_id',
          ...(this.hfInstall.apiKey.trim()    ? { hf_api_key:    this.hfInstall.apiKey.trim() }    : {}),
          ...(this.hfInstall.quantization     ? { quantization:  this.hfInstall.quantization }      : {}),
        };
        const resp = await fetch(`/api/v2/models/install?${new URLSearchParams({ source: repoId })}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
        // Reset form on success
        this.hfInstall = { repoId: '', name: '', base: 'sd-1', apiKey: '', quantization: '', pending: false };
      } catch (e) {
        this._setError(`Install failed: ${e.message}`);
      } finally {
        this.hfInstall.pending = false;
      }
    },

    async deleteModel(key) {
      if (!confirm('Delete this model?')) return;
      try {
        await fetch(`/api/v2/models/${key}`, { method: 'DELETE' });
        await this.loadModels();
        if (this.modelKey === key) this.modelKey = this.models[0]?.key || '';
      } catch (e) { this._setError('Could not delete model.'); }
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
      } catch (e) { this._setError(`Rename failed: ${e.message}`); }
    },

    async setModelQuantization(key, quantization) {
      try {
        const resp = await fetch(`/api/v2/models/${key}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          // Send empty string to explicitly clear quantization
          body: JSON.stringify({ quantization: quantization || '' }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        await this.loadModels();
      } catch (e) { this._setError(`Could not update quantization: ${e.message}`); }
    },

    // ----------------------------------------------------------------
    // Preview + aspect ratio
    // ----------------------------------------------------------------

    togglePreview() {
      this.showPreview = !this.showPreview;
      if (!this.showPreview) {
        // Preview turned off: drop any live-preview frame and apply priority logic
        this._previewDataUrl = null;
        this._currentDisplayName = null;
        this._refreshDisplay();
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
        case 'predictionType': this.predictionType = meta.prediction_type || 'default'; break;
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
            case '3': e.preventDefault(); this.activeTab = 'errors';   break;
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
