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
    _galleries: [],

    // ---- UI state ----
    activeTab: 'generate',
    models: [],
    selectedImageName: null,
    selectedImageMeta: null,
    displaySrc: null,
    imageLoading: false,
    previewProgress: 0,
    isGenerating: false,       // true while at least one job is in-flight
    isPaused: false,           // true when worker is paused (won't start new batches)
    maxBatchSize: 0,           // user-set cap on batch size (0 = auto)
    probeBatchSize: 0,         // current probe value from the memory advisor
    editingBatchSize: false,   // true while the inline batch-size input is shown
    editBatchVal: 0,           // working value during editing
    vramUsed: 0,               // current device memory allocated (bytes)
    vramTotal: 0,              // total device memory (bytes)
    imagesPerSec: 0,           // rolling-window avg images/sec over last 20 gens
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

    // ---- Img2img state ----
    sourceImage: null,          // null | {name, url, width, height}
    sourceStrength: 0.75,

    // ---- LoRA state ----
    loraEntries: [],            // all available LoRAs from GET /api/v1/loras/
    activeLoras: [],            // [{filename: "x.safetensors", strength: 1.0}, ...]
    _selectedLoraFile: '',      // dropdown binding for LoRA picker

    // ---- Preview & aspect-ratio state ----
    showPreview: true,
    showInfoOverlay: false,
    lockAspectRatio: false,
    selectedAspectRatio: 'custom',
    _aspectRatioValue: null,
    _settingBoth: false,
    _widthTimer: null,
    _heightTimer: null,

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

    // idk
    loadedModelKey: null,
    clearingCache: false,
    resolutionStep: 8,
    cropRect: null,          // { left, top, width, height } in source pixels; null = no crop
    _showCropModal: false,

    async init() {
      Layout.init();
      await this.loadModels();
      await this.fetchLoras();
      this._updateResolutionStep();
      this.setupKeyboardShortcuts();
      this.loadWorkerStatus();

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

      this.$watch('modelKey', () => this._updateResolutionStep());

      // Restore persisted local scan path
      try {
        const saved = localStorage.getItem('localScanPath');
        if (saved) this.localScanPath = saved;
      } catch (_) {}

      this.$watch('localScanPath', (val) => {
        try { localStorage.setItem('localScanPath', val); } catch (_) {}
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
        } else if (event === 'worker_paused') {
          this.isPaused = true;
        } else if (event === 'worker_resumed') {
          this.isPaused = false;
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

      // Img2img: receive source image from gallery context menu
      window.addEventListener('set-source-image', (e) => {
        const { name, url, width, height } = e.detail;
        this.sourceImage = { name, url, width, height };
        this.cropRect = { left: 0, top: 0, width, height };
        this.lockAspectRatio = true;
        this._aspectRatioValue = width / height;
        this._conformResolution('incoming', { width, height });
      });

      // Reconcile any pending/processing items that were in flight before this
      // page session started (e.g. server restarted, client reconnected).
      await this._reconcileQueue();
      await this.checkLoadedModel();

      // Provide model list to gallery for regenerate-with-different-model feature
      window.addEventListener('request-models-list', async () => {
        try {
          const resp = await fetch('/api/v2/models/');
          const data = await resp.json();
          window.dispatchEvent(new CustomEvent('receive-models-list', {
            detail: { models: data.models },
          }));
        } catch (e) {}
      });
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
        } else {
          // Server has no active items — clear stale frontend tracking state.
          // This handles the case where WS events were missed (reconnect, tab
          // switch) and isGenerating/queueItems got stuck from a prior session.
          if (this.queueItems.length > 0 || this.isGenerating) {
            this.queueItems = [];
            this.isGenerating = false;
            this.previewProgress = 0;
            this._previewDataUrl = null;
            this._currentDisplayName = null;
          }
        }
      } catch (e) {}
    },

    // ----------------------------------------------------------------
    // Generation
    // ----------------------------------------------------------------

    async generate(prepend = false) {
      this.snapResolution();
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
      const batch = {
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
      };
      if (this.sourceImage) {
        batch.init_image = this.sourceImage.name;
        batch.strength = this.sourceStrength;
        batch.source_width = this.sourceImage.width;
        batch.source_height = this.sourceImage.height;
      }
      if (this.sourceImage && this.cropRect) {
        const q = this._quantizedCropRect();
        if (q) {
          batch.crop_left   = q.left / this.sourceImage.width;
          batch.crop_top    = q.top / this.sourceImage.height;
          batch.crop_width  = q.width / this.sourceImage.width;
          batch.crop_height = q.height / this.sourceImage.height;
        }
      }
      if (this.activeLoras.length > 0) {
        batch.lora_keys = this.activeLoras.map(l => l.filename);
        batch.lora_strengths = this.activeLoras.map(l => l.strength);
      }
      return { prepend, batch };
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
    // Img2img
    // ----------------------------------------------------------------

    async uploadSourceImage() {
      return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = async () => {
          const file = input.files?.[0];
          if (!file) return resolve();
          try {
            const formData = new FormData();
            formData.append('file', file);
            const resp = await fetch('/api/v1/images/upload-source', {
              method: 'POST',
              body: formData,
            });
            if (!resp.ok) { this._setError('Upload failed'); return resolve(); }
            const data = await resp.json();
            this.sourceImage = {
              name: data.image_name,
              url: `/api/v1/images/i/${data.image_name}/full`,
              width: data.width,
              height: data.height,
            };
            this.cropRect = { left: 0, top: 0, width: data.width, height: data.height };
            this.lockAspectRatio = true;
            this._aspectRatioValue = data.width / data.height;
            this._conformResolution('incoming', { width: data.width, height: data.height });
          } catch (e) {
            this._setError('Upload failed');
          }
          resolve();
        };
        input.click();
      });
    },

    clearSourceImage() {
      this.sourceImage = null;
      this.cropRect = null;
      this.lockAspectRatio = false;
      this._aspectRatioValue = null;
      this.selectedAspectRatio = 'custom';
      this._setBothWidthAndHeight(512, 512);
    },

    // ----------------------------------------------------------------
    // LoRA management
    // ----------------------------------------------------------------

    async fetchLoras() {
      try {
        const resp = await fetch('/api/v1/loras/');
        if (!resp.ok) return;
        this.loraEntries = await resp.json();
      } catch (e) {}
    },

    addLora(filename) {
      if (!filename || this.activeLoras.find(l => l.filename === filename)) return;
      this.activeLoras.push({ filename, strength: 1.0 });
      this._selectedLoraFile = '';
    },

    removeLora(index) {
      this.activeLoras.splice(index, 1);
    },

    updateLoraStrength(index, value) {
      this.activeLoras[index].strength = parseFloat(value);
    },

    // ----------------------------------------------------------------
    // Crop modal
    // ----------------------------------------------------------------

    async openCropModal() {
      console.log('crop: openCropModal', { sourceImage: this.sourceImage, cropRect: this.cropRect });
      this._showCropModal = true;
      await this.$nextTick();
      const img = this.$refs?.cropImg;
      console.log('crop: after $nextTick', { img, complete: img?.complete, clientWidth: img?.clientWidth, naturalWidth: img?.naturalWidth });
      if (img && !img.complete) {
        await new Promise(resolve => { img.onload = resolve; img.onerror = resolve; });
        console.log('crop: image loaded via onload', { clientWidth: img.clientWidth, naturalWidth: img.naturalWidth });
      }
      this._refreshCropOverlay();
    },

    _refreshCropOverlay() {
      console.log('crop: _refreshCropOverlay');
      const container = this.$refs?.cropContainer;
      const svg = container?.querySelector('svg.crop-overlay');
      if (!svg || !this.cropRect || !this.sourceImage) return;

      if (this.$refs?.cropImg) {
        this.$refs.cropImg.offsetHeight; // force reflow
      }

      const scale = this._cropDisplayScale();
      console.log('crop: _refreshCropOverlay scale', scale);
      if (!scale || !isFinite(scale)) return;

      const r = this._scaledRect(this.cropRect);
      const q = this._scaledRect(this._quantizedCropRect());
      const hs = 10, half = hs / 2;

      function set(id, attrs) {
        const el = svg.querySelector(id);
        if (!el) return;
        for (const [k, v] of Object.entries(attrs)) {
          el.setAttribute(k, v);
        }
      }

      // Free rect
      if (r) {
        set('.crop-rect-free', { x: r.left, y: r.top, width: r.width, height: r.height });
      }
      // Quantized rect
      if (q) {
        set('.crop-rect-quantized', { x: q.left, y: q.top, width: q.width, height: q.height });
      }
      // Handles
      if (r) {
        const cursors = {
          nw: 'nwse-resize', ne: 'nesw-resize',
          sw: 'nesw-resize', se: 'nwse-resize',
          n: 'ns-resize', s: 'ns-resize',
          e: 'ew-resize', w: 'ew-resize',
        };
        const positions = {
          nw: { x: r.left - half, y: r.top - half },
          ne: { x: r.left + r.width - half, y: r.top - half },
          sw: { x: r.left - half, y: r.top + r.height - half },
          se: { x: r.left + r.width - half, y: r.top + r.height - half },
          n:  { x: r.left + r.width / 2 - half, y: r.top - half },
          s:  { x: r.left + r.width / 2 - half, y: r.top + r.height - half },
          e:  { x: r.left + r.width - half, y: r.top + r.height / 2 - half },
          w:  { x: r.left - half, y: r.top + r.height / 2 - half },
        };
        for (const [dir, pos] of Object.entries(positions)) {
          set(`.crop-handle[data-handle="${dir}"]`, {
            x: pos.x, y: pos.y, width: hs, height: hs, cursor: cursors[dir],
          });
        }
      }
    },

    closeCropModal() {
      this._showCropModal = false;
      this._cropDragState = null;
    },

    _quantizedCropRect() {
      if (!this.cropRect) return null;
      const s = this.resolutionStep;
      const { left, top, width, height } = this.cropRect;
      const cx = left + width / 2;
      const cy = top + height / 2;
      const qw = Math.max(64, Math.round(width / s) * s);
      const qh = Math.max(64, Math.round(height / s) * s);
      return {
        left: cx - qw / 2,
        top: cy - qh / 2,
        width: qw,
        height: qh,
      };
    },

    // Return inline style for the left-panel thumbnail so only the
    // crop‑rect portion of the source image is visible.
    // When there is no crop (full‑image) we fall back to cover/center.
    _thumbnailBgStyle() {
      if (!this.sourceImage) return {};
      const s = this.sourceImage;
      if (!this.cropRect
        || (this.cropRect.left === 0 && this.cropRect.top === 0
            && this.cropRect.width === s.width && this.cropRect.height === s.height)) {
        return {
          backgroundImage: `url(${s.url})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        };
      }
      const c = this.cropRect;
      const scale = Math.min(48 / c.width, 48 / c.height);
      return {
        backgroundImage: `url(${s.url})`,
        backgroundSize: `${s.width * scale}px ${s.height * scale}px`,
        backgroundPosition: `${-c.left * scale + (48 - c.width * scale) / 2}px ${-c.top * scale + (48 - c.height * scale) / 2}px`,
        backgroundRepeat: 'no-repeat',
      };
    },

    _cropDisplayScale() {
      const img = this.$refs?.cropImg;
      if (!img || !this.sourceImage) return 1;
      const scale = (img.clientWidth || img.naturalWidth || 0) / this.sourceImage.width;
      console.log('crop: _cropDisplayScale', { clientWidth: img.clientWidth, naturalWidth: img.naturalWidth, sourceW: this.sourceImage.width, scale });
      return scale;
    },

    _scaledRect(rect) {
      if (!rect || !this.sourceImage) { console.log('crop: _scaledRect null', { rect: !!rect, sourceImage: !!this.sourceImage }); return null; }
      const scale = this._cropDisplayScale();
      if (!scale || !isFinite(scale)) { console.log('crop: _scaledRect bad scale', { scale }); return null; }
      return {
        left: rect.left * scale,
        top: rect.top * scale,
        width: rect.width * scale,
        height: rect.height * scale,
      };
    },

    onCropMouseDown(e) {
      const handle = e.target.dataset.handle;
      const rect = this.cropRect;
      if (!rect || !this.sourceImage) return;

      const container = this.$refs.cropContainer;
      const cr = container.getBoundingClientRect();
      const scale = this._cropDisplayScale();
      const mx = (e.clientX - cr.left) / scale;
      const my = (e.clientY - cr.top) / scale;

      if (handle) {
        this._cropDragState = { type: 'resize', handle, startX: mx, startY: my, startRect: { ...rect } };
      } else {
        this._cropDragState = { type: 'drag', startX: mx, startY: my, startRect: { ...rect } };
      }

      this._cropMoveHandler = (ev) => this._onCropMouseMove(ev);
      this._cropUpHandler = () => this._onCropMouseUp();
      document.addEventListener('mousemove', this._cropMoveHandler);
      document.addEventListener('mouseup', this._cropUpHandler);
    },

    _onCropMouseMove(e) {
      const state = this._cropDragState;
      if (!state) return;
      const container = this.$refs.cropContainer;
      const cr = container.getBoundingClientRect();
      const scale = this._cropDisplayScale();
      const mx = (e.clientX - cr.left) / scale;
      const my = (e.clientY - cr.top) / scale;
      const dx = mx - state.startX;
      const dy = my - state.startY;
      const r = state.startRect;

      let newRect;
      if (state.type === 'drag') {
        newRect = {
          left: Math.max(0, Math.min(r.left + dx, this.sourceImage.width - r.width)),
          top: Math.max(0, Math.min(r.top + dy, this.sourceImage.height - r.height)),
          width: r.width,
          height: r.height,
        };
      } else {
        newRect = this._computeResizeRect(r, state.handle, dx, dy);
      }
      this.cropRect = newRect;
      this._refreshCropOverlay();
      if (state.type === 'resize') this._syncOutputToCropAspect();
    },

    _onCropMouseUp() {
      this._cropDragState = null;
      if (this._cropMoveHandler) {
        document.removeEventListener('mousemove', this._cropMoveHandler);
        document.removeEventListener('mouseup', this._cropUpHandler);
        this._cropMoveHandler = null;
        this._cropUpHandler = null;
      }
    },

    _syncOutputToCropAspect() {
      if (!this.cropRect) return;
      const aspect = this.cropRect.width / this.cropRect.height;
      const s = this.resolutionStep;
      const totalPx = this.width * this.height;
      let h = Math.round(Math.sqrt(totalPx / aspect) / s) * s;
      let w = Math.round(h * aspect / s) * s;
      this.height = Math.max(256, h);
      this.width = Math.max(256, w);
    },

    _computeResizeRect(r, handle, dx, dy) {
      let { left, top, width, height } = r;
      const minSize = 64;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

      switch (handle) {
        case 'nw': left += dx; top += dy; width -= dx; height -= dy; break;
        case 'ne': top += dy; width += dx; height -= dy; break;
        case 'sw': left += dx; width -= dx; height += dy; break;
        case 'se': width += dx; height += dy; break;
        case 'n':  top += dy; height -= dy; break;
        case 's':  height += dy; break;
        case 'w':  left += dx; width -= dx; break;
        case 'e':  width += dx; break;
      }

      if (width < minSize) {
        if (handle === 'nw' || handle === 'sw' || handle === 'w') left -= minSize - width;
        width = minSize;
      }
      if (height < minSize) {
        if (handle === 'nw' || handle === 'ne' || handle === 'n') top -= minSize - height;
        height = minSize;
      }

      const sw = this.sourceImage.width;
      const sh = this.sourceImage.height;
      left = clamp(left, 0, sw - width);
      top = clamp(top, 0, sh - height);
      width = clamp(width, minSize, sw);
      height = clamp(height, minSize, sh);

      return { left, top, width, height };
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
      if (what === 'all' || what === 'positive')  { this.prompt = meta.prompt || ''; }
      if (what === 'all' || what === 'negative')  { this.negativePrompt = meta.negative_prompt || ''; }
      if (what === 'all' || what === 'prompt')    { this.prompt = meta.prompt || ''; this.negativePrompt = meta.negative_prompt || ''; }
      if (what === 'all' || what === 'loras')     {
        const keys = meta.lora_keys || [];
        const strengths = meta.lora_strengths || [];
        this.activeLoras = keys.map((k, i) => ({ filename: k, strength: strengths[i] ?? 1.0 }));
      }
      if (what === 'all' || what === 'size')    { this._conformResolution('incoming', { width: meta.width, height: meta.height }); }
      if (what === 'all' || what === 'model')   { this.modelKey = meta.model_key || this.modelKey; }
      if (what === 'all' || what === 'seed')    { this.seed = meta.seed; }
      if (what === 'all' || what === 'steps')   { this.steps = meta.steps; this.cfg = meta.cfg; this.cfgRescale = meta.cfg_rescale_multiplier; this.sampler = meta.sampler; this.predictionType = meta.prediction_type || 'default'; }
      if (what === 'all' && meta.source_image) {
        this.sourceImage = {
          name: meta.source_image,
          url: `/api/v1/images/i/${meta.source_image}/full`,
          width: meta.source_width || 512,
          height: meta.source_height || 512,
        };
        this.sourceStrength = meta.strength ?? 0.75;
        if (meta.crop_width > 0) {
          this.cropRect = {
            left: Math.round(meta.crop_left * meta.source_width),
            top: Math.round(meta.crop_top * meta.source_height),
            width: Math.round(meta.crop_width * meta.source_width),
            height: Math.round(meta.crop_height * meta.source_height),
          };
        } else {
          this.cropRect = {
            left: 0,
            top: 0,
            width: meta.source_width || 512,
            height: meta.source_height || 512,
          };
        }
      } else if (what === 'all') {
        this.sourceImage = null;
        this.cropRect = null;
      }
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
        const [metaResp, galleryResp] = await Promise.all([
          fetch(`/api/v1/images/i/${imageName}/metadata`),
          fetch(`/api/v1/images/i/${imageName}/gallery`),
        ]);
        if (metaResp.ok) {
          const meta = await metaResp.json();
          if (galleryResp.ok) {
            const gallery = await galleryResp.json();
            meta.board_id = gallery.board_id;
            meta.board_name = gallery.board_name;
          }
          this.selectedImageMeta = meta;
        }
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

    async cancelGeneration() {
      try {
        await fetch('/api/v1/queue/default/cancel_current', { method: 'PUT' });
      } catch (e) {
        this._setError('Could not cancel generation.');
      }
    },

    async togglePause() {
      try {
        const method = this.isPaused ? 'resume' : 'pause';
        await fetch(`/api/v1/queue/default/${method}`, { method: 'PUT' });
        this.isPaused = !this.isPaused;
      } catch (e) {
        this._setError('Could not toggle pause.');
      }
    },

    async loadWorkerStatus() {
      try {
        const resp = await fetch('/api/v1/queue/default/status');
        if (!resp.ok) return;
        const data = await resp.json();
        this.isPaused = data.paused;
        this.maxBatchSize = data.max_batch_size;
        this.editBatchVal = this.maxBatchSize;
        this.probeBatchSize = data.probe_batch_size ?? 0;
        this.vramUsed = data.vram_used ?? 0;
        this.vramTotal = data.vram_total ?? 0;
        this.imagesPerSec = data.images_per_sec ?? 0;
      } catch (e) {}
    },

    async submitBatchSize() {
      this.editingBatchSize = false;
      const val = Math.max(0, parseInt(this.editBatchVal) || 0);
      if (val === this.maxBatchSize) return;
      this.maxBatchSize = val;  // optimistic update
      try {
        const resp = await fetch('/api/v1/queue/default/max_batch_size', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_batch_size: val }),
        });
        if (resp.ok) {
          const data = await resp.json();
          this.maxBatchSize = data.max_batch_size;
          this.editBatchVal = this.maxBatchSize;
        }
      } catch (e) {
        this._setError('Could not set max batch size.');
        this.loadWorkerStatus();  // revert to server truth
      }
    },

    async setMaxBatchSize(n) {
      const val = Math.max(0, parseInt(n) || 0);
      try {
        await fetch('/api/v1/queue/default/max_batch_size', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ max_batch_size: val }),
        });
        this.maxBatchSize = val;
      } catch (e) { this._setError('Could not set max batch size.'); }
    },

    async loadQueueDetails() {
      try {
        const resp = await fetch('/api/v1/queue/default/');
        if (!resp.ok) return;
        const data = await resp.json();
        this.queueDetails = data.items || [];
      } catch (e) {}
      // Keep loaded-model indicator and VRAM in sync while on the queue tab.
      await this.checkLoadedModel();
      await this.loadWorkerStatus();
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
      } catch (e) {
        console.error("failed to load model:", e)
      }
    },

    async clearModelCache() {
      if (this.clearingCache || (this.isGenerating && !this.isPaused)) return;
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
      this._setBothWidthAndHeight(this.height, this.width);
    },

    _setBothWidthAndHeight(w, h) {
      this._settingBoth = true;
      this.width = w;
      this.height = h;
      this._settingBoth = false;
      this.selectedAspectRatio = this._detectAspectRatio(w, h);
    },

    _conformResolution(source, dims) {
      const s = this.resolutionStep;
      switch (source) {
        case 'width-typed': {
          const w = Math.max(256, Math.round(dims.width / s) * s);
          if (this.lockAspectRatio && this._aspectRatioValue) {
            const h = Math.max(256, Math.round(w / this._aspectRatioValue / s) * s);
            this._setBothWidthAndHeight(w, h);
          } else {
            this._setBothWidthAndHeight(w, this.height);
          }
          break;
        }
        case 'height-typed': {
          const h = Math.max(256, Math.round(dims.height / s) * s);
          if (this.lockAspectRatio && this._aspectRatioValue) {
            const w = Math.max(256, Math.round(h * this._aspectRatioValue / s) * s);
            this._setBothWidthAndHeight(w, h);
          } else {
            this._setBothWidthAndHeight(this.width, h);
          }
          break;
        }
        case 'aspect-ratio':
        case 'incoming': {
          const w = Math.max(256, Math.round(dims.width / s) * s);
          const h = Math.max(256, Math.round(dims.height / s) * s);
          this._setBothWidthAndHeight(w, h);
          break;
        }
      }
    },

    onWidthInput() {
      clearTimeout(this._widthTimer);
      this._widthTimer = setTimeout(() => {
        this._conformResolution('width-typed', { width: this.width });
      }, 300);
    },

    onWidthStep(dir) {
      this.width += dir * this.resolutionStep;
      this._conformResolution('width-typed', { width: this.width });
    },

    onWidthBlur() {
      clearTimeout(this._widthTimer);
      this._conformResolution('width-typed', { width: this.width });
    },

    onHeightInput() {
      clearTimeout(this._heightTimer);
      this._heightTimer = setTimeout(() => {
        this._conformResolution('height-typed', { height: this.height });
      }, 300);
    },

    onHeightStep(dir) {
      this.height += dir * this.resolutionStep;
      this._conformResolution('height-typed', { height: this.height });
    },

    onHeightBlur() {
      clearTimeout(this._heightTimer);
      this._conformResolution('height-typed', { height: this.height });
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
        case 'size':          this._conformResolution('incoming', { width: meta.width, height: meta.height }); break;
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
      const [rw, rh] = ratio.split(':').map(Number);
      this._aspectRatioValue = rw / rh;
      this.lockAspectRatio = true;
      const targetHeight = Math.round(this.width / this._aspectRatioValue);
      this._conformResolution('aspect-ratio', { width: this.width, height: targetHeight });
    },

    _updateResolutionStep() {
      const m = this.models.find(m => m.key === this.modelKey);
      this.resolutionStep = m?.base === 'sana' ? 32 : 8;
    },

    snapResolution() {
      const s = this.resolutionStep;
      const w = Math.max(256, Math.round(this.width  / s) * s);
      const h = Math.max(256, Math.round(this.height / s) * s);
      if (w !== this.width || h !== this.height) {
        this._setBothWidthAndHeight(w, h);
      }
    },

    _snap(n) { return Math.max(256, Math.round(n / this.resolutionStep) * this.resolutionStep); },

    // ----------------------------------------------------------------
    // Gallery navigation
    // ----------------------------------------------------------------

    navigateToGallery(boardId, imageName) {
      if (!boardId) return;
      this.activeTab = 'generate';
      window.dispatchEvent(new CustomEvent('navigate-board', {
        detail: { boardId, imageName },
      }));
    },

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
