/* frontend/gallery.js — Gallery panel Alpine.js component */

const GALLERY_STATE_KEY = 'diffusers-ui:gallery-state';

// The seven "Match by" checkboxes of comparison mode (COMPARISON_GALLERY_MODE.md §3.4).
// `id` doubles as the canonical API field name; `model` maps to model_key server-side.
const COMPARE_FIELD_OPTIONS = [
  { id: 'prompt', label: 'Prompt' },
  { id: 'negative_prompt', label: 'Negative prompt' },
  { id: 'seed', label: 'Seed' },
  { id: 'width', label: 'Width' },
  { id: 'height', label: 'Height' },
  { id: 'cfg', label: 'CFG' },
  { id: 'model', label: 'Model' },
];

const COMPARE_DEFAULT_FIELDS = ['prompt', 'negative_prompt', 'seed', 'width', 'height', 'cfg'];

function gallery() {
  return {
    galleries: [],
    selectedBoardId: null,
    images: [],
    search: '',
    sortBy: 'request_serial_number',
    sortDir: 'desc',
    starredFirst: false,
    showOptions: false,
    boardDropdownOpen: false,
    moveBoardDropdownOpen: false,
    offset: 0,
    limit: 50,
    total: 0,
    loading: false,
    hasMore: true,
    selectedImageName: null,
    thumbsPerRow: 3,
    _dragSrcIdx: null,
    _pendingRestore: false,

    // ── Comparison mode ──────────────────────────────────────────────────
    mode: 'browse',                 // 'browse' | 'compare'
    compareModelsPersisted: false,  // set by _restoreGalleryUiState; true iff a saved selection
                                    //   exists — switches §3.2-A (first entry) vs §3.2-B (later)
    compareRef: null,               // image_name of the reference image
    compareFields: [...COMPARE_DEFAULT_FIELDS],  // 'model' unchecked by default (§2.3)
    compareModels: [],              // model_keys checked (persisted; §3.2 rules)
    compareKnownModels: [],         // model_keys seen at last compare-mode entry (persisted;
                                    //   its absence in the blob marks "first entry ever")
    compareAllModels: [],           // [{model_key, model_name, image_count, last_used, registered}]
    compareConfig: { matchOpen: true, modelsOpen: true },  // collapsible groups
    compareLoading: false,
    _compareReloadTimer: null,      // debounced WS-triggered reload handle
    compareFieldOptions: COMPARE_FIELD_OPTIONS,  // x-for can't see top-level consts

    // ── Multi-select ─────────────────────────────────────────────────────
    selectedImageNames: [],   // all highlighted thumbnails
    _lastClickIdx: null,      // anchor for shift-range
    _moving: false,

    // ── Navigation state ─────────────────────────────────────────────────
    _isNavigating: false,     // true while jumping to a specific offset/image

    // ── Context menu ─────────────────────────────────────────────────────
    ctxMenu: { open: false, x: 0, y: 0 },

    // ── Move modal ───────────────────────────────────────────────────────
    moveModal: { open: false, targetBoardId: null },

    // ── Regenerate modal ────────────────────────────────────────────────
    regenModal: {
      open: false,
      selectedModelKey: null,
      priority: false,   // checked → enqueue at rank 2 (ahead of rank-1 work)
      _enqueuing: false,
      _progress: null,   // null | { done: 0, total: 0 }
    },
    availableModels: [],

    _notifyGalleriesUpdated() {
      window.dispatchEvent(new CustomEvent('galleries-updated', {
        detail: { galleries: this.galleries },
      }));
    },

    _notifyBoardChanged() {
      window.dispatchEvent(new CustomEvent('board-changed', {
        detail: { boardId: this.selectedBoardId },
      }));
      // Also notify the app panel so its targetBoardId stays in sync
      window.dispatchEvent(new CustomEvent('board-changed', {
        detail: { boardId: this.selectedBoardId },
      }));
    },

    async init() {
      this._restoreGalleryUiState();
      await this.loadGalleries();
      this._notifyBoardChanged();
      this.$watch('selectedBoardId', (newVal, oldVal) => {
        this._notifyBoardChanged();
        this._saveGalleryState(oldVal);
      });
      this.$watch('selectedImageName', () => this._saveGalleryState(this.selectedBoardId));

      for (const f of ['selectedBoardId', 'search', 'sortBy', 'sortDir', 'starredFirst', 'thumbsPerRow',
                       'compareFields', 'compareModels', 'compareKnownModels', 'compareConfig']) {
        this.$watch(f, () => this._persistGalleryUiState());
      }
      window.addEventListener('pagehide', () => this._persistGalleryUiState(true));

      this.setupInfiniteScroll();
      this.setupArrowNav();
      window.addEventListener('image-completed', (e) => {
        this.prependImage(e.detail.imageName);
      });
      window.addEventListener('ws-gallery-event', (e) => {
        this._handleWsEvent(e.detail);
      });
      this.$watch('images', (imgs) => {
        window.dispatchEvent(new CustomEvent('gallery-first-image-changed', {
          detail: { imageName: imgs[0]?.image_name ?? null },
        }));
      });

      // Navigate to a specific gallery and select an image (from info overlay).
      // Loads pages until the target image is found, then assigns all at once
      // so partial thumbnails are never rendered mid-search.
      window.addEventListener('navigate-board', async (e) => {
        const { boardId, imageName } = e.detail;
        if (!boardId || !imageName) return;
        this.mode = 'browse';  // §3.6: board-scoped navigation always exits comparison mode
        this._isNavigating = true;
        this.selectedBoardId = boardId;
        this.$dispatch('select-image', { imageName: null });
        this.selectedImageName = null;
        this._clearSelection();
        this.images = [];
        this.offset = 0;
        this.total = 0;
        this.hasMore = true;

        const temp = [];
        let loadOff = 0;
        let found = false;
        while (!found) {
          try {
            const params = new URLSearchParams({
              board_id: boardId, offset: loadOff, limit: this.limit,
            });
            if (this.search) params.set('search', this.search);
            params.set('sort_by', this.sortBy);
            params.set('sort_dir', this.sortDir);
            if (this.starredFirst) params.set('starred_first', 'true');
            const resp = await fetch(`/api/v1/images/?${params}`);
            if (!resp.ok) break;
            const data = await resp.json();
            this.total = data.total;
            const batch = data.items;
            temp.push(...batch);

            const localIdx = batch.findIndex(i => i.image_name === imageName);
            if (localIdx !== -1) {
              found = true;
              this.images = temp;
              this.offset = loadOff + batch.length;
              this.hasMore = batch.length === this.limit;
              const globalIdx = loadOff + localIdx;
              this.selectImage(imageName, globalIdx);
              this.$nextTick(() => {
                document.querySelectorAll('.thumb-wrapper')[globalIdx]
                  ?.scrollIntoView({ block: 'nearest' });
              });
              break;
            }
            loadOff += batch.length;
            if (batch.length < this.limit) break;
          } catch (_) { break; }
        }
        if (!found) {
          if (temp.length > 0) {
            this.images = temp.slice(0, this.limit);
            this.offset = this.limit;
            this.hasMore = temp.length === this.limit;
          } else {
            this.images = [];
            this.offset = 0;
            this.hasMore = true;
            await this.loadMore();
          }
        }
        this._isNavigating = false;
      });

      // Receive models list from app component
      window.addEventListener('receive-models-list', (e) => {
        this.availableModels = e.detail.models;
      });

      // Close context menu on any outside click; Escape closes both menu + modal
      window.addEventListener('click', () => { this.ctxMenu.open = false; });
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.ctxMenu.open = false;
          if (this.regenModal.open) { this.closeRegenerateModal(); return; }
          this.moveModal.open = false;
        }
      });
    },

    async loadGalleries() {
      try {
        const resp = await fetch('/api/v1/boards/');
        if (!resp.ok) return;
        this.galleries = await resp.json();
        this._notifyGalleriesUpdated();
        if (this.galleries.length > 0) {
          if (!this.galleries.some(g => g.board_id === this.selectedBoardId)) {
            this.selectedBoardId = this.galleries[0].board_id;
          }
          this._pendingRestore = true;
          await this.loadMore();
        }
      } catch (e) {}
    },

    onBoardChange() {
      this.$dispatch('select-image', { imageName: null });
      this.selectedImageName = null;
      this._clearSelection();
      this._pendingRestore = true;
      this.resetAndLoad();
    },

    resetAndLoad() {
      this.images = []; this.offset = 0; this.total = 0; this.hasMore = true;
      this.loadMore();
    },

    async selectGallery(boardId) {
      this.selectedBoardId = boardId;
      this.$dispatch('select-image', { imageName: null });
      this.selectedImageName = null;
      this._clearSelection();
      this._pendingRestore = true;
      this.resetAndLoad();
    },

    _galleryStateKey(boardId) {
      return `gallery-state:${boardId}`;
    },

    _restoreGalleryUiState() {
      const saved = Persist.get(GALLERY_STATE_KEY, {});
      if (!saved || typeof saved !== 'object') return;
      if (typeof saved.selectedBoardId === 'string') this.selectedBoardId = saved.selectedBoardId;
      if (typeof saved.search === 'string')          this.search = saved.search;
      if (typeof saved.sortBy === 'string')          this.sortBy = saved.sortBy;
      if (typeof saved.sortDir === 'string')         this.sortDir = saved.sortDir;
      if (typeof saved.starredFirst === 'boolean')   this.starredFirst = saved.starredFirst;
      const cols = parseInt(saved.thumbsPerRow);
      if (cols >= 1 && cols <= 8) this.thumbsPerRow = cols;

      // Comparison mode (§5.1). `mode` itself is deliberately NOT persisted:
      // after reload there is no guaranteed reference image, so the panel
      // starts in browse mode; an ignore-mode rule also means a persisted
      // 'compare' value from a malformed blob is harmless.
      // A saved compare:models ARRAY — even empty — marks "not first entry ever".
      this.compareModelsPersisted = Array.isArray(saved['compare:models']);
      if (Array.isArray(saved['compare:models'])) {
        this.compareModels = saved['compare:models'].filter(k => typeof k === 'string');
      }
      if (Array.isArray(saved['compare:known'])) {
        this.compareKnownModels = saved['compare:known'].filter(k => typeof k === 'string');
      }
      if (Array.isArray(saved['compare:fields'])) {
        // Ignore names from a future/older field set so they can't poison state.
        const valid = COMPARE_FIELD_OPTIONS.map(o => o.id);
        const fields = saved['compare:fields'].filter(f => valid.includes(f));
        this.compareFields = fields.length ? fields : [...COMPARE_DEFAULT_FIELDS];
      }
      const groups = saved['compare:groups'];
      if (groups && typeof groups === 'object') {
        if (typeof groups.matchOpen === 'boolean')  this.compareConfig.matchOpen = groups.matchOpen;
        if (typeof groups.modelsOpen === 'boolean') this.compareConfig.modelsOpen = groups.modelsOpen;
      }
    },

    _persistGalleryUiState(flush = false) {
      const state = {
        selectedBoardId: this.selectedBoardId,
        search: this.search,
        sortBy: this.sortBy,
        sortDir: this.sortDir,
        starredFirst: this.starredFirst,
        thumbsPerRow: this.thumbsPerRow,
      };
      // The presence of `compare:models` in the blob is what marks "not the
      // first entry ever" (§3.2-A/B) — so the compare keys may only be written
      // once a selection actually exists (compareModelsPersisted is set by
      // _applyCompareSelection on first entry). Writing them unconditionally
      // from component defaults would make a pure-browsing session look
      // configured and flip the first entry to rule B.
      if (this.compareModelsPersisted) {
        state['compare:fields'] = this.compareFields;
        state['compare:models'] = this.compareModels;
        state['compare:known'] = this.compareKnownModels;
        state['compare:groups'] = this.compareConfig;
      }
      if (flush) Persist.flush(GALLERY_STATE_KEY, state);
      else Persist.setDebounced(GALLERY_STATE_KEY, state);
    },

    _saveGalleryState(boardId) {
      if (!boardId) return;
      try {
        const state = { selectedImageName: this.selectedImageName };
        localStorage.setItem(this._galleryStateKey(boardId), JSON.stringify(state));
      } catch (_) {}
    },

    _loadGalleryState(boardId) {
      if (!boardId) return null;
      try {
        const raw = localStorage.getItem(this._galleryStateKey(boardId));
        return raw ? JSON.parse(raw) : null;
      } catch (_) { return null; }
    },

    _restoreGalleryState() {
      const state = this._loadGalleryState(this.selectedBoardId);
      if (!state || !state.selectedImageName) return;
      const idx = this.images.findIndex(i => i.image_name === state.selectedImageName);
      if (idx !== -1) {
        this.selectImage(state.selectedImageName, idx);
        this.$nextTick(() => {
          document.querySelectorAll('.thumb-wrapper')[idx]
            ?.scrollIntoView({ block: 'nearest' });
        });
      }
    },

    _clearSelection() {
      this.selectedImageNames = [];
      this._lastClickIdx = null;
    },

    // ── Comparison mode (COMPARISON_GALLERY_MODE.md §5.1) ─────────────────

    async toggleCompareMode() {
      if (this.mode === 'browse') {
        // Reference = displayed image, else the first loaded image of the board (§2.2).
        const ref = this.selectedImageName || this.images[0]?.image_name;
        if (!ref) return;
        this.compareRef = ref;
        this.mode = 'compare';
        // Always re-fetch: counts/names change as models are used, renamed, installed.
        await this._loadComparisonModels();
        const anchorKey = await this._resolveAnchorKey(ref);
        this._applyCompareSelection(anchorKey);
        this.resetAndLoad();
        // Persist so the updated selection/known-keys land even if the user
        // touches no checkbox before exiting.
        this._persistGalleryUiState();
      } else {
        this.mode = 'browse';
        this.resetAndLoad();  // browse fetch path unchanged (§2.7)
      }
    },

    async _loadComparisonModels() {
      try {
        const resp = await fetch('/api/v1/images/comparison_models');
        if (!resp.ok) return;
        const data = await resp.json();
        this.compareAllModels = data.models || [];
      } catch (_) { /* keep the previous list; selection degrades to anchor-only */ }
    },

    async _resolveAnchorKey(imageName) {
      // model_key of the reference image; loaded grid entries carry it, but if
      // the image isn't on the current page (e.g. after navigate-board) fetch metadata.
      const local = this.images.find(i => i.image_name === imageName);
      if (local?.model_key) return local.model_key;
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) return (await resp.json()).model_key || null;
      } catch (_) {}
      return null;
    },

    _applyCompareSelection(anchorKey) {
      // §3.2. A: first entry ever → anchor ∪ last two added (registry order).
      // §3.2. B: later entries → prune(persisted) ∪ {anchor} ∪ keys that are
      // new since the persisted known-keys snapshot (auto-check freshly
      // installed models once; manual unchecks of known models survive).
      const firstEntry = !this.compareModelsPersisted;
      const registeredKeys = this.compareAllModels
        .filter(m => m.registered).map(m => m.model_key);
      const lastTwoAdded = registeredKeys.slice(-2);
      const allKeys = this.compareAllModels.map(m => m.model_key);
      const dedupe = (list) => [...new Set(list.filter(k => k))];

      if (firstEntry) {
        this.compareModels = dedupe([anchorKey, ...lastTwoAdded]);
      } else {
        const kept = this.compareModels.filter(k => allKeys.includes(k));
        const newlyAdded = allKeys.filter(k => !this.compareKnownModels.includes(k));
        this.compareModels = dedupe([...kept, anchorKey, ...newlyAdded]);
      }
      this.compareKnownModels = allKeys;
      this.compareModelsPersisted = true;
    },

    onCompareConfigChanged() {
      // Any checkbox change re-queries the grid automatically (§2.4).
      this.resetAndLoad();
    },

    toggleCompareGroup(which) {
      // Replace (not mutate) so the compareConfig $watch fires and persists.
      this.compareConfig = { ...this.compareConfig, [which]: !this.compareConfig[which] };
      this._persistGalleryUiState();
    },

    compareSelectAllModels() {
      this.compareModels = this.compareAllModels.map(m => m.model_key);
      this.onCompareConfigChanged();
    },

    compareSelectNoneModels() {
      this.compareModels = [];
      this.onCompareConfigChanged();
    },

    _debouncedCompareReload() {
      // New generation may or may not match the current hash — re-query
      // instead of inserting optimistically (§2.8).
      if (this._compareReloadTimer) clearTimeout(this._compareReloadTimer);
      this._compareReloadTimer = setTimeout(() => {
        this._compareReloadTimer = null;
        if (this.mode === 'compare') this.resetAndLoad();
      }, 800);
    },

    async loadComparePage() {
      if (this.compareLoading) return;
      this.loading = true;
      this.compareLoading = true;
      try {
        const isFirstPage = this.offset === 0;
        const params = new URLSearchParams({
          reference: this.compareRef, offset: this.offset, limit: this.limit,
        });
        for (const f of this.compareFields) params.append('fields', f);
        for (const k of this.compareModels) params.append('models', k);
        const resp = await fetch(`/api/v1/images/compare?${params}`);
        if (!resp.ok) throw new Error(resp.statusText);
        const data = await resp.json();
        const batch = data.items;
        this.total = data.total;
        this.images.push(...batch);
        this.offset += batch.length;
        this.hasMore = batch.length === this.limit;
        // Highlight the reference like a normal selection on the first page
        // (replaces per-board restore in compare mode).
        if (isFirstPage && this.compareRef) {
          const idx = this.images.findIndex(i => i.image_name === this.compareRef);
          if (idx !== -1) this.selectImage(this.compareRef, idx);
        }
      } catch (e) {
      } finally {
        this.loading = false;
        this.compareLoading = false;
      }
    },

    // ── Gallery position navigation ──────────────────────────────────────

    async jumpToOffset(offset) {
      if (this.loading || this._isNavigating) return;
      this._isNavigating = true;
      this.images = [];
      this.offset = Math.max(0, Math.min(offset, Math.max(0, this.total - 1)));
      this.hasMore = true;
      this.loading = false;
      await this.loadMore();
      document.getElementById('gallery-contents')?.scrollTo(0, 0);
      this._isNavigating = false;
    },

    jumpBy(delta) {
      const current = Math.max(0, this.offset - this.limit);
      const target = Math.max(0, Math.floor((current + delta) / this.limit) * this.limit);
      this.jumpToOffset(target);
    },

    jumpToStart() { this.jumpToOffset(0); },

    jumpToEnd() {
      const last = Math.max(0, Math.floor((this.total - 1) / this.limit) * this.limit);
      this.jumpToOffset(last);
    },

    jumpToMid() {
      const mid = Math.max(0, Math.floor(this.total / 2 / this.limit) * this.limit);
      this.jumpToOffset(mid);
    },

    async loadMore() {
      if (this.loading || !this.hasMore || this._isNavigating) return;
      // Compare mode has no board; it queries the compare endpoint instead (§5.1).
      if (this.mode === 'compare') { await this.loadComparePage(); return; }
      if (!this.selectedBoardId) return;
      this.loading = true;
      try {
        const params = new URLSearchParams({
          board_id: this.selectedBoardId, offset: this.offset, limit: this.limit,
        });
        if (this.search) params.set('search', this.search);
        params.set('sort_by', this.sortBy);
        params.set('sort_dir', this.sortDir);
        if (this.starredFirst) params.set('starred_first', 'true');
        const resp = await fetch(`/api/v1/images/?${params}`);
        if (!resp.ok) throw new Error(resp.statusText);
        const data = await resp.json();
        const batch = data.items;
        this.total = data.total;
        this.images.push(...batch);
        this.offset += batch.length;
        this.hasMore = batch.length === this.limit;
      } catch (e) {
      } finally {
        this.loading = false;
        // Per-board restore is a browse-mode concept; in compare mode the
        // reference is highlighted by loadComparePage() itself.
        if (this._pendingRestore && this.mode === 'browse') {
          this._pendingRestore = false;
          this._restoreGalleryState();
        }
      }
    },

    _insertImageInOrder(img) {
      const dir = this.sortDir === 'asc' ? 1 : -1;
      const key = this.sortBy === 'created_at' ? 'created_at' : 'request_serial_number';
      const getVal = (item) => {
        const v = item[key];
        if (v === null || v === undefined) return dir === 1 ? Infinity : -Infinity;
        return v;
      };

      const isStarredFirst = this.starredFirst;
      const compare = (a, b) => {
        if (isStarredFirst && a.starred !== b.starred) {
          return a.starred ? -1 : 1;
        }
        const va = getVal(a);
        const vb = getVal(b);
        return va < vb ? -dir : va > vb ? dir : 0;
      };

      const imgs = [...this.images];
      let lo = 0, hi = imgs.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (compare(imgs[mid], img) <= 0) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      imgs.splice(lo, 0, img);
      this.images = imgs;
    },

    async prependImage(imageName) {
      if (this.mode === 'compare') return;  // may not match the hash — WS branch reloads instead
      if (this.images.some(i => i.image_name === imageName)) return;
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) {
          const meta = await resp.json();
          if (!this.images.some(i => i.image_name === meta.image_name)) {
            this._insertImageInOrder(meta);
            this.offset += 1;
            this.total += 1;
          }
        }
      } catch (e) { this.resetAndLoad(); }
    },

    setupInfiniteScroll() {
      const sentinel = document.getElementById('gallery-sentinel');
      if (!sentinel) return;
      new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) this.loadMore();
      }, { rootMargin: '100px' }).observe(sentinel);
    },

    setupArrowNav() {
      document.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
        e.preventDefault();

        if (this.images.length === 0) return;

        const idx = this.images.findIndex(i => i.image_name === this.selectedImageName);
        if (idx === -1) {
          this.selectImage(this.images[0].image_name, 0);
          return;
        }

        const isNext = e.key === 'ArrowRight' || e.key === 'ArrowDown';
        const delta  = e.key === 'ArrowDown' || e.key === 'ArrowUp' ? this.thumbsPerRow : 1;
        const newIdx = Math.max(0, Math.min(this.images.length - 1, idx + (isNext ? delta : -delta)));

        if (newIdx !== idx) {
          this.selectImage(this.images[newIdx].image_name, newIdx);
          document.querySelectorAll('.thumb-wrapper')[newIdx]?.scrollIntoView({ block: 'nearest' });
        }
        if (newIdx >= this.images.length - 10) this.loadMore();
      });
    },

    // Plain single-select (also used by arrow nav)
    selectImage(imageName, idx = null) {
      this.selectedImageName = imageName;
      this.selectedImageNames = imageName ? [imageName] : [];
      if (idx !== null) this._lastClickIdx = idx;
      this.$dispatch('select-image', { imageName, fromGallery: true });
    },

    // Click handler for thumbnails — handles plain / shift / cmd(ctrl)
    handleThumbClick(e, imageName, idx) {
      if (e.shiftKey && this._lastClickIdx !== null) {
        // Extend selection range from anchor to current
        const lo = Math.min(this._lastClickIdx, idx);
        const hi = Math.max(this._lastClickIdx, idx);
        this.selectedImageNames = this.images.slice(lo, hi + 1).map(i => i.image_name);
      } else if (e.metaKey || e.ctrlKey) {
        // Toggle individual image
        const pos = this.selectedImageNames.indexOf(imageName);
        if (pos === -1) {
          this.selectedImageNames = [...this.selectedImageNames, imageName];
        } else {
          this.selectedImageNames = this.selectedImageNames.filter(n => n !== imageName);
        }
        this._lastClickIdx = idx;
      } else {
        // Plain click — single select + update main display
        this.selectImage(imageName, idx);
      }
    },

    handleExternalSelect(event) {
      const name = event.detail.imageName;
      this.selectedImageName = name;
      if (name) {
        this.selectedImageNames = [name];
        const idx = this.images.findIndex(i => i.image_name === name);
        if (idx !== -1) this._lastClickIdx = idx;
      } else {
        this.selectedImageNames = [];
      }
    },

    // ── Hover actions ────────────────────────────────────────────────────

    async starImage(imageName, starred) {
      const img = this.images.find(i => i.image_name === imageName);
      if (img) img.starred = starred;
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/star?starred=${starred}`, { method: 'PATCH' });
        if (resp.ok) {
          const updated = await resp.json();
          if (img) img.starred = updated.starred;
          this.$dispatch('image-star-updated', { imageName, starred: updated.starred });
        }
      } catch (e) {
        if (img) img.starred = !starred;
      }
    },

    async deleteImage(imageName) {
      if (!confirm('Delete this image?')) return;
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}`, { method: 'DELETE' });
        if (!resp.ok) return;
        this.images = this.images.filter(i => i.image_name !== imageName);
        this.offset = Math.max(0, this.offset - 1);
        this.total  = Math.max(0, this.total  - 1);
        this.selectedImageNames = this.selectedImageNames.filter(n => n !== imageName);
        if (this.selectedImageName === imageName) {
          this.selectedImageName = null;
          this.$dispatch('select-image', { imageName: null, fromGallery: true });
        }
      } catch (e) {}
    },

    // ── Bulk operations ──────────────────────────────────────────────────

    async deleteSelected() {
      const names = [...this.selectedImageNames];
      if (!names.length) return;
      if (!confirm(`Delete ${names.length} image${names.length !== 1 ? 's' : ''}?`)) return;
      this.ctxMenu.open = false;
      let removedMain = false;
      for (const name of names) {
        try {
          const resp = await fetch(`/api/v1/images/i/${name}`, { method: 'DELETE' });
          if (resp.ok) {
            this.images = this.images.filter(i => i.image_name !== name);
            this.offset = Math.max(0, this.offset - 1);
            this.total  = Math.max(0, this.total  - 1);
            if (this.selectedImageName === name) removedMain = true;
          }
        } catch (e) {}
      }
      this.selectedImageNames = [];
      if (removedMain) {
        this.selectedImageName = null;
        this.$dispatch('select-image', { imageName: null, fromGallery: true });
      }
    },

    // ── Context menu ─────────────────────────────────────────────────────

    openContextMenu(e, imageName, idx) {
      e.preventDefault();
      e.stopPropagation();
      // If right-clicked image isn't already selected, make it the selection
      if (!this.selectedImageNames.includes(imageName)) {
        this.selectedImageNames = [imageName];
        this._lastClickIdx = idx;
      }
      this.ctxMenu = { open: true, x: e.clientX, y: e.clientY };
    },

    // ── Move modal ───────────────────────────────────────────────────────

    openMoveModal() {
      this.ctxMenu.open = false;
      this.moveModal = { open: true, targetBoardId: null };
    },

    closeMoveModal() {
      this.moveModal.open = false;
    },

    async confirmMove() {
      const { targetBoardId } = this.moveModal;
      if (!targetBoardId) return;
      const names = [...this.selectedImageNames];
      this._moving = true;
      this.moveModal.open = false;
      try {
        const resp = await fetch('/api/v1/images/move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_names: names,
            source_board_id: this.selectedBoardId,
            target_board_id: targetBoardId,
          }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        this.images = this.images.filter(i => !names.includes(i.image_name));
        this.offset = Math.max(0, this.offset - names.length);
        this.total  = Math.max(0, this.total  - names.length);
        if (names.includes(this.selectedImageName)) {
          this.selectedImageName = null;
          this.$dispatch('select-image', { imageName: null, fromGallery: true });
        }
        this.selectedImageNames = [];
      } catch (err) {
        alert(`Move failed: ${err.message}`);
      } finally {
        this._moving = false;
      }
    },

    // ── Regenerate modal ──────────────────────────────────────────────

    openRegenerateModal() {
      this.ctxMenu.open = false;
      this.regenModal = { open: true, selectedModelKey: null, priority: false, _enqueuing: false, _progress: null };
      window.dispatchEvent(new CustomEvent('request-models-list'));
    },

    closeRegenerateModal() {
      if (this.regenModal._enqueuing) return;
      this.regenModal.open = false;
    },

    async confirmRegenerate() {
      const names = [...this.selectedImageNames];
      const modelKey = this.regenModal.selectedModelKey;
      // The modal checkbox is the ⏫ prepend option: rank 2 (min) so these
      // queue ahead of plain rank-1 work. Ordering uses batch.priority only.
      const priority = this.regenModal.priority ? 2 : 1;
      if (!names.length || !modelKey) return;

      this.regenModal._enqueuing = true;
      this.regenModal._progress = { done: 0, total: names.length };

      try {
        for (const name of names.toReversed()) {
          const metaResp = await fetch(`/api/v1/images/i/${name}/metadata`);
          if (!metaResp.ok) continue;
          const meta = await metaResp.json();

          const batch = {
            prompt: meta.prompt,
            negative_prompt: meta.negative_prompt || "",
            seed: meta.seed,
            width: meta.width,
            height: meta.height,
            steps: meta.steps,
            cfg: meta.cfg,
            cfg_rescale_multiplier: meta.cfg_rescale_multiplier || 0.0,
            sampler: meta.sampler,
            prediction_type: meta.prediction_type || "default",
            model_key: modelKey,
            board_id: this.selectedBoardId,
            count: 1,
            priority: priority,
          };
          if (meta.source_image) {
            batch.init_image = meta.source_image;
            batch.strength = meta.strength;
            batch.source_width = meta.source_width;
            batch.source_height = meta.source_height;
            batch.crop_left = meta.crop_left || 0.0;
            batch.crop_top = meta.crop_top || 0.0;
            batch.crop_width = meta.crop_width || 0.0;
            batch.crop_height = meta.crop_height || 0.0;
          }

          await fetch('/api/v1/queue/default/enqueue_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batch }),
          });

          this.regenModal._progress.done++;
        }
        this.selectedImageNames = [];
        this.regenModal.open = false;
      } catch (e) {
        alert(`Regenerate failed: ${e.message}`);
      } finally {
        this.regenModal._enqueuing = false;
        this.regenModal._progress = null;
      }
    },

    // ── Img2img source ───────────────────────────────────────────────────

    async useAsImg2imgSource() {
      const name = this.selectedImageNames[0];
      if (!name) return;
      this.ctxMenu.open = false;
      try {
        const resp = await fetch(`/api/v1/images/i/${name}/metadata`);
        if (!resp.ok) return;
        const meta = await resp.json();
        window.dispatchEvent(new CustomEvent('set-source-image', {
          detail: {
            name,
            url: `/api/v1/images/i/${name}/full`,
            width: meta.width,
            height: meta.height,
          },
        }));
      } catch (_) {}
    },

    // ── Drag-and-drop reordering ─────────────────────────────────────────

    onDragStart(idx, e) {
      this._dragSrcIdx = idx;
      e.dataTransfer.effectAllowed = 'move';
    },

    onDragOver(idx, e) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
    },

    onDrop(idx) {
      if (this._dragSrcIdx === null || this._dragSrcIdx === idx) {
        this._dragSrcIdx = null;
        return;
      }
      const imgs = [...this.images];
      const [moved] = imgs.splice(this._dragSrcIdx, 1);
      imgs.splice(idx, 0, moved);
      this.images = imgs;
      this._dragSrcIdx = null;
    },

    onDragEnd() { this._dragSrcIdx = null; },

    // ── WebSocket event handler ──────────────────────────────────────────

    _handleWsEvent(msg) {
      const { event, data } = msg;
      if (event === 'board_created') {
        if (!this.galleries.some(g => g.board_id === data.board_id)) {
          data.image_count = data.image_count ?? 0;
          data.latest_image = data.latest_image ?? null;
          data.latest_image_name = data.latest_image_name ?? data.latest_image ?? null;
          this.galleries.push(data);
          this._notifyGalleriesUpdated();
        }
      } else if (event === 'board_renamed') {
        const g = this.galleries.find(g => g.board_id === data.board_id);
        if (g) { g.board_name = data.board_name; this._notifyGalleriesUpdated(); }
      } else if (event === 'board_deleted') {
        this.galleries = this.galleries.filter(g => g.board_id !== data.board_id);
        this._notifyGalleriesUpdated();
        if (this.selectedBoardId === data.board_id) {
          this.selectedBoardId = this.galleries[0]?.board_id ?? null;
          this.$dispatch('select-image', { imageName: null });
          this.selectedImageName = null;
          this._clearSelection();
          this.resetAndLoad();
        }
      } else if (event === 'image_created') {
        const g = this.galleries.find(x => x.board_id === data.board_id);
        if (g) {
          g.image_count = (g.image_count || 0) + 1;
          g.latest_image = data.image_name;
          g.latest_image_name = data.image_name;
        }
        this._notifyGalleriesUpdated();
        if (this.mode === 'compare') {
          // The new image may not match the current identity hash — re-query
          // (debounced) instead of inserting (§2.8).
          this._debouncedCompareReload();
        } else if (data.board_id === this.selectedBoardId &&
            !this.images.some(i => i.image_name === data.image_name)) {
          this._insertImageInOrder(data);
          this.offset += 1;
          this.total += 1;
        }
      } else if (event === 'image_deleted') {
        const gDel = this.galleries.find(x => x.board_id === data.board_id);
        if (gDel) {
          gDel.image_count = Math.max(0, (gDel.image_count || 0) - 1);
          if (gDel.latest_image === data.image_name || gDel.latest_image_name === data.image_name) {
            gDel.latest_image = null;
            gDel.latest_image_name = null;
          }
          this._notifyGalleriesUpdated();
        }
        if (this.images.some(i => i.image_name === data.image_name)) {
          this.images = this.images.filter(i => i.image_name !== data.image_name);
          this.offset = Math.max(0, this.offset - 1);
          this.total = Math.max(0, this.total - 1);
          this.selectedImageNames = this.selectedImageNames.filter(n => n !== data.image_name);
          if (this.selectedImageName === data.image_name) {
            this.selectedImageName = null;
            this.$dispatch('select-image', { imageName: null, fromGallery: true });
          }
        }
      } else if (event === 'images_moved') {
        const srcG = this.galleries.find(x => x.board_id === data.source_board_id);
        if (srcG) {
          const moved = (data.image_names || []).length;
          srcG.image_count = Math.max(0, (srcG.image_count || 0) - moved);
          if ((data.image_names || []).includes(srcG.latest_image) || (data.image_names || []).includes(srcG.latest_image_name)) {
            srcG.latest_image = null;
            srcG.latest_image_name = null;
          }
        }
        const tgtG = this.galleries.find(x => x.board_id === data.target_board_id);
        if (tgtG) {
          const moved = (data.image_names || []).length;
          tgtG.image_count = (tgtG.image_count || 0) + moved;
          if (!tgtG.latest_image && moved > 0) {
            tgtG.latest_image = data.image_names[0];
            tgtG.latest_image_name = data.image_names[0];
          }
        }
        this._notifyGalleriesUpdated();
        // Another tab moved images out of (or into) this gallery
        if (data.source_board_id === this.selectedBoardId) {
          const movedSet = new Set(data.image_names);
          const removedCount = this.images.filter(i => movedSet.has(i.image_name)).length;
          this.images = this.images.filter(i => !movedSet.has(i.image_name));
          this.offset = Math.max(0, this.offset - removedCount);
          this.total  = Math.max(0, this.total  - removedCount);
          this.selectedImageNames = this.selectedImageNames.filter(n => !movedSet.has(n));
          if (movedSet.has(this.selectedImageName)) {
            this.selectedImageName = null;
            this.$dispatch('select-image', { imageName: null, fromGallery: true });
          }
        }
      }
    },

    async createGallery() {
      const name = prompt('Gallery name:', 'New Gallery');
      if (!name) return;
      try {
        const resp = await fetch(`/api/v1/boards/?board_name=${encodeURIComponent(name)}`, { method: 'POST' });
        if (resp.ok) {
          const g = await resp.json();
          this.galleries.push(g);
          this._notifyGalleriesUpdated();
          this.selectedBoardId = g.board_id;
          this.resetAndLoad();
        }
      } catch (e) {}
    },
  };
}

// Register with Alpine so x-data="gallery" resolves reliably before DOM scan.
document.addEventListener('alpine:init', () => {
  Alpine.data('gallery', gallery);
});

window.gallery = gallery;

