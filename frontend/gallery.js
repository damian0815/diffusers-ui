/* frontend/gallery.js — Gallery panel Alpine.js component */

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
    offset: 0,
    limit: 50,
    total: 0,
    loading: false,
    hasMore: true,
    selectedImageName: null,
    thumbsPerRow: 3,
    _dragSrcIdx: null,
    _pendingRestore: false,

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
      prepend: false,
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
      await this.loadGalleries();
      this._notifyBoardChanged();
      this.$watch('selectedBoardId', (newVal, oldVal) => {
        this._notifyBoardChanged();
        this._saveGalleryState(oldVal);
      });
      this.$watch('selectedImageName', () => this._saveGalleryState(this.selectedBoardId));

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
        if (this.galleries.length > 0 && !this.selectedBoardId) {
          this.selectedBoardId = this.galleries[0].board_id;
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
      if (this.loading || !this.hasMore || !this.selectedBoardId || this._isNavigating) return;
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
        if (this._pendingRestore) {
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
      this.regenModal = { open: true, selectedModelKey: null, prepend: false, _enqueuing: false, _progress: null };
      window.dispatchEvent(new CustomEvent('request-models-list'));
    },

    closeRegenerateModal() {
      if (this.regenModal._enqueuing) return;
      this.regenModal.open = false;
    },

    async confirmRegenerate() {
      const names = [...this.selectedImageNames];
      const modelKey = this.regenModal.selectedModelKey;
      const prepend = this.regenModal.prepend;
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
            body: JSON.stringify({ prepend, batch }),
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
        if (!this.galleries.some(g => g.board_id === data.board_id))
          this.galleries.push(data);
      } else if (event === 'board_renamed') {
        const g = this.galleries.find(g => g.board_id === data.board_id);
        if (g) g.board_name = data.board_name;
      } else if (event === 'board_deleted') {
        this.galleries = this.galleries.filter(g => g.board_id !== data.board_id);
        if (this.selectedBoardId === data.board_id) {
          this.selectedBoardId = this.galleries[0]?.board_id ?? null;
          this.$dispatch('select-image', { imageName: null });
          this.selectedImageName = null;
          this._clearSelection();
          this.resetAndLoad();
        }
      } else if (event === 'image_created') {
        if (data.board_id === this.selectedBoardId &&
            !this.images.some(i => i.image_name === data.image_name)) {
          this._insertImageInOrder(data);
          this.offset += 1;
          this.total += 1;
        }
      } else if (event === 'image_deleted') {
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

