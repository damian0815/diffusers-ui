/* frontend/gallery.js — Gallery panel Alpine.js component */

function gallery() {
  return {
    galleries: [],
    selectedBoardId: null,
    images: [],
    search: '',
    starredOnly: false,
    offset: 0,
    limit: 50,
    total: 0,
    loading: false,
    hasMore: true,
    selectedImageName: null,
    thumbsPerRow: 3,
    _dragSrcIdx: null,

    // ── Multi-select ─────────────────────────────────────────────────────
    selectedImageNames: [],   // all highlighted thumbnails
    _lastClickIdx: null,      // anchor for shift-range
    _moving: false,

    // ── Context menu ─────────────────────────────────────────────────────
    ctxMenu: { open: false, x: 0, y: 0 },

    // ── Move modal ───────────────────────────────────────────────────────
    moveModal: { open: false, targetBoardId: null },

    _notifyBoardChanged() {
      window.dispatchEvent(new CustomEvent('board-changed', {
        detail: { boardId: this.selectedBoardId },
      }));
    },

    async init() {
      await this.loadGalleries();
      this._notifyBoardChanged();
      this.$watch('selectedBoardId', () => this._notifyBoardChanged());

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

      // Close context menu on any outside click; Escape closes both menu + modal
      window.addEventListener('click', () => { this.ctxMenu.open = false; });
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          this.ctxMenu.open = false;
          this.moveModal.open = false;
        }
      });
    },

    async loadGalleries() {
      try {
        const resp = await fetch('/api/v1/boards/');
        if (!resp.ok) return;
        this.galleries = await resp.json();
        if (this.galleries.length > 0 && !this.selectedBoardId) {
          this.selectedBoardId = this.galleries[0].board_id;
          await this.loadMore();
        }
      } catch (e) {}
    },

    onBoardChange() {
      this.$dispatch('select-image', { imageName: null });
      this.selectedImageName = null;
      this._clearSelection();
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
      this.resetAndLoad();
    },

    _clearSelection() {
      this.selectedImageNames = [];
      this._lastClickIdx = null;
    },

    async loadMore() {
      if (this.loading || !this.hasMore || !this.selectedBoardId) return;
      this.loading = true;
      try {
        const params = new URLSearchParams({
          board_id: this.selectedBoardId, offset: this.offset, limit: this.limit,
        });
        if (this.search) params.set('search', this.search);
        if (this.starredOnly) params.set('starred_only', 'true');
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
      }
    },

    async prependImage(imageName) {
      if (this.images.some(i => i.image_name === imageName)) return;
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) {
          const meta = await resp.json();
          if (!this.images.some(i => i.image_name === meta.image_name)) {
            this.images.unshift(meta);
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
          this.images.unshift(data);
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
