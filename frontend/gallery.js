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

    async init() {
      await this.loadGalleries();
      this.setupInfiniteScroll();
      this.setupArrowNav();
      window.addEventListener('image-completed', (e) => {
        this.prependImage(e.detail.imageName);
      });
      window.addEventListener('ws-gallery-event', (e) => {
        this._handleWsEvent(e.detail);
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

    onBoardChange() { this.resetAndLoad(); },

    resetAndLoad() {
      this.images = []; this.offset = 0; this.total = 0; this.hasMore = true;
      this.loadMore();
    },

    async selectGallery(boardId) {
      this.selectedBoardId = boardId;
      this.resetAndLoad();
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
      if (this.images.some(i => i.image_name === imageName)) return; // already added via WS
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
          this.selectImage(this.images[0].image_name);
          return;
        }

        const isNext = e.key === 'ArrowRight' || e.key === 'ArrowDown';
        const delta  = e.key === 'ArrowDown' || e.key === 'ArrowUp' ? this.thumbsPerRow : 1;
        const newIdx = Math.max(0, Math.min(this.images.length - 1, idx + (isNext ? delta : -delta)));

        if (newIdx !== idx) {
          this.selectImage(this.images[newIdx].image_name);
          document.querySelectorAll('.thumb-wrapper')[newIdx]?.scrollIntoView({ block: 'nearest' });
        }
        if (newIdx >= this.images.length - 10) this.loadMore();
      });
    },

    selectImage(imageName) {
      this.selectedImageName = imageName;
      this.$dispatch('select-image', { imageName, fromGallery: true });
    },

    handleExternalSelect(event) {
      this.selectedImageName = event.detail.imageName;
    },

    // ── Hover actions ────────────────────────────────────────────────────

    async starImage(imageName, starred) {
      const img = this.images.find(i => i.image_name === imageName);
      if (img) img.starred = starred; // optimistic
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/star?starred=${starred}`, { method: 'PATCH' });
        if (resp.ok) {
          const updated = await resp.json();
          if (img) img.starred = updated.starred;
          this.$dispatch('image-star-updated', { imageName, starred: updated.starred });
        }
      } catch (e) {
        if (img) img.starred = !starred; // revert
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
        if (this.selectedImageName === imageName) {
          this.selectedImageName = null;
          this.$dispatch('select-image', { imageName: null, fromGallery: true });
        }
      } catch (e) {}
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
          if (this.selectedImageName === data.image_name) {
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

