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
    loading: false,
    hasMore: true,
    selectedImageName: null,
    thumbsPerRow: 3,

    async init() {
      await this.loadGalleries();
      this.setupInfiniteScroll();
      this.setupArrowNav();
      window.addEventListener('image-completed', (e) => {
        this.prependImage(e.detail.imageName);
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
      this.images = []; this.offset = 0; this.hasMore = true;
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
        const batch = await resp.json();
        this.images.push(...batch);
        this.offset += batch.length;
        this.hasMore = batch.length === this.limit;
      } catch (e) {
      } finally {
        this.loading = false;
      }
    },

    async prependImage(imageName) {
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) {
          const meta = await resp.json();
          this.images.unshift(meta);
          this.offset += 1;
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
          // Scroll selected thumbnail into view
          document.querySelectorAll('.thumb-wrapper')[newIdx]?.scrollIntoView({ block: 'nearest' });
        }
        // Pre-load more images when nearing the end
        if (newIdx >= this.images.length - 10) this.loadMore();
      });
    },

    selectImage(imageName) {
      this.selectedImageName = imageName;
      this.$dispatch('select-image', { imageName });
    },

    handleExternalSelect(event) {
      this.selectedImageName = event.detail.imageName;
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
