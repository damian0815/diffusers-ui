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

    async init() {
      await this.loadGalleries();
      this.setupInfiniteScroll();
      // Listen for new images from the generator
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
      } catch (e) {
        // non-fatal
      }
    },

    onBoardChange() {
      this.resetAndLoad();
    },

    resetAndLoad() {
      this.images = [];
      this.offset = 0;
      this.hasMore = true;
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
          board_id: this.selectedBoardId,
          offset: this.offset,
          limit: this.limit,
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
        // non-fatal
      } finally {
        this.loading = false;
      }
    },

    async prependImage(imageName) {
      // Fetch the metadata and prepend to the current view
      try {
        const resp = await fetch(`/api/v1/images/i/${imageName}/metadata`);
        if (resp.ok) {
          const meta = await resp.json();
          this.images.unshift(meta);
          this.offset += 1;
        }
      } catch (e) {
        // Fall back to a soft reload
        this.resetAndLoad();
      }
    },

    setupInfiniteScroll() {
      const sentinel = document.getElementById('gallery-sentinel');
      if (!sentinel) return;
      const observer = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) this.loadMore();
      }, { rootMargin: '100px' });
      observer.observe(sentinel);
    },

    selectImage(imageName) {
      this.selectedImageName = imageName;
      // Bubble up to the parent app component
      this.$dispatch('select-image', { imageName });
    },

    handleExternalSelect(event) {
      // Called when app.js selects an image (e.g. from the completed job)
      this.selectedImageName = event.detail.imageName;
    },

    async createGallery() {
      const name = prompt('Gallery name:', 'New Gallery');
      if (!name) return;
      try {
        const resp = await fetch(`/api/v1/boards/?board_name=${encodeURIComponent(name)}`, {
          method: 'POST',
        });
        if (resp.ok) {
          const gallery = await resp.json();
          this.galleries.push(gallery);
          this.selectedBoardId = gallery.board_id;
          this.resetAndLoad();
        }
      } catch (e) {
        // non-fatal
      }
    },
  };
}

