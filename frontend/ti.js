function ti() {
    return {
        status: 'idle',
        config: {
            gallery_id: '',
            placeholder_token: '<sks>',
            initializer_token: 'dog',
            num_vectors: 1,
            learnable_property: 'object',
            class_prompt: '',
            model_key: '',
            resolution: 512,
            max_train_steps: 500,
            learning_rate: 5e-4,
            lr_scheduler: 'constant',
            train_batch_size: 1,
            mixed_precision: 'fp16',
            seed: 42,
            validation_prompt: '',
            validation_steps: 100,
            save_steps: 100,
            output_name: 'learned_embeds',
            repeats: 100
        },
        galleries: [],
        models: [],
        currentStep: 0,
        totalSteps: 0,
        trainLoss: 0,
        valLoss: null,
        lr: 0,
        elapsed: 0,
        samples: [],
        embeddings: [],
        error: null,

        async init() {
            await this.loadGalleries();
            await this.loadModels();
            await this.loadEmbeddings();
            this.listenForEvents();
        },

        listenForEvents() {
            window.addEventListener('ws-gallery-event', (e) => {
                const {type, data} = e.detail;
                if (type === 'ti_progress') this.updateProgress(data);
                if (type === 'ti_validation') this.updateValidation(data);
                if (type === 'ti_complete') this.onComplete(data);
                if (type === 'ti_error') this.onError(data);
            });
        },

        async loadGalleries() {
            const res = await fetch('/api/v1/boards/');
            this.galleries = await res.json();
            if (this.galleries.length > 0 && !this.config.gallery_id) {
                this.config.gallery_id = this.galleries[0].board_id;
            }
        },

        async loadModels() {
            const res = await fetch('/api/v1/models/');
            this.models = (await res.json()).filter(m => m.base !== 'sana');
            if (this.models.length > 0 && !this.config.model_key) {
                this.config.model_key = this.models[0].key;
            }
        },

        async loadEmbeddings() {
            const res = await fetch('/api/v1/training/embeddings');
            this.embeddings = await res.json();
        },

        async startTraining() {
            this.error = null;
            const res = await fetch('/api/v1/training/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(this.config)
            });
            if (res.ok) {
                this.status = 'running';
            } else {
                const err = await res.json();
                this.error = err.detail || 'Failed to start training';
            }
        },

        async pauseTraining() {
            await fetch('/api/v1/training/pause', { method: 'PUT' });
            this.status = 'paused';
        },

        async resumeTraining() {
            await fetch('/api/v1/training/resume', { method: 'PUT' });
            this.status = 'running';
        },

        async stopTraining() {
            await fetch('/api/v1/training/stop', { method: 'PUT' });
        },

        async deleteEmbedding(name) {
            if (confirm(`Delete embedding ${name}?`)) {
                await fetch(`/api/v1/training/embeddings/${name}`, { method: 'DELETE' });
                await this.loadEmbeddings();
                // Also notify the generate tab
                window.dispatchEvent(new CustomEvent('request-models-list'));
            }
        },

        updateProgress(data) {
            this.status = 'running';
            this.currentStep = data.current_step;
            this.totalSteps = data.total_steps;
            this.trainLoss = data.train_loss;
            this.lr = data.lr;
            this.elapsed = data.elapsed_seconds;
        },

        updateValidation(data) {
            this.valLoss = data.val_loss;
            this.samples = data.samples;
        },

        onComplete(data) {
            this.status = 'idle';
            this.loadEmbeddings();
        },

        onError(data) {
            this.status = 'idle';
            this.error = data.error;
        },

        formatTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
        }
    }
}
