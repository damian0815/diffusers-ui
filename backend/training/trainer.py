import asyncio
import contextlib
import logging
import random
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import safetensors.torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from backend.config import settings
from backend.domain import TIStatus, TIState
from backend.training.config import TIConfig
from backend.training.dataset import TextualInversionDataset

from backend.model_registry.loader import PipelineLoader
from backend.model_registry.registry import ModelRegistry
from backend.pipeline.preview import PreviewBroadcaster
from backend.training.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def isolate_rng(include_cuda=True):
    """Save and restore all RNG states so validation doesn't perturb training."""
    # Save states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = []
    if include_cuda and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cuda_states.append(torch.cuda.get_rng_state(i))
    yield
    # Restore states
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    if include_cuda and torch.cuda.is_available():
        for i, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, i)

def _embedding_forward_with_offsets(module, input_args, output):
    """Registered as a forward hook on the Embedding layer."""
    offset_weight = _apply_weight_offsets(
        module.training_token_ids, module, module.embedding_offsets_individual
    )
    return F.embedding(
        input_args[0], offset_weight, module.padding_idx, module.max_norm,
        module.norm_type, module.scale_grad_by_freq, module.sparse
    )

def _apply_weight_offsets(training_token_ids, original_embeddings, offsets):
    """Build a full-size weight tensor with offsets applied at the training positions."""
    offset_weight = original_embeddings.weight.clone()
    offset_weight.index_add_(0, training_token_ids, offsets)
    return offset_weight

class TITrainer:
    def __init__(self, loader: PipelineLoader, registry: ModelRegistry, broadcaster: PreviewBroadcaster,
                 embedding_service: EmbeddingService | None = None):
        self._loader = loader
        self._registry = registry
        self._broadcaster = broadcaster
        self._embedding_service = embedding_service
        self._state = TIState(status=TIStatus.IDLE)
        self._config: TIConfig | None = None
        self._task: asyncio.Task | None = None
        self._paused = asyncio.Event()
        self._paused.set()
        self._stop_requested = False
        self._training_token_ids = None

    @property
    def state(self) -> TIState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state.status in (TIStatus.RUNNING, TIStatus.PAUSED)

    async def start(self, config: TIConfig):
        if self.is_running:
            raise ValueError("Training is already running")
        self._config = config
        self._stop_requested = False
        self._state = TIState(status=TIStatus.RUNNING, config_name=config.output_name, total_steps=config.max_train_steps)
        self._task = asyncio.create_task(self._run())

    def pause(self):
        if self._state.status == TIStatus.RUNNING:
            self._paused.clear()
            self._state.status = TIStatus.PAUSED

    def resume(self):
        if self._state.status == TIStatus.PAUSED:
            self._paused.set()
            self._state.status = TIStatus.RUNNING

    def stop(self):
        if self.is_running:
            self._stop_requested = True
            self._paused.set()

    async def _run(self):
        try:
            entry = self._registry.get(self._config.model_key)
            if not entry:
                raise ValueError(f"Model {self._config.model_key} not found")
            if entry.base == "sana":
                raise ValueError("SANA models are not supported for Textual Inversion")

            # Load pipeline and freeze everything
            # TITrainer handles loading the pipeline. Since PipelineLoader holds one model,
            # we must ensure inference is not running. The frontend enforces mutual exclusion.
            pipeline = await asyncio.to_thread(self._loader.load, entry)
            
            text_encoder = pipeline.text_encoder
            tokenizer = pipeline.tokenizer
            vae = pipeline.vae
            unet = pipeline.unet
            noise_scheduler = pipeline.scheduler

            # Add tokens
            placeholder_token_ids = _add_tokens(tokenizer, self._config)
            text_encoder.resize_token_embeddings(len(tokenizer))
            _initialize_embeddings(text_encoder, tokenizer, placeholder_token_ids, self._config)

            # Freeze everything
            vae.requires_grad_(False)
            unet.requires_grad_(False)
            text_encoder.requires_grad_(False)

            # Create offset parameter
            embedding_dim = text_encoder.get_input_embeddings().weight.shape[1]
            offsets = nn.Parameter(
                torch.zeros(len(placeholder_token_ids), embedding_dim, device=text_encoder.device, dtype=text_encoder.dtype),
                requires_grad=True,
            )

            # Register hook
            embeddings_layer = text_encoder.get_input_embeddings()
            embeddings_layer.training_token_ids = torch.tensor(placeholder_token_ids, device=text_encoder.device, dtype=torch.long)
            embeddings_layer.embedding_offsets_individual = offsets
            handle = embeddings_layer.register_forward_hook(_embedding_forward_with_offsets)
            self._training_token_ids = placeholder_token_ids
            self._offsets = offsets

            # Prepare dataset
            from backend.gallery.service import GalleryService
            from backend.gallery.db import AsyncSessionLocal
            
            async with AsyncSessionLocal() as session:
                service = GalleryService(session)
                images = await service.list_images(board_id=self._config.gallery_id, limit=100000)
                image_names = [img.image_name for img in images]
                await session.commit()

            if not image_names:
                raise ValueError("Gallery is empty")

            dataset = TextualInversionDataset(
                image_names, tokenizer, self._config.placeholder_token,
                self._config.resolution, self._config.repeats,
                self._config.learnable_property,
                class_prompt=self._config.class_prompt,
            )
            dataloader = DataLoader(dataset, batch_size=self._config.train_batch_size, shuffle=True)

            # Optimizer
            optimizer = AdamW([offsets], lr=self._config.learning_rate,
                              betas=(self._config.adam_beta1, self._config.adam_beta2),
                              weight_decay=self._config.adam_weight_decay,
                              eps=self._config.adam_epsilon)

            # Split
            val_size = min(len(dataset) // 5, 50)
            train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
            val_dataloader = DataLoader(val_dataset, batch_size=self._config.train_batch_size)

            weight_dtype = _resolve_dtype(self._config.mixed_precision)
            t0 = time.time()

            for step in range(self._config.max_train_steps):
                await self._paused.wait()
                if self._stop_requested:
                    break

                text_encoder.train()
                batch = next(iter(dataloader))

                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                    latents *= vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device))[0].to(weight_dtype)
                model_pred = unet(noisy_latents.to(weight_dtype), timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self._state.current_step = step + 1
                self._state.train_loss = loss.item()
                self._state.lr = optimizer.param_groups[0]["lr"]
                self._state.elapsed_seconds = time.time() - t0

                await self._broadcaster.broadcast_event("ti_progress", self._state.__dict__)

                if (step + 1) % self._config.validation_steps == 0:
                    val_loss, val_images = await self._run_validation(
                        text_encoder, vae, unet, noise_scheduler, tokenizer,
                        val_dataloader, self._config, weight_dtype
                    )
                    self._state.val_loss = val_loss
                    self._state.samples = val_images
                    await self._broadcaster.broadcast_event("ti_validation", self._state.__dict__)

                if (step + 1) % self._config.save_steps == 0:
                    await self._save_embeddings(text_encoder, placeholder_token_ids, self._offsets, step + 1)

            # Final save
            await self._save_embeddings(text_encoder, placeholder_token_ids, self._offsets, None)
            
            # Clean up hook and restore embeddings
            handle.remove()
            self._loader.unload()
            self._state.status = TIStatus.IDLE
            await self._broadcaster.broadcast_event("ti_complete", {"final_loss": self._state.train_loss})

        except Exception as e:
            self._state.status = TIStatus.IDLE
            self._state.error = str(e)
            logger.exception("Training failed")
            await self._broadcaster.broadcast_event("ti_error", {"error": str(e)})
        finally:
            self._task = None

    async def _run_validation(self, text_encoder, vae, unet, scheduler, tokenizer, val_dataloader, config, weight_dtype):
        text_encoder.eval()
        total_loss = 0.0
        count = 0
        
        with isolate_rng():
            torch.manual_seed(config.validation_seed)
            random.seed(config.validation_seed)
            np.random.seed(config.validation_seed)
            
            with torch.no_grad():
                for batch in val_dataloader:
                    latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                    latents *= vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device))[0].to(weight_dtype)
                    model_pred = unet(noisy_latents.to(weight_dtype), timesteps, encoder_hidden_states).sample
                    
                    if scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    
                    total_loss += F.mse_loss(model_pred.float(), target.float(), reduction="mean").item()
                    count += 1

            # Generate samples if prompt provided
            sample_images = []
            if config.validation_prompt:
                # Temporarily merge offsets for generation
                _apply_weight_offsets_in_place(text_encoder.get_input_embeddings(), self._training_token_ids, self._offsets)
                from diffusers import DiffusionPipeline
                pipe = DiffusionPipeline.from_pretrained(
                    self._registry.get(config.model_key).source,
                    text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, vae=vae
                )
                gen = torch.Generator().manual_seed(config.validation_seed)
                for _ in range(config.num_validation_images):
                    img = pipe(config.validation_prompt, num_inference_steps=25, generator=gen).images[0]
                    name = f"{uuid.uuid4()}.png"
                    img.save(settings.images_dir / name)
                    sample_images.append(name)
                del pipe

        text_encoder.train()

        return total_loss / max(count, 1), sample_images

    async def _save_embeddings(self, text_encoder, placeholder_token_ids, offsets, step):
        merged = _apply_weight_offsets(torch.tensor(placeholder_token_ids, device=text_encoder.device), text_encoder.get_input_embeddings(), offsets)
        embeds = merged.detach().cpu()

        embed_dict = {self._config.placeholder_token: embeds}
        suffix = f"-step{step}" if step else ""
        path = settings.embeddings_dir / f"{self._config.output_name}{suffix}.safetensors"
        settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(embed_dict, str(path), metadata={"format": "pt"})
        
        # Update local state for UI
        self._state.embedding_name = f"{self._config.output_name}{suffix}"

        # Notify embedding service so the worker picks up new files
        if self._embedding_service:
            self._embedding_service.rescan()

def _add_tokens(tokenizer, config):
    placeholder_tokens = [config.placeholder_token]
    if config.num_vectors > 1:
        placeholder_tokens += [f"{config.placeholder_token}.{i}" for i in range(1, config.num_vectors)]
    tokenizer.add_tokens(placeholder_tokens)
    return tokenizer.convert_tokens_to_ids(placeholder_tokens)

def _initialize_embeddings(text_encoder, tokenizer, placeholder_token_ids, config):
    initializer_token_id = tokenizer.convert_tokens_to_ids(config.initializer_token)
    if initializer_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Initializer token '{config.initializer_token}' not in vocabulary")
        
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            text_encoder.get_input_embeddings().weight[token_id] = text_encoder.get_input_embeddings().weight[initializer_token_id].clone()

def _resolve_dtype(mp_str):
    if mp_str == "fp16":
        return torch.float16
    elif mp_str == "bf16":
        return torch.bfloat16
    return torch.float32

def _apply_weight_offsets_in_place(embeddings_layer, training_token_ids, offsets):
    """Temporarily merge offsets for inference"""
    merged = _apply_weight_offsets(training_token_ids, embeddings_layer, offsets)
    embeddings_layer.weight.data[training_token_ids] = merged[training_token_ids]
