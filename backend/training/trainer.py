import asyncio
import contextlib
import logging
import math
import random
import threading
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
from tqdm import tqdm

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

def _draw_flowmatch_timesteps(batch_size: int, num_timesteps: int,
                              timesteps: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Draw random float timesteps for flow-matching training.

    Draws sigma uniformly in [0, 1), converts to float timestep values
    via reversed-index lookup into the scheduler's timesteps tensor.
    """
    sigmas = torch.rand(batch_size, device=device)
    indices = (sigmas * num_timesteps).long().clamp(0, num_timesteps - 1)
    reversed_indices = num_timesteps - 1 - indices
    return timesteps[reversed_indices.to(timesteps.device)]


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

class EmbeddingEMA:
    """Exponential moving average of embedding offsets for stable saved output."""

    def __init__(self, offsets: nn.Parameter, decay: float = 0.9999):
        self.decay = decay
        self.shadow = offsets.data.clone()

    @torch.no_grad()
    def update(self, offsets: nn.Parameter):
        self.shadow.mul_(self.decay).add_(offsets.data, alpha=1 - self.decay)

    def get(self) -> torch.Tensor:
        return self.shadow.clone()


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
        self._sample_request_event = asyncio.Event()
        self._sample_cancel_event = threading.Event()

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

    def request_samples(self):
        """Request on-demand sample generation. No-op if not running."""
        if self.is_running:
            self._sample_request_event.set()

    def cancel_samples(self):
        """Cancel running on-demand sample generation."""
        self._sample_cancel_event.set()

    async def _run(self):
        handle = None
        try:
            entry = self._registry.get(self._config.model_key)
            if not entry:
                raise ValueError(f"Model {self._config.model_key} not found")

            is_sana = entry.base == "sana"
            self._is_sana = is_sana

            # Load pipeline and freeze everything
            # TITrainer handles loading the pipeline. Since PipelineLoader holds one model,
            # we must ensure inference is not running. The frontend enforces mutual exclusion.
            pipeline = await asyncio.to_thread(self._loader.load, entry)
            self._pipeline = pipeline
            
            text_encoder = pipeline.text_encoder
            tokenizer = pipeline.tokenizer
            vae = pipeline.vae

            # prevent int overrun errors
            tokenizer.model_max_length = 512

            if is_sana:
                denoising_model = pipeline.transformer
                from diffusers import FlowMatchEulerDiscreteScheduler
                noise_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000, shift=1.0
                )
            else:
                denoising_model = pipeline.unet
                noise_scheduler = pipeline.scheduler
            self._noise_scheduler = noise_scheduler

            # Add tokens
            placeholder_token_ids = _add_tokens(tokenizer, self._config)
            text_encoder.resize_token_embeddings(len(tokenizer))
            _initialize_embeddings(text_encoder, tokenizer, placeholder_token_ids, self._config)

            # Freeze everything
            vae.requires_grad_(False)
            denoising_model.requires_grad_(False)
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
            self._training_token_ids = embeddings_layer.training_token_ids
            self._offsets = offsets

            ema = EmbeddingEMA(offsets, decay=0.998)

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
                num_vectors=self._config.num_vectors,
                augment_max_rotation_degrees=self._config.augment_max_rotation_degrees,
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

            # SANA models are loaded in bfloat16 by the pipeline loader,
            # so training must use bfloat16 to match the model weights.
            if is_sana and weight_dtype != torch.bfloat16:
                weight_dtype = torch.bfloat16

            t0 = time.time()

            steps_per_epoch = max(1, len(train_dataset) // self._config.train_batch_size)
            total_epochs = math.ceil(self._config.max_train_steps / steps_per_epoch)
            self._state.total_steps = self._config.max_train_steps
            self._state.total_epochs = total_epochs
            self._state.total_steps_in_epoch = steps_per_epoch

            global_step = 0

            # Generate initial samples at step 0 before training starts
            if self._config.validation_prompt:
                val_loss, val_images = await self._run_validation(
                    val_dataloader, self._config, weight_dtype
                )
                self._state.val_loss = val_loss
                self._state.samples = val_images
                await self._broadcaster.broadcast_event("ti_validation", self._state.__dict__)

            epoch_pbar = tqdm(range(total_epochs), desc="Epochs", unit="epoch", position=0)
            for epoch in epoch_pbar:
                if self._stop_requested:
                    break

                self._state.current_epoch = epoch + 1
                self._state.steps_in_epoch = 0

                step_pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="step", position=1, leave=False)
                for batch_idx, batch in enumerate(step_pbar):
                    await self._paused.wait()
                    if self._stop_requested:
                        break

                    # Check for on-demand sample request
                    if self._sample_request_event.is_set():
                        self._sample_request_event.clear()
                        await self._generate_samples_on_demand(self._config)

                    text_encoder.train()

                    with torch.no_grad():
                        if is_sana:
                            latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent
                        else:
                            latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                        latents *= vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    if is_sana:
                        timesteps = _draw_flowmatch_timesteps(
                            bsz, noise_scheduler.config.num_train_timesteps,
                            noise_scheduler.timesteps, device=latents.device,
                        )
                        noisy_latents = noise_scheduler.scale_noise(latents, timesteps, noise)
                    else:
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                        )
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    if is_sana:
                        text_input_ids = batch["input_ids"].to(text_encoder.device)
                        attention_mask = (text_input_ids != tokenizer.pad_token_id).long()
                        encoder_hidden_states = text_encoder(
                            text_input_ids, attention_mask=attention_mask
                        )[0].to(weight_dtype)
                    else:
                        encoder_hidden_states = text_encoder(
                            batch["input_ids"].to(text_encoder.device)
                        )[0].to(weight_dtype)

                    if is_sana:
                        model_pred = denoising_model(
                            hidden_states=noisy_latents.to(denoising_model.device, dtype=weight_dtype),
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=timesteps.to(device=noisy_latents.device, dtype=weight_dtype),
                        ).sample
                    else:
                        model_pred = denoising_model(
                            noisy_latents.to(weight_dtype), timesteps, encoder_hidden_states
                        ).sample

                    if is_sana:
                        target = noise - latents
                    elif noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(offsets)

                    global_step += 1
                    self._state.current_step = global_step
                    self._state.steps_in_epoch = batch_idx + 1
                    self._state.train_loss = loss.item()
                    self._state.lr = optimizer.param_groups[0]["lr"]
                    self._state.elapsed_seconds = time.time() - t0

                    step_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)
                    epoch_pbar.set_postfix(step=f"{global_step}/{self._config.max_train_steps}", loss=f"{loss.item():.4f}", refresh=False)

                    await self._broadcaster.broadcast_event("ti_progress", self._state.__dict__)
                    await asyncio.sleep(0)

                    if global_step % self._config.validation_steps == 0:
                        val_loss, val_images = await self._run_validation(
                            val_dataloader, self._config, weight_dtype
                        )
                        self._state.val_loss = val_loss
                        self._state.samples = val_images
                        await self._broadcaster.broadcast_event("ti_validation", self._state.__dict__)

                    if global_step % self._config.save_steps == 0:
                        await self._save_embeddings(text_encoder, self._training_token_ids, ema.get(), global_step)

                    if global_step >= self._config.max_train_steps:
                        break

                step_pbar.close()

                if global_step >= self._config.max_train_steps:
                    break

            epoch_pbar.close()

            # Final save
            await self._save_embeddings(text_encoder, self._training_token_ids, ema.get(), None)

            self._state.status = TIStatus.IDLE
            await self._broadcaster.broadcast_event("ti_complete", {"final_loss": self._state.train_loss})

        except Exception as e:
            self._state.status = TIStatus.IDLE
            self._state.error = str(e)
            logger.exception("Training failed")
            await self._broadcaster.broadcast_event("ti_error", {"error": str(e)})
        finally:
            if handle is not None:
                handle.remove()
            self._loader.unload()
            self._task = None

    async def _run_validation(self, val_dataloader, config, weight_dtype):
        text_encoder = self._pipeline.text_encoder
        vae = self._pipeline.vae
        tokenizer = self._pipeline.tokenizer
        scheduler = self._noise_scheduler
        if self._is_sana:
            denoising_model = self._pipeline.transformer
        else:
            denoising_model = self._pipeline.unet

        text_encoder.eval()
        total_loss = 0.0
        count = 0
        
        with isolate_rng():
            torch.manual_seed(config.validation_seed)
            random.seed(config.validation_seed)
            np.random.seed(config.validation_seed)
            
            with torch.no_grad():
                for batch in val_dataloader:
                    if self._is_sana:
                        latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent
                    else:
                        latents = vae.encode(batch["pixel_values"].to(vae.dtype)).latent_dist.sample()
                    latents *= vae.config.scaling_factor
                    noise = torch.randn_like(latents)

                    if self._is_sana:
                        timesteps = _draw_flowmatch_timesteps(
                            latents.shape[0], scheduler.config.num_train_timesteps,
                            scheduler.timesteps, device=latents.device,
                        )
                        noisy_latents = scheduler.scale_noise(latents, timesteps, noise)
                    else:
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    
                    if self._is_sana:
                        text_input_ids = batch["input_ids"].to(text_encoder.device)
                        attention_mask = (text_input_ids != tokenizer.pad_token_id).long()
                        encoder_hidden_states = text_encoder(
                            text_input_ids, attention_mask=attention_mask
                        )[0].to(weight_dtype)
                    else:
                        encoder_hidden_states = text_encoder(
                            batch["input_ids"].to(text_encoder.device)
                        )[0].to(weight_dtype)

                    if self._is_sana:
                        model_pred = denoising_model(
                            hidden_states=noisy_latents.to(weight_dtype),
                            encoder_hidden_states=encoder_hidden_states,
                            timestep=timesteps.to(device=noisy_latents.device, dtype=weight_dtype),
                        ).sample
                    else:
                        model_pred = denoising_model(
                            noisy_latents.to(weight_dtype), timesteps, encoder_hidden_states
                        ).sample
                    
                    if self._is_sana:
                        target = noise - latents
                    elif scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    
                    total_loss += F.mse_loss(model_pred.float(), target.float(), reduction="mean").item()
                    count += 1

            # Generate samples if prompt provided
            sample_images = []
            if config.validation_prompt:
                pipe = self._pipeline
                gen = torch.Generator().manual_seed(config.validation_seed)
                pipe_kwargs = dict(num_inference_steps=25, generator=gen)
                if self._is_sana:
                    pipe_kwargs["guidance_scale"] = 4.5
                with torch.no_grad():
                    for _ in range(config.num_validation_images):
                        img = pipe(config.validation_prompt, **pipe_kwargs).images[0]
                        name = f"{uuid.uuid4()}.png"
                        img.save(settings.images_dir / name)
                        sample_images.append(name)

        text_encoder.train()

        return total_loss / max(count, 1), sample_images

    async def _generate_samples_on_demand(self, config):
        """Generate sample images on-demand. Cancellable via _sample_cancel_event."""
        if not config.validation_prompt:
            return

        self._state.generating_samples = True
        self._sample_cancel_event.clear()
        await self._broadcaster.broadcast_event("ti_sample_status", {"generating": True})

        try:
            with isolate_rng(), torch.no_grad():
                pipe = self._pipeline
                gen = torch.Generator().manual_seed(config.validation_seed)
                pipe_kwargs = dict(num_inference_steps=25, generator=gen)
                if self._is_sana:
                    pipe_kwargs["guidance_scale"] = 4.5

                sample_images = []
                for i in range(config.num_validation_images):
                    if self._sample_cancel_event.is_set():
                        break
                    img = pipe(config.validation_prompt, **pipe_kwargs).images[0]
                    name = f"{uuid.uuid4()}.png"
                    img.save(settings.images_dir / name)
                    sample_images.append(name)

            if sample_images:
                self._state.samples = sample_images
                await self._broadcaster.broadcast_event("ti_validation", self._state.__dict__)
        finally:
            self._state.generating_samples = False
            await self._broadcaster.broadcast_event("ti_sample_status", {"generating": False})

    async def _save_embeddings(self, text_encoder, training_token_ids, offsets, step):
        merged = _apply_weight_offsets(training_token_ids, text_encoder.get_input_embeddings(), offsets)
        embeds = merged[training_token_ids].detach().cpu()
        logging.info(f"extracting embeddings for placeholder tokens: {training_token_ids.tolist()}. means: {embeds.mean(dim=1)}")

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
    bare_name = config.placeholder_token.strip("<>")
    placeholder_tokens = [f"{bare_name}.{i}" for i in range(config.num_vectors)]
    tokenizer.add_tokens(placeholder_tokens)
    return tokenizer.convert_tokens_to_ids(placeholder_tokens)

def _initialize_embeddings(text_encoder, tokenizer, placeholder_token_ids, config):
    init_ids = tokenizer.encode(config.initializer_token, add_special_tokens=False)
    init_ids = init_ids[:len(placeholder_token_ids)]

    embedding_dim = text_encoder.get_input_embeddings().weight.shape[1]
    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            if i < len(init_ids):
                text_encoder.get_input_embeddings().weight[token_id] = (
                    text_encoder.get_input_embeddings().weight[init_ids[i]].clone()
                )
            else:
                text_encoder.get_input_embeddings().weight[token_id] = (
                    torch.randn(embedding_dim, device=text_encoder.device, dtype=text_encoder.dtype)
                )

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
