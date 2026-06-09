from typing import Optional, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput




class SDPipelineInferenceFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

    @property
    def init_noise_sigma(self):
        return 1

    def scale_model_input(self, x, t):
        return x

    def add_noise(self, latents, noise, timesteps):
        return self.scale_noise(latents, timesteps, noise)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        **kwargs,
    ):
        if self.config.use_dynamic_shifting and 'mu' not in kwargs:
            kwargs['mu'] = self.shift
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        #print(f'timesteps ({num_inference_steps}):', self.timesteps)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        **kwargs,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
        """
        Override of the parent Euler step that additionally computes and exposes
        ``pred_original_sample`` — the scheduler's estimate of the fully-denoised
        latent at the current step.

        In rectified-flow (flow-matching) the model predicts velocity ``v``:
            ``x_t = (1 - t) * x_0  +  t * ε``
            ``v   = ε - x_0``

        The denoised estimate follows directly:
            ``x_0_pred = x_t  −  σ_t * v``

        The result is stored in two places so callers can always get it:

        * ``self._last_pred_original_sample`` — direct attribute access from the
          worker's step callback without relying on the diffusers pipeline to
          thread it through ``callback_kwargs``.
        * ``output["pred_original_sample"]`` — injected into the
          ``FlowMatchEulerDiscreteSchedulerOutput`` (which is a ``BaseOutput`` /
          ``OrderedDict``), so diffusers' standard
          ``callback_on_step_end_tensor_inputs`` machinery can pick it up if the
          pipeline supports it.
        """
        # Initialise the step index the same way the parent does, so that we can
        # read the current sigma *before* the parent increments _step_index.
        if self.step_index is None:
            self._init_step_index(timestep)

        # Capture sigma at the *current* step (parent will increment after its step).
        sigma = self.sigmas[self.step_index]

        # Rectified-flow denoised estimate: x_0_pred = x_t − σ * v_pred
        # Cast to float32 to match the precision the parent uses internally.
        pred_original_sample = sample.to(torch.float32) - sigma * model_output.to(torch.float32)
        pred_original_sample = pred_original_sample.to(model_output.dtype)

        # Run the actual Euler update (also increments _step_index).
        output = super().step(model_output, timestep, sample, **kwargs)

        # ── Expose pred_original_sample ──────────────────────────────────────
        # 1) Instance attribute — accessible from the worker callback directly.
        self._last_pred_original_sample: torch.Tensor = pred_original_sample

        # 2) Inject into the BaseOutput / OrderedDict so the standard diffusers
        #    callback_on_step_end_tensor_inputs mechanism can forward it.
        if isinstance(output, FlowMatchEulerDiscreteSchedulerOutput):
            try:
                output["pred_original_sample"] = pred_original_sample
            except Exception:
                pass  # defensive: never let preview injection break generation
        # ─────────────────────────────────────────────────────────────────────

        return output

