# How to enable `pred_original_sample` previews

By default the live-latent preview broadcasts the **current noisy latent** at
each step.  This is fast and always available, but it looks very noisy in the
first half of generation.

Many schedulers also compute an estimate of what the **fully-denoised latent**
will look like at each step (`pred_original_sample`, also called
`pred_x0`).  Using this estimate instead produces a much cleaner preview from
around step 10 onward — the image looks recognisable far earlier.

---

## Enabling it (one line change)

Open `backend/pipeline/worker.py` and change the flag near the top of the
file:

```python
# before (default — disabled)
USE_PRED_ORIGINAL_SAMPLE: bool = False

# after (enabled)
USE_PRED_ORIGINAL_SAMPLE: bool = True
```

Restart the server.  No other changes are needed.

---

## Which schedulers support it?

The flag is safe to enable regardless of the active sampler.  The step
callback falls back to the noisy latent automatically when
`pred_original_sample` is not available.

| Scheduler | `pred_original_sample` available? |
|---|---|
| `EulerDiscreteScheduler` | ✅ yes |
| `EulerAncestralDiscreteScheduler` | ✅ yes |
| `DPMSolverMultistepScheduler` | ✅ yes (from step 2+) |
| `DPMSolverSDEScheduler` | ✅ yes |
| `KDPM2AncestralDiscreteScheduler` | ✅ yes |
| `DDIMScheduler` | ✅ yes |
| `LMSDiscreteScheduler` | ❌ no — falls back to noisy latent |
| `SDPipelineInferenceFlowMatchEulerDiscreteScheduler` (euler_fm_dynshift) | ✅ yes — computed in our `step()` override |

---

## How it works under the hood

The diffusers `callback_on_step_end` mechanism passes a dict of tensors
(`callback_kwargs`) into the step callback.  The keys are controlled by
`callback_on_step_end_tensor_inputs`, but diffusers validates that list against
a **pipeline-defined whitelist** (e.g. `['latents', 'prompt_embeds', ...]`).
`pred_original_sample` is never in that whitelist, so it cannot be requested
there.

Instead, `pred_original_sample` is retrieved directly from the scheduler
instance attribute `pipe.scheduler._last_pred_original_sample`, set at every
step by our `SDPipelineInferenceFlowMatchEulerDiscreteScheduler.step()` override
(and available to any future scheduler override that follows the same pattern).

The step callback reads it as:

```python
if USE_PRED_ORIGINAL_SAMPLE:
    sample = getattr(pipe.scheduler, "_last_pred_original_sample", None)
    if sample is None:
        sample = callback_kwargs.get("latents")   # graceful fallback
else:
    sample = callback_kwargs.get("latents")
```

The selected tensor is passed to `latents_to_preview_image`, which applies the
appropriate SD 1.x or SDXL RGB-factor matrix to produce a coloured JPEG
without invoking the VAE.

---

## Mathematical background

### Flow-matching (euler_fm_dynshift)

`SDPipelineInferenceFlowMatchEulerDiscreteScheduler` overrides `step()` to
expose `pred_original_sample`.  In rectified flow the forward process is a
straight-line interpolation:

```
x_t = (1 − t) · x₀  +  t · ε        (σ = t)
```

The model predicts the velocity `v = ε − x₀`, so the denoised estimate is:

```
x₀_pred = x_t − σ · v_pred
```

This is computed before the Euler update increments the step index, stored in
both `self._last_pred_original_sample` and injected into the
`FlowMatchEulerDiscreteSchedulerOutput` dict so either retrieval path works.

### Other schedulers that don't natively expose pred_original_sample

For `LMSDiscreteScheduler` you would need Tweedie's formula:

```
x₀_pred = (xₜ − σₜ · noise_pred) / αₜ
```

where `αₜ` and `σₜ` are the signal and noise scales at timestep `t`.  This
requires hooking into the scheduler's internal state and is left as a future
enhancement.

