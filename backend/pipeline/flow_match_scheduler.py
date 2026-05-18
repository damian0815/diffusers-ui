from typing import Optional, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

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
        num_inference_steps: Optional[int]=None,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        if self.config.use_dynamic_shifting and 'mu' not in kwargs:
            kwargs['mu'] = self.shift
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        #print(f'timesteps ({num_inference_steps}):', self.timesteps)
