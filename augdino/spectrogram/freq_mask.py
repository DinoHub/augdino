from torchaudio.transforms import FrequencyMasking
import torch
import random
from typing import Optional

class CustomFrequencyMasking(torch.nn.Module):

    def __init__(
            self,
            freq_mask_param: Optional[int] = 70,
            num_freq_masks: Optional[int] = 1,
            p: Optional[float] = 0.5) -> None:
    
        super(CustomFrequencyMasking, self).__init__()

        self.mask = FrequencyMasking(freq_mask_param)
        self.num_freq_masks = num_freq_masks
        self.p = p

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if random.random() <= self.p:
            for _ in range(self.num_freq_masks):
                spectrogram = self.mask(spectrogram)
        return spectrogram