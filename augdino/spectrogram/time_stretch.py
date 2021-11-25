from torchaudio.transforms import TimeStretch
import torch
import random

class CustomTimeStretch(torch.nn.Module):

    def __init__(
            self,
            fixed_rates: list,
            p: float = 0.5) -> None:
    
        super(CustomTimeStretch, self).__init__()

        self.time_stretches = [TimeStretch(fixed_rate=x) for x in fixed_rates]
        self.p = p
        self.transform_parameters = {}

    def randomize_parameters(self):
        self.transform_parameters['stretch'] = random.choice(self.time_stretches)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:

        if random.random() <= 0.5:
            self.randomize_parameters()
            stretch = self.transform_parameters['stretch']
            spectrogram = stretch(spectrogram)

        return spectrogram
