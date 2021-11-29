from torchaudio.transforms import TimeStretch
import torch
import random

class CustomTimeStretch(torch.nn.Module):

    def __init__(
            self,
            fixed_rates: list,
            p: float = 0.5) -> None:
    
        super(CustomTimeStretch, self).__init__()

        self.fixed_rates = fixed_rates
        self.p = p

        self.stretch = TimeStretch()
        self.transform_parameters = {}

    def randomize_parameters(self):
        self.transform_parameters['stretch_rate'] = random.choice(self.fixed_rates)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:

        if random.random() <= 0.5:
            self.randomize_parameters()
            spectrogram = self.stretch(spectrogram, self.transform_parameters['stretch_rate'])

        return spectrogram
