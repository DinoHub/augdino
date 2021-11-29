import torch
import julius
import random

def convert_frequencies_to_mels(f: torch.Tensor) -> torch.Tensor:
    """
    Convert f hertz to m mels
    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def convert_mels_to_frequencies(m: torch.Tensor) -> torch.Tensor:
    """
    Convert m mels to f hertz
    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)

class LowPassFilter:
    def __init__(
        self,
        min_cutoff_freq=150,
        max_cutoff_freq=7500,
        p: float = 0.5,
        sample_rate: int = 16000,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param p:
        :param sample_rate:
        """

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq

        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

        self.p = p
        self.sample_rate = sample_rate
        self.transform_parameters = {}

    def randomize_parameters(self):

        # Sample frequencies uniformly in mel space, then convert back to frequency
        
        dist = torch.distributions.Uniform(
            low=convert_frequencies_to_mels(
                torch.tensor(
                    self.min_cutoff_freq,
                    dtype=torch.float32,
                )
            ),
            high=convert_frequencies_to_mels(
                torch.tensor(
                    self.max_cutoff_freq,
                    dtype=torch.float32,
                )
            ),
            validate_args=True,
        )
        
        self.transform_parameters["cutoff_freq"] = convert_mels_to_frequencies(
            dist.sample()
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:

        if random.random() <= self.p:
            self.randomize_parameters()

            cutoffs_as_fraction_of_sample_rate = self.transform_parameters["cutoff_freq"] / self.sample_rate  
            waveform = julius.lowpass_filter(waveform, cutoffs_as_fraction_of_sample_rate)
        
        return waveform

