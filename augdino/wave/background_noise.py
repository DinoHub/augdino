import random
from pathlib import Path
from typing import Union, List

import torch

from ..utils.io import Audio
from ..utils.file import find_audio_files

def calculate_rms(samples):
    """
    Calculates the root mean square.
    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    return torch.sqrt(torch.mean(torch.square(samples)))

class AddBackgroundNoise:
    def __init__(
        self,
        local_path: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        p: float = 0.5,
        sample_rate: int = 16000,
        precache: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param local_path: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximium SNR in dB.
        :param p:
        :param sample_rate:
        """

        if isinstance(local_path, (list, tuple, set)):
            # TODO: check that one can read audio files
            self.local_path = list(local_path)
        else:
            self.local_path = find_audio_files(local_path)

        self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.local_path) == 0:
            raise Exception("There are no supported audio files found.")

        self.precache = precache
        if self.precache:
            print('Caching all background noise audio files')
            self.bg_noises = [self.audio(p) for p in self.local_path]

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.p = p
        self.transform_parameters = {}

    def random_background(self, target_num_samples: int) -> torch.Tensor:
        pieces = []

        # TODO: support repeat short samples instead of concatenating from different files

        missing_num_samples = target_num_samples
        if self.precache:
            while missing_num_samples > 0:
                background_samples = random.choice(self.bg_noises)
                background_num_samples = background_samples.shape[-1]

                if background_num_samples > missing_num_samples:
                    sample_offset = random.randint(0, background_num_samples - missing_num_samples)
                    background_samples = background_samples[:,sample_offset:sample_offset + missing_num_samples]
                    missing_num_samples = 0
                else:
                    missing_num_samples -= background_num_samples

                pieces.append(background_samples)

        else:
            while missing_num_samples > 0:
                background_path = random.choice(self.local_path)
                background_num_samples = self.audio.get_num_samples(background_path)

                if background_num_samples > missing_num_samples:
                    sample_offset = random.randint(
                        0, background_num_samples - missing_num_samples
                    )
                    num_samples = missing_num_samples
                    background_samples = self.audio(
                        background_path, sample_offset=sample_offset, num_samples=num_samples
                    )
                    missing_num_samples = 0
                else:
                    background_samples = self.audio(background_path)
                    missing_num_samples -= background_num_samples

                pieces.append(background_samples)

        #  the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
        #  the outer call to rms_normalize ensures that the resulting background has an RMS of 1
        #  (this simplifies "apply_transform" logic)
        return self.audio.rms_normalize(
            torch.cat([self.audio.rms_normalize(piece) for piece in pieces], dim=1)
        )

    def randomize_parameters(self, selected_samples: torch.Tensor, sample_rate: int = None):
        """
        :params selected_samples: (num_channels, num_samples)
        """

        _, num_samples = selected_samples.shape

        self.transform_parameters["background"] = self.random_background(num_samples)
        self.transform_parameters["snr_in_db"] = torch.rand(1) * (self.max_snr_in_db - self.min_snr_in_db) + self.min_snr_in_db 

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:

        if random.random() <= self.p:

            self.randomize_parameters(selected_samples=waveform)

            background = self.transform_parameters["background"].to(waveform.device)
            snr_in_db = self.transform_parameters["snr_in_db"].to(waveform.device)

            background_rms = calculate_rms(waveform) / (
                10 ** (snr_in_db / 20)
            )

            waveform = waveform + background_rms * background

        return waveform