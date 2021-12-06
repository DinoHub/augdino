from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import chain, count, islice, repeat
from typing import Union, Callable, List, Optional
from torch.nn.functional import pad
import torch
import torchaudio.transforms as T
from primePy import primes
from math import log2
import warnings
import random

warnings.simplefilter("ignore")

# https://stackoverflow.com/a/46623112/9325832
def _combinations_without_repetition(r, iterable=None, values=None, counts=None):
    if iterable:
        values, counts = zip(*Counter(iterable).items())

    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), f(count(j), counts[j:])):
            indices[i] = j


def get_fast_shifts(
    sample_rate: int,
    condition: Optional[Callable] = lambda x: x >= 0.5 and x <= 2 and x != 1,
) -> List[Fraction]:
    """
    Search for pitch-shift targets that can be computed quickly for a given sample rate.
    Parameters
    ----------
    sample_rate: int
        The sample rate of an audio clip.
    condition: Callable [optional]
        A function to validate fast shift ratios.
        Default is `lambda x: x >= 0.5 and x <= 2 and x != 1` (between -1 and +1 octaves).
    Returns
    -------
    output: List[Fraction]
        A list of fast pitch-shift target ratios
    """
    fast_shifts = set()
    factors = primes.factors(sample_rate)
    products = []
    for i in range(1, len(factors) + 1):
        products.extend(
            [
                reduce(lambda x, y: x * y, x)
                for x in _combinations_without_repetition(i, iterable=factors)
            ]
        )
    for i in products:
        for j in products:
            f = Fraction(i, j)
            if condition(f):
                fast_shifts.add(f)
    return list(fast_shifts)


def semitones_to_ratio(semitones: float) -> Fraction:
    """
    Convert semitonal shifts into ratios.
    Parameters
    ----------
    semitones: float
        The number of semitones for a desired shift.
    Returns
    -------
    output: Fraction
        A Fraction indicating a pitch shift ratio
    """
    return Fraction(2.0 ** (semitones / 12.0))


def ratio_to_semitones(ratio: Fraction) -> float:
    """
    Convert rational shifts to semitones.
    Parameters
    ----------
    ratio: Fraction
        The ratio for a desired shift.
    Returns
    -------
    output: float
        The magnitude of a pitch shift in semitones
    """
    return float(12.0 * log2(ratio))

class PitchShift(torch.nn.Module):
    def __init__(
        self, 
        min_transpose_semitones: float = -4.0,
        max_transpose_semitones: float = 4.0,
        sample_rate: int = 16000,
        p: float = 0.5
    ) -> None:
        """
        Shift the pitch of a batch of waveforms by a given amount.
        Parameters
        ----------
        shift: float OR Fraction
            `float`: Amount to pitch-shift in # of bins. (1 bin == 1 semitone if `bins_per_octave` == 12)
            `Fraction`: A `fractions.Fraction` object indicating the shift ratio. Usually an element in `get_fast_shifts()`.
        sample_rate: int
            The sample rate of the input audio clips.
        bins_per_octave: int [optional]
            Number of bins per octave. Default is 12.
        n_fft: int [optional]
            Size of FFT. Default is `sample_rate // 64`.
        hop_length: int [optional]
            Size of hop length. Default is `n_fft // 32`.
        Returns
        -------
        output: torch.Tensor [shape=(batch_size, channels, samples)]
            The pitch-shifted batch of audio clips
        """

        self.sample_rate = sample_rate
        self.n_fft = sample_rate // 64
        self.hop_length = self.n_fft // 32
        self.p = p

        fast_shifts = get_fast_shifts(
            sample_rate,
            lambda x: x >= semitones_to_ratio(min_transpose_semitones)
            and x <= semitones_to_ratio(max_transpose_semitones)
            and x != 1,
        )
        self.num_shifts = len(fast_shifts)
        self.resamplers = [T.Resample(self.sample_rate, int(self.sample_rate / shift)) for shift in fast_shifts]
        self.stretchers = [T.TimeStretch(fixed_rate=float(1/shift), n_freq=self.n_fft//2+1, hop_length=self.hop_length) for shift in fast_shifts]

        self.transform_parameters = {}

    def randomize_params(self) -> None:

        self.transform_parameters['idx'] = random.choice(range(self.num_shifts))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:

        output = waveform

        if random.random() <= self.p:
            self.randomize_params()

            trf_idx = self.transform_parameters['idx']
            resampler = self.resamplers[trf_idx]
            stretcher = self.stretchers[trf_idx]

            output = torch.stft(output, self.n_fft, self.hop_length)[None, ...]
            output = stretcher(output)
            output = torch.istft(output[0], self.n_fft, self.hop_length)
            output = resampler(output)

            if output.shape[1] >= waveform.shape[1]:
                output = output[:, :waveform.shape[1]]
            else:
                output = pad(output, pad=(0, waveform.shape[1] - output.shape[1], 0, 0))

        return output