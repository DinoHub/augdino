import torch
import math
import random
from torch.nn.functional import pad
from typing import Optional

def phase_vocoder(
        complex_specgrams: torch.Tensor,
        rate: float,
        phase_advance: torch.Tensor
) -> torch.Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.

    Args:
        complex_specgrams (Tensor):
            Either a real tensor of dimension of `(..., freq, num_frame, complex=2)`
            or a tensor of dimension `(..., freq, num_frame)` with complex dtype.
        rate (float): Speed-up factor
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of `(freq, 1)`

    Returns:
        Tensor:
            Stretched spectrogram. The resulting tensor is of the same dtype as the input
            spectrogram, but the number of frames is changed to ``ceil(num_frame / rate)``.

    Example - With Tensor of complex dtype
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time)
        >>> complex_specgrams = torch.randn(2, freq, 300, dtype=torch.cfloat)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, freq)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231])

    Example - With Tensor of real dtype and extra dimension for complex field
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time, complex=2)
        >>> complex_specgrams = torch.randn(2, freq, 300, 2)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, freq)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231, 2])
    """
    if rate == 1.0:
        return complex_specgrams

    if not complex_specgrams.is_complex():
        warnings.warn(
            "The support for pseudo complex type in `torchaudio.functional.phase_vocoder` and "
            "`torchaudio.transforms.TimeStretch` is now deprecated and will be removed "
            "from 0.11 release."
            "Please migrate to native complex type by converting the input tensor with "
            "`torch.view_as_complex`. "
            "Please refer to https://github.com/pytorch/audio/issues/1337 "
            "for more details about torchaudio's plan to migrate to native complex type."
        )
        if complex_specgrams.size(-1) != 2:
            raise ValueError(
                "complex_specgrams must be either native complex tensors or "
                "real valued tensors with shape (..., 2)")

    is_complex = complex_specgrams.is_complex()

    if not is_complex:
        complex_specgrams = torch.view_as_complex(complex_specgrams)

    # pack batch
    shape = complex_specgrams.size()
    complex_specgrams = complex_specgrams.reshape([-1] + list(shape[-2:]))

    # Figures out the corresponding real dtype, i.e. complex128 -> float64, complex64 -> float32
    # Note torch.real is a view so it does not incur any memory copy.
    real_dtype = torch.real(complex_specgrams).dtype
    time_steps = torch.arange(
        0,
        complex_specgrams.size(-1),
        rate,
        device=complex_specgrams.device,
        dtype=real_dtype)

    alphas = time_steps % 1.0
    phase_0 = complex_specgrams[..., :1].angle()

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 2])

    # (new_bins, freq, 2)
    complex_specgrams_0 = complex_specgrams.index_select(-1, time_steps.long())
    complex_specgrams_1 = complex_specgrams.index_select(-1, (time_steps + 1).long())

    angle_0 = complex_specgrams_0.angle()
    angle_1 = complex_specgrams_1.angle()

    norm_0 = complex_specgrams_0.abs()
    norm_1 = complex_specgrams_1.abs()

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    complex_specgrams_stretch = torch.polar(mag, phase_acc)

    # unpack batch
    complex_specgrams_stretch = complex_specgrams_stretch.reshape(shape[:-2] + complex_specgrams_stretch.shape[1:])

    if not is_complex:
        return torch.view_as_real(complex_specgrams_stretch)
    return complex_specgrams_stretch


class CustomTimeStretch(torch.nn.Module):

    def __init__(
            self,
            fixed_rates: list,
            hop_length: Optional[int] = None,
            n_freq: int = 201,
            p: float = 0.5,
            return_same_shape: bool = True,
            return_real: bool = True) -> None:
    
        super(CustomTimeStretch, self).__init__()

        self.fixed_rates = fixed_rates

        self.n_fft = (n_freq - 1) * 2
        self.hop_length = hop_length if hop_length is not None else self.n_fft // 2
        self.p = p

        self.return_same_shape = return_same_shape
        self.return_real = return_real

        self.register_buffer('phase_advance', torch.linspace(0, math.pi * self.hop_length, n_freq)[..., None])

        self.transform_parameters = {}

    def randomize_parameters(self):
        self.transform_parameters['stretch_rate'] = random.choice(self.fixed_rates)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:

        if random.random() <= self.p:
            self.randomize_parameters()
            input_spec = spectrogram
            rate = self.transform_parameters['stretch_rate']
            spectrogram = phase_vocoder(input_spec, rate, self.phase_advance)

            if self.return_same_shape:
                diff = input_spec.shape[-1] - spectrogram.shape[-1]
                if rate > 1:
                    spectrogram = pad(spectrogram, (0, diff), 'constant', 0)
                else:
                    spectrogram = spectrogram[:,:,:diff]

        if self.return_real:
            return torch.view_as_real(spectrogram)

        return spectrogram
