import torch
from torch import Tensor
from typing import Optional

class LogMelTransform(torch.nn.Module):

    def __init__(self, log_offset: Optional[float] = 1e-7) -> None:
        
        super(LogMelTransform, self).__init__()
        self.log_offset = log_offset

    def __call__(self, melspectrogram: Tensor) -> Tensor:
        return torch.log(melspectrogram + self.log_offset)

class Transpose(torch.nn.Module):
    
    def __init__(self, dim=None) -> None:
        super(Transpose, self).__init__()
        self.dim = dim

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.T

class Unsqueeze(torch.nn.Module):

    def __init__(self, dim) -> None:
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.unsqueeze(tensor, self.dim)

class Squeeze(torch.nn.Module):
    def __init__(self, dim) -> None:
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.squeeze(tensor, self.dim)

class MagnitudeConvert(torch.nn.Module):
    def __init__(self, power: float) -> None:
        super(MagnitudeConvert, self).__init__()
        self.power = power

    def __call__(self, tensor: Tensor) -> Tensor:
        return tensor.abs().pow(self.power)

class RealTransform(torch.nn.Module):
    def __init__(self, dim) -> None:
        super(RealTransform, self).__init__()
        self.dim = dim

    def __call__(self, tensor: Tensor) -> Tensor:
        return torch.view_as_real(tensor)

class ComplexSpectrogram(torch.nn.Module):

    def __init__(
        self, 
        n_fft: int, win_length: int, hop_length: int, 
        normalized: bool = True,        
        ) -> None:

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized = normalized
        
        window = torch.hann_window(self.win_length)
        self.register_buffer('window', window)

    def __call__(self, waveform: Tensor) -> Tensor:

        # pack batch
        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])

        spec_f = torch.stft(
            input=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=False,
            center=True,
            pad_mode='reflect',
            onesided=True,
            return_complex=True
        )
        # unpack batch
        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

        if self.normalized:
            spec_f /= self.window.pow(2.).sum().sqrt()
        return spec_f