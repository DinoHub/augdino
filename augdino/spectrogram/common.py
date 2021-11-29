import torch
from torch import Tensor
from typing import Optional, List, Dict, Any
from torchaudio.transforms import MelSpectrogram
from ..utils.compose import Compose

class LogMelTransform(torch.nn.Module):

    def __init__(self, log_offset: Optional[float] = 1e-7) -> None:
        
        super(LogMelTransform, self).__init__()
        self.log_offset = log_offset

    def forward(self, melspectrogram: Tensor) -> Tensor:
        return torch.log(melspectrogram + self.log_offset)

class Transpose(torch.nn.Module):
    
    def __init__(self, dim=None) -> None:
        super(Transpose, self).__init__()
        self.dim = dim

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.T

class Unsqueeze(torch.nn.Module):

    def __init__(self, dim) -> None:
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, tensor: Tensor) -> Tensor:
        return torch.unsqueeze(tensor, self.dim)

class Squeeze(torch.nn.Module):
    def __init__(self, dim) -> None:
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, tensor: Tensor) -> Tensor:
        return torch.squeeze(tensor, self.dim)

class CustomMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        f_max: int, f_min: int, n_fft: int, n_mels: int, win_length: int, hop_length: int, sample_rate: int = 16000,
        squeeze: bool = False, logmel: bool = False, transpose_at_end: bool = True
        ) -> None:

        melspec = MelSpectrogram(
            sample_rate=sample_rate, 
            f_max=f_max, f_min=f_min, 
            n_fft=n_fft, n_mels=n_mels,
            win_length=win_length, hop_length=hop_length
            )

        compose_list = []
        if squeeze:
            compose_list.append(Squeeze(dim=0))
        compose_list.append(melspec)
        if logmel:
            compose_list.append(LogMelTransform(log_offset=1e-7))
        if transpose_at_end:
            compose_list.append(Transpose())

        self.composed = Compose(compose_list)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.composed(waveform)