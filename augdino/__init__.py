from .wave.background_noise import AddBackgroundNoise
from .wave.clip_distortion import ClipDistortion
from .wave.colored_noise import AddColoredNoise
from .wave.gain import Gain
from .wave.low_pass import LowPassFilter
from .wave.polarity_inversion import PolarityInversion
from .wave.reverse_overlay import ReverseOverlay
from .wave.shift import Shift
from .wave.reverb import Reverb

from .spectrogram.freq_mask import CustomFrequencyMasking as FreqMask
from .spectrogram.time_mask import CustomTimeMasking as TimeMask
from .spectrogram.time_stretch import CustomTimeStretch as TimeStretch

from .utils.compose import Compose

from typing import List, Dict, Any

AUGMENTATIONS = {
    # wave
    'background_noise': AddBackgroundNoise,
    'clip_distortion': ClipDistortion,
    'gain': Gain,
    'polarity_inversion': PolarityInversion,
    'low_pass': LowPassFilter,
    'colored_noise': AddColoredNoise,          
    'reverse_overlay': ReverseOverlay,
    'shift': Shift,
    'reverb': Reverb,
    # spectrogram
    'freq_mask': FreqMask,
    'time_mask': TimeMask,
    'time_stretch': TimeStretch,
}

def compose_augmentation(augment_list: List[str], augment_configs: Dict[str, Any]):
    
    compose_list = []

    for aug_name in augment_list:
        transformation = AUGMENTATIONS[aug_name](**augment_configs[aug_name])
        compose_list.append(transformation)

    return Compose(compose_list)