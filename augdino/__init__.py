from .wave.background_noise import AddBackgroundNoise
from .wave.clip_distortion import ClipDistortion
from .wave.colored_noise import AddColoredNoise
from .wave.gain import Gain
from .wave.low_pass import LowPassFilter
from .wave.polarity_inversion import PolarityInversion
from .wave.reverse_overlay import ReverseOverlay
from .wave.shift import Shift
from .wave.reverb import Reverb
from .wave.pitch_shift import PitchShift

from .spectrogram.freq_mask import CustomFrequencyMasking as FreqMask
from .spectrogram.time_mask import CustomTimeMasking as TimeMask
from .spectrogram.time_stretch import CustomTimeStretch as TimeStretch

from .common.common import CustomMelSpectrogram
from .utils.compose import Compose

from typing import List, Dict, Any, Optional

WAVE_AUGMENTATIONS = {
    'background_noise': AddBackgroundNoise,
    'clip_distortion': ClipDistortion,
    'gain': Gain,
    'polarity_inversion': PolarityInversion,
    'low_pass': LowPassFilter,
    'colored_noise': AddColoredNoise,          
    'reverse_overlay': ReverseOverlay,
    'shift': Shift,
    'reverb': Reverb,
    'pitch_shift': PitchShift
}

SPEC_AUGMENTATIONS = {
    'freq_mask': FreqMask,
    'time_mask': TimeMask,
    'time_stretch': TimeStretch,
}

def compose_transformations(
    wave_augment_cfg: List[Dict[str, Any]],
    mel_transform_cfg: Optional[Dict[str, Any]] = None,
    spec_augment_cfg: Optional[List[Dict[str, Any]]] = None,
    ) -> Compose:
    
    compose_list = []

    for aug_name in wave_augment_cfg.keys():
        transformation = WAVE_AUGMENTATIONS[aug_name](**wave_augment_cfg[aug_name])
        compose_list.append(transformation)

    if mel_transform_cfg is not None:
        compose_list.append(CustomMelSpectrogram(**mel_transform_cfg))

    if spec_augment_cfg is not None:
        assert mel_transform_cfg is not None, 'Spectrogram augmentation(s) is being added when no melspectrogram configs are in place.'
        for aug_name in spec_augment_cfg.keys():
            transformation = SPEC_AUGMENTATIONS[aug_name](**spec_augment_cfg[aug_name])
            compose_list.append(transformation)

    return Compose(compose_list)