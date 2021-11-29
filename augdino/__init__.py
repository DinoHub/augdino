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
from .spectrogram.common import *

from .utils.compose import Compose

from torchaudio.transforms import Spectrogram, MelScale
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
    'spectrogram': Spectrogram,
    'mel_scale': MelScale,
    'freq_mask': FreqMask,
    'time_mask': TimeMask,
    'time_stretch': TimeStretch,
    'custom_melspec': CustomMelSpectrogram,
    # common transforms
    'transpose': Transpose,
    'squeeze': Squeeze,
    'unsqueeze': Unsqueeze,
    'log_transform': LogMelTransform,
}

def compose_transformations(
    wave_augment_cfg: Optional[List[Dict[str, Any]]] = None,
    spec_augment_cfg: Optional[List[Dict[str, Any]]] = None,
    ) -> Compose:
    
    compose_list = []
    if wave_augment_cfg is not None:
        for aug_name in wave_augment_cfg.keys():
            transformation = WAVE_AUGMENTATIONS[aug_name](**wave_augment_cfg[aug_name])
            compose_list.append(transformation)

    if spec_augment_cfg is not None:
        for idx, aug_name in enumerate(spec_augment_cfg.keys()):
            if idx == 0: 
                assert aug_name == 'spectrogram', 'To apply spectrogram augmentations, Spectrogram transformation must come first'

            transformation = SPEC_AUGMENTATIONS[aug_name](**spec_augment_cfg[aug_name])
            compose_list.append(transformation)

    return Compose(compose_list)