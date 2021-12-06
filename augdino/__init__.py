from .wave import *
from .spectrogram import *
from .utils import Compose

from torchaudio.transforms import MelScale
from typing import Dict, Any, Optional

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
    'spectrogram': ComplexSpectrogram,
    'mel_scale': MelScale,
    'freq_mask': FreqMask,
    'time_mask': TimeMask,
    'time_stretch': TimeStretch,
    # common transforms
    'transpose': Transpose,
    'squeeze': Squeeze,
    'unsqueeze': Unsqueeze,
    'magnitude_convert': MagnitudeConvert,
    'log_transform': LogMelTransform,
    'real_transform': RealTransform
}

def compose_transformations(
    wave_augment_cfg: Optional[Dict[Dict[str, Any]]] = None,
    spec_augment_cfg: Optional[Dict[Dict[str, Any]]] = None,
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