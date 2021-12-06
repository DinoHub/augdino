# AugDino
A module for torch-based audio data augmentations

## Requirements
```bash
torch>=1.8.1
torchaudio>=0.8.1
librosa==0.8.1
julius>=0.2.6
primePy>=1.3
```

## Installation
Install this repository as a package with the following command
```bash
pip install git+https://github.com/DinoHub/augdino.git 
```

## Supported Augmentations

### At a Glance

#### Wave Augmentations
- AddBackgroundNoise
- AddColoredNoise
- ClipDistortion
- Gain
- LowPassFilter
- PitchShift*
- PolarityInversion
- Reverberation
- ReverseOverlay
- Shift

#### Spectrogram Augmentations
- FreqMask
- TimeMask
- TimeStretch

#### Common Transformations
- ComplexSpectrogram**
- Transpose
- Squeeze
- Unsqueeze
- MagnitudeConvert
- LogMelTransform
- RealTransform

*PitchShift is rather resource intensive.

**TimeStretch requires spectrogram as a complex tensor. (torchaudio v0.10's Spectrogram transformation outputs complex tensor by default, otherwise use Augdino's ComplexSpectrogram transformation)

## How to Use

### Single Transformation Example
```python
import torch
from augdino.wave import PitchShift

pitch_shift = PitchShift(min_transpose_semitones=-4.0, max_transpose_smeitones=4.0, sample_rate=16000, p=1.0)

sample_16k = torch.rand([1,16000])
sample_16k_pitched = pitch_shift(sample_16k)
```

### Multiple Transformation Example
```python
import torch
from augdino.utils import Compose
from augdino.wave import AddBackgroundNoise, ClipDistortion, Gain

# add background noise from specific folder, at a 0.5 probability
bgnoise = AddBackgroundNoise(
            local_path='/datasets/noise_folder',
            min_snr_in_db=3.0,
            max_snr_in_db=30.0,
            sample_rate=16000,
            p=0.5)

# distort clip at a 0.5 probability
distort = ClipDistortion(
            min_percent_threshold=0,
            max_percent_threshold=15,
            p=0.5)

# add gain at 0.5 probability
gain = Gain(min_gain_db:-12,
            max_gain_db: 12,
            p=0.5)
           
composed = Compose([bgnoise, distort, gain])

sample_16k = torch.rand([1,16000])
sample_16k_transformed = composed(sample_16k)
```

### Using compose_transformations
The package also provides an utility method to compose transformations via dictionaries/dictionary-like methods. (Refer to the `__init__.py` file in the main folder for all the required keys.)

```python
import torch
from augdino import compose_transformations

wave_transforms = {
  'background_noise': {
    'local_path': '/datasets/noise_folder',
    'min_snr_in_db': 3.0,
    'max_snr_in_db: 30.0,
    'sample_rate': 16000,
    'p': 0.5,
  },
}
spec_transforms = {
  'spectrogram': {
    'n_fft': 512,
    'win_length': 400,
    'hop_length': 160,
  },
  'magnitude_convert': {
    'power': 2.0,
  },
  'freq_mask': {
    'freq_mask_param': 10,
    'num_freq_masks': 2,
    'p': 0.5,
  }
}
composed = compose_transformations(wave_transforms, spec_transforms)

sample_16k = torch.rand([1,16000])
sample_16k_transformed = composed(sample_16k)
```
