from augdino.wave import *
import torch
import time

def run_time_trial(augment, tensor, num_times):

    start = time.perf_counter()
    for i in range(num_times):
        _ = augment(tensor)
    end = time.perf_counter()

    return end - start

def test_background_noise(num_times=1000):

    augment = AddBackgroundNoise(local_path = '/home/daniel/datasets/processed/wham_noise/tr', precache=False, p = 1.0,)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_background_noise_precache(num_times=1000):

    augment = AddBackgroundNoise(local_path = '/home/daniel/datasets/processed/wham_noise/tr', precache=True, p = 1.0,)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_reverb(num_times=1000):

    augment = Reverb(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_reverb_small(num_times=1000):

    augment = Reverb(min_reverb=10, max_reverb=20, min_damp_factor=10, max_damp_factor=20, min_room_size=10, max_room_size=20, p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_clip_distortion(num_times=1000):

    augment = ClipDistortion(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_colored_noise(num_times=1000):

    augment = AddColoredNoise(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_gain(num_times=1000):

    augment = Gain(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_low_pass(num_times=1000):

    augment = LowPassFilter(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_polarity_inversion(num_times=1000):

    augment = PolarityInversion(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_pitch_shift(num_times=1000):

    augment = PitchShift(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_shift(num_times=1000):

    augment = Shift(p=1.0)
    tensor = torch.rand([1, 16000])

    return run_time_trial(augment, tensor, num_times)

def test_all(num_times=5000):

    print(f'each augmentation is applied {num_times} times')
    print('-------')

    # print(f'AddBackgroundNoise:             {round(test_background_noise(num_times), 3)}(s)')
    # print(f'AddBackgroundNoise Precached:   {round(test_background_noise_precache(5000), 3)}(s)')
    # print(f'Reverb:                         {round(test_reverb(num_times), 3)}(s)')
    print(f'ClipDistortion:                 {round(test_clip_distortion(num_times), 3)}(s)')
    # print(f'AddColoredNoise:                {round(test_colored_noise(num_times), 3)}(s)')
    # print(f'Gain:                           {round(test_gain(num_times), 3)}(s)')
    # print(f'LowPassFilter:                  {round(test_low_pass(num_times), 3)}(s)')
    # print(f'PolarityInversion:              {round(test_polarity_inversion(num_times), 3)}(s)')
    # print(f'PitchShift:                     {round(test_pitch_shift(num_times), 3)}(s)')
    # print(f'Shift:                          {round(test_shift(num_times), 3)}(s)')

if __name__ == '__main__':
    test_all()