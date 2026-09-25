######## 3rd step ########

'''
this will detect the voice activity 
using pretrainned model 

1st function will give the timestamps 

speech_timestamps = [
    {"start": 32000, "end": 80000},
    {"start": 112000, "end": 160000}
]

and the 2nd function will extract the speech segments


speech segment is simething like -> [
    array([0.012, 0.034, -0.021, ...]),  # first speech segment
    array([0.008, -0.015, 0.027, ...])   # second speech segment
]

'''

import torch
from silero_vad import load_silero_vad, get_speech_timestamps


# Load the pretrained VAD model
vad_model = load_silero_vad()


def detect_speech(audio, sample_rate):
    # Convert NumPy waveform to PyTorch tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32) # as silero_vad expect tensor so we convert the audio numpy array into tensor

    # Detect speech regions
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate
    )

    return speech_timestamps


def extract_speech(audio, speech_timestamps):
    speech_segments = []

    for segment in speech_timestamps:
        start = segment["start"]
        end = segment["end"]

        speech = audio[start:end]
        speech_segments.append(speech)

    return speech_segments