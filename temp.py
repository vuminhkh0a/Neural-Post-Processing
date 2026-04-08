import os
import librosa
import numpy as np
import soundfile as sf
import sys

def adpcm_encode(x, num_bits=1):
    """
    x: numpy array float32 [-1, 1]
    return: quantized indices (int)
    """
    x = x.astype(np.float32)

    q_levels = 2 ** num_bits
    max_q = q_levels // 2 - 1
    min_q = -q_levels // 2

    step_size = 0.01
    alpha = 1.5
    beta = 0.9

    x_rec = 0.0
    indices = []

    for n in range(len(x)):
        pred = x_rec
        e = x[n] - pred

        q = int(np.round(e / step_size))
        q = np.clip(q, min_q, max_q)

        indices.append(q)

        e_q = q * step_size
        x_rec = pred + e_q

        if abs(q) > (max_q // 2):
            step_size *= alpha
        else:
            step_size *= beta

        step_size = np.clip(step_size, 1e-4, 1.0)

    return np.array(indices, dtype=np.int32)

def adpcm_decode(indices, num_bits=1):
   
    q_levels = 2 ** num_bits
    max_q = q_levels // 2 - 1

    step_size = 0.01
    alpha = 1.5
    beta = 0.9

    x_rec = 0.0
    output = []

    for q in indices:
        pred = x_rec

        e_q = q * step_size
        x_rec = pred + e_q
        output.append(x_rec)

        if abs(q) > (max_q // 2):
            step_size *= alpha
        else:
            step_size *= beta

        step_size = np.clip(step_size, 1e-4, 1.0)

    return np.array(output, dtype=np.float32)

IRMAS_PATH = './data/IRMAS'
LIBRISPEECH_PATH = './data/LibriSpeech'

for dataset_path in [IRMAS_PATH, LIBRISPEECH_PATH]:
        DATASET_WAV_PATH = os.path.join(dataset_path, 'wav')
        DEGRADED_DATASET_WAV_PATH = os.path.join(dataset_path, 'degraded_wav')

        for fol_type in os.listdir(DATASET_WAV_PATH):
            fol = os.path.join(DATASET_WAV_PATH, fol_type)
            degraded_fol = os.path.join(DEGRADED_DATASET_WAV_PATH, fol_type)

            for file_path in os.listdir(fol):
                wav_file_path = os.path.join(fol, file_path)
                degraded_wav_file_path = os.path.join(degraded_fol, file_path)
                
                print(wav_file_path)
                print(degraded_wav_file_path)
                sys.stdout.flush()

                waveform, sr = librosa.load(wav_file_path)
                indices = adpcm_encode(waveform, num_bits=2)
                x_degraded = adpcm_decode(indices, num_bits=2)
                sf.write(data=x_degraded, file=degraded_wav_file_path, samplerate=sr)


# cd ~
# cd KhoaVM
# source npp_env/bin/activate
# cd z_Neural_Post_Processing
# nohup python3 temp.py > log.txt 2>&1 &
# tail -f log.txt


