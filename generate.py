import torchaudio
from torchaudio.io import CodecConfig
import librosa
import os
import numpy as np

IRMAS_PATH = './data/IRMAS'
LIBRISPEECH_PATH = './data/LibriSpeech'

def generate_mp3(bit_rate_kbps=32000):
    for dataset_path in [IRMAS_PATH, LIBRISPEECH_PATH]:
        DATASET_WAV_PATH = os.path.join(dataset_path, 'wav')
        DATASET_MP3_PATH = os.path.join(dataset_path, 'mp3')

        for fol_type in os.listdir(DATASET_WAV_PATH):
            fol = os.path.join(DATASET_WAV_PATH, fol_type)

            for file_path in os.listdir(fol):
                wav_file_path = os.path.join(fol, file_path)
                mp3_file_path = os.path.join(DATASET_MP3_PATH, fol_type, file_path)
                mp3_file_path = os.path.splitext(mp3_file_path)[0] + ".mp3"

                waveform, sr = torchaudio.load(wav_file_path)
                torchaudio.save(mp3_file_path, waveform, sr, format="mp3", compression=CodecConfig(bit_rate=bit_rate_kbps))

def calculate_averange_length():
    avg_lengths = []
    for dataset_path in [IRMAS_PATH, LIBRISPEECH_PATH]:
        total = 0.0
        count = 0.0
        DATASET_WAV_PATH = os.path.join(dataset_path, 'wav')

        for fol_type in os.listdir(DATASET_WAV_PATH):
            fol = os.path.join(DATASET_WAV_PATH, fol_type)

            for file_path in os.listdir(fol):
                wav_file_path = os.path.join(fol, file_path)

                x_wav, sr_wav = librosa.load(wav_file_path, sr=None)

                total += x_wav.shape[0]
                count += 1

        avg_lengths.append(total / count)

    return avg_lengths

# 200055.30551118212
# 141956.9174014696


def adpcm_encode(x, num_bits=2):
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

def adpcm_decode(indices, num_bits=2):
   
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