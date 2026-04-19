import gradio as gr
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

from model import *
from data import spec_to_audio
from generate import adpcm_encode, adpcm_decode
import segmentation_models_pytorch as smp

IRMAS_PATH_CP = './checkpoints/IRMAS_checkpoints'
LIBRISPEECH_PATH_CP = './checkpoints/LibriSpeech_checkpoints'

def init(model_name, audio_type, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    dataset_name = 'LibriSpeech' if audio_type == 'Speech' else 'IRMAS'

    if model_name == 'Autoencoder':
        model = Autoencoder().to(device)
    elif model_name == 'Unet':
        model = Unet().to(device)
    elif model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name="vgg11_bn",
            in_channels=1,
            classes=1
        ).to(device)
    elif model_name == 'UnetAttention':
        model = AttU_Net(img_ch=1, output_ch=1).to(device)
    elif model_name == 'Proposed':
        model = Proposed(img_ch=1, output_ch=1).to(device)

    if dataset_name == 'IRMAS':
        checkpoint = os.path.join(IRMAS_PATH_CP, f'{dataset_name}_{model_name}.pth')
    else:
        checkpoint = os.path.join(LIBRISPEECH_PATH_CP, f'{dataset_name}_{model_name}.pth')

    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    return model, device

def waveform_to_spec(x):
    return librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1022, hop_length=512)))

def spec_to_waveform(spec_db, n_fft=1022, hop_length=512):
    spec = librosa.db_to_amplitude(spec_db)

    audio = librosa.griffinlim(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=256,
    )

    return audio

def compute_metrics(original, enhanced):
    mse = np.mean((original - enhanced)**2)
    snr = 10 * np.log10(np.mean(original**2) / (np.mean((original-enhanced)**2)+1e-8))
    return mse, snr

def plot_waveform(x, sr, title):
    fig, ax = plt.subplots()
    librosa.display.waveshow(x, sr=sr, ax=ax)
    ax.set_title(title)
    return fig

def plot_spec(x, sr, title):
    fig = plt.figure()
    plt.specgram(x, NFFT=1022, Fs=sr, noverlap=512)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    return fig

def change_length(x):
    target_len = 1048064

    if len(x) > target_len:
        return x[:target_len]
    elif len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    return x

def process(audio, model_name, audio_type, bitrate):
    model, device = init(model_name, audio_type, "cuda")

    # sr, original_waveform = audio
    original_waveform, sr = librosa.load(audio, sr=64000)

    # if not np.issubdtype(original_waveform.dtype, np.floating):
    #     original_waveform = original_waveform.astype(np.float32)

    # if original_waveform.ndim > 1:
    #     original_waveform = librosa.to_mono(original_waveform.T)

    original_waveform = change_length(original_waveform)
    original_spec = waveform_to_spec(original_waveform)

    indices = adpcm_encode(original_waveform, num_bits=bitrate)
    degraded_waveform = adpcm_decode(indices, num_bits=bitrate)
    degraded_waveform = change_length(degraded_waveform)
    degraded_spec = waveform_to_spec(degraded_waveform)
    degraded_spec = torch.tensor(degraded_spec).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        enhanced_spec = model(degraded_spec).squeeze(0).squeeze(0).cpu().detach().numpy()
    degraded_spec = degraded_spec.squeeze(0).squeeze(0).cpu().detach().numpy()
    enhanced_waveform = spec_to_waveform(enhanced_spec)
    original_waveform = spec_to_waveform(original_spec)
    degraded_waveform = spec_to_waveform(degraded_spec)
   
    mse, snr = compute_metrics(original_spec, enhanced_spec)
    metrics_text = f"MSE: {mse:.6f} | SNR: {snr:.2f} dB"

    return (
        (sr, original_waveform),
        (sr, degraded_waveform),
        (sr, enhanced_waveform),

        plot_waveform(original_waveform, sr, "Original Waveform"),
        plot_waveform(degraded_waveform, sr, "Degraded Waveform"),
        plot_waveform(enhanced_waveform, sr, "Enhanced Waveform"),

        plot_spec(original_waveform, sr, "Original Spectrogram"),
        plot_spec(degraded_waveform, sr, "Degraded Spectrogram"),
        plot_spec(enhanced_waveform, sr, "Enhanced Spectrogram"),

        metrics_text
    )

with gr.Blocks() as demo:
    gr.Markdown("# Enhancing Compressed Audio Using Neural Post-Processing")

    # audio_input = gr.Audio(label="Input Audio", type="numpy")
    audio_input = gr.Audio(label="Input Audio", type="filepath")

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["Autoencoder", "Unet", "UnetPlusPlus", "UnetAttention", "Proposed"],
            value=None,
            label="Model",
            allow_custom_value=False
        )

        audio_type = gr.Dropdown(
            choices=["Speech", "Music"],
            value=None,
            label="Audio Type"
        )

        bitrate = gr.Slider(1, 8, value=2, step=1, label="ADPCM Bitrate")

    run_btn = gr.Button("Run")

    with gr.Row():
        original_audio = gr.Audio(label="Original")
        degraded_audio = gr.Audio(label="Degraded")
        enhanced_audio = gr.Audio(label="Enhanced")

    with gr.Row():
        wf1 = gr.Plot(label="Original Waveform")
        wf2 = gr.Plot(label="Degraded Waveform")
        wf3 = gr.Plot(label="Enhanced Waveform")

    with gr.Row():
        sp1 = gr.Plot(label="Original Spectrogram")
        sp2 = gr.Plot(label="Degraded Spectrogram")
        sp3 = gr.Plot(label="Enhanced Spectrogram")

    metrics = gr.Textbox(label="Metrics")

    run_btn.click(
        process,
        inputs=[audio_input, model_choice, audio_type, bitrate],
        outputs=[
            original_audio,
            degraded_audio,
            enhanced_audio,
            wf1, wf2, wf3,
            sp1, sp2, sp3,
            metrics
        ]
    )

demo.launch()