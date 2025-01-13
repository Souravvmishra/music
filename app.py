from audiocraft.models import MusicGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64
import random
import audiocraft

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    return model

def generate_music_tensors(description, duration: int):
    em_map = {'happy':["Ashtami title rough track_instrumental.wav","Group Song Band Vadya FINAL--2_instrumental.wav","Maimeda Baalelu Title Song Master---1_instrumental.wav"],
              'sad':["Perumala Ballal Master_instrumental.wav","Sanvi song To listen--1_instruemntal.wav"],
              'romantic':["Love song asthami for Lyrics_instrumental.wav","Sanvi song To listen--1_instrumental.wav"],
              'angry':"Godduba Chend Master_instrumental.wav"
              }
    description = description.lower()
    if "happy" in description:
        file_path = random.choice(em_map["happy"])
        melody_waveform, sr = torchaudio.load(file_path)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(1, 1, 1)
    elif "sad" in description:
        file_path = random.choice(em_map["sad"])
        melody_waveform, sr = torchaudio.load(file_path)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(1, 1, 1)
    
    elif "romantic" in description:
        file_path = random.choice(em_map["romantic"])
        melody_waveform, sr = torchaudio.load(file_path)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(1, 1, 1)
    
    elif "angry" in description:
        file_path = em_map["angry"]
        melody_waveform, sr = torchaudio.load(file_path)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(1, 1, 1)
    
    else:
        file_path = random.choice(em_map["happy"])
        melody_waveform, sr = torchaudio.load(file_path)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(1, 1, 1)

    model = load_model()
    print("Description: ", description)
    print("Duration: ", duration)

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate_with_chroma(
        descriptions=[str(description)+f" similar to {melody_waveform}"],
        progress=True,
        return_tokens=True,
        melody_wavs=melody_waveform,
        melody_sample_rate=sr
    )

    return output[0]


def save_audio(samples: torch.Tensor):
    """Renders an audio player for the given audio samples and saves them to a local directory.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
        save_path (str): path to the directory where audio should be saved.
    """

    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon= "musical_note",
    page_title= "Music Gen"
)

def main():

    st.title("Text to Music Generator🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using Music Gen Small model.")

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if text_area and time_slider:
        # st.json({
        #     'Your Description': text_area,
        #     'Selected Time Duration (in Seconds)': time_slider
        # })

        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        print("Music Tensors: ", music_tensors)
        save_music_file = save_audio(music_tensors)
        audio_filepath = f'audio_output/audio_{text_area}.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    

