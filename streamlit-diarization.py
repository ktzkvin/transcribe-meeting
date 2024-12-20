import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
from pyannote.audio import Pipeline
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import time

# Hugging Face Token
TOKEN_FILE = ".token"

def get_token_from_file():
    """
    Récupérer le token Hugging Face depuis le fichier .token
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file:
            return file.read().strip()
    return None

def save_token_to_file(token):
    """
    Sauvegarder le token Hugging Face dans le fichier .token
    """
    with open(TOKEN_FILE, "w") as file:
        file.write(token)

@st.cache_resource
def load_diarization_pipeline(token):
    """
    Charger le pipeline de diarisation de PyAnnote avec mise en cache.
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline

@st.cache_resource
def load_whisper_pipeline():
    """
    Charger le modèle Whisper avec mise en cache.
    """
    model_id = "openai/whisper-large-v3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

# Récupérer le token
hf_token = get_token_from_file()

if hf_token:
    st.success("Token loaded from .token file.")
    login(token=hf_token)
    st.session_state.hf_token = hf_token
else:
    manual_token = st.text_input("Enter your Hugging Face Token:", type="password")
    if st.button("Login"):
        if manual_token:
            login(token=manual_token)
            save_token_to_file(manual_token)
            st.session_state.hf_token = manual_token
            st.success("Successfully logged in and token saved!")
        else:
            st.warning("Please provide a valid Hugging Face Token.")

if "hf_token" in st.session_state:
    st.header("Speaker Diarization and Transcription Process")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"Whisper and Pyannote are running on: **{device}**")

    diarization_pipeline = load_diarization_pipeline(st.session_state.hf_token)
    whisper_pipeline = load_whisper_pipeline()

    # Réglages dans la sidebar
    st.sidebar.subheader("⚙️ Settings")
    speaker_options = [""] + [str(i) for i in range(1, 11)]

    if "num_speakers" not in st.session_state:
        st.session_state.num_speakers = ""
    if "min_speakers" not in st.session_state:
        st.session_state.min_speakers = ""
    if "max_speakers" not in st.session_state:
        st.session_state.max_speakers = ""

    def update_num_speakers():
        st.session_state.min_speakers = ""
        st.session_state.max_speakers = ""

    def update_min_max_speakers():
        st.session_state.num_speakers = ""

    num_speakers = st.sidebar.selectbox(
        "Number of Speakers", speaker_options, key="num_speakers", on_change=update_num_speakers
    )
    min_speakers = st.sidebar.selectbox(
        "Minimum Speakers", speaker_options, key="min_speakers", on_change=update_min_max_speakers
    )
    max_speakers = st.sidebar.selectbox(
        "Maximum Speakers", speaker_options, key="max_speakers", on_change=update_min_max_speakers
    )

    # Section 1: Enregistrement audio
    st.header("Record Audio from Microphone")
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        format="wav"
    )

    chosen_audio_path = None

    if audio and "bytes" in audio:
        chosen_audio_path = "streamlit-records/output.wav"
        with open(chosen_audio_path, "wb") as f:
            f.write(audio["bytes"])
        st.audio(audio["bytes"], format='audio/wav')
        st.success(f"Audio saved as '{chosen_audio_path}'.")
    else:
        st.warning("No audio recorded.")

    st.markdown("---")

    # Section 2: Importer un fichier audio ou utiliser un existant
    st.header("Choose Audio Source")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        chosen_audio_path = "streamlit-records/uploaded_audio.wav"
        with open(chosen_audio_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.audio(uploaded_file, format='audio/wav')
        st.success(f"Uploaded audio saved as '{chosen_audio_path}'.")

    if st.button("Use Existing Audio"):
        chosen_audio_path = "audios/audio_DER.wav"
        st.write(f"Using default audio: {chosen_audio_path}")
        st.audio(chosen_audio_path, format='audio/wav')

    st.markdown("---")

    # État des boutons dans st.session_state
    if "run_transcription" not in st.session_state:
        st.session_state.run_transcription = False
    if "run_diarization" not in st.session_state:
        st.session_state.run_diarization = False

    # Boutons pour exécuter les processus
    if st.button("Run Transcription"):
        st.session_state.run_transcription = True

    if st.button("Run Diarization"):
        st.session_state.run_diarization = True

    # Exécution de la transcription si le bouton est pressé
    if st.session_state.run_transcription and chosen_audio_path and os.path.exists(chosen_audio_path):
        st.header("Running Transcription with Whisper")
        with st.spinner("Transcribing the audio..."):
            start_time = time.time()
            transcription = whisper_pipeline(chosen_audio_path)
        end_time = time.time()
        st.markdown("### Transcription Result")
        st.markdown(transcription["text"])
        st.info(f"Transcription completed in **{end_time - start_time:.2f} seconds**")
        st.session_state.run_transcription = False  # Réinitialiser l'état
        
    # Exécution de la diarisation si le bouton est pressé
    if st.session_state.run_diarization and chosen_audio_path and os.path.exists(chosen_audio_path):
        st.header("Running Speaker Diarization")
        with st.spinner("Processing the audio with PyAnnote..."):
            diarization_args = {}
            if st.session_state.num_speakers:
                diarization_args["num_speakers"] = int(st.session_state.num_speakers)
            if st.session_state.min_speakers:
                diarization_args["min_speakers"] = int(st.session_state.min_speakers)
            if st.session_state.max_speakers:
                diarization_args["max_speakers"] = int(st.session_state.max_speakers)
            start_time = time.time()
            diarization_result = diarization_pipeline(chosen_audio_path, **diarization_args)
        end_time = time.time()
        st.success("Diarization Completed!")
        st.info(f"Diarization completed in **{end_time - start_time:.2f} seconds**")
        st.session_state.run_diarization = False  # Réinitialiser l'état
else:
    st.warning("Please log in to Hugging Face to proceed.")
