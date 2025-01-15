import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
from pyannote.audio import Pipeline
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import time
import wave

# Hugging Face Token
TOKEN_FILE = ".token"

def get_token_from_file():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file:
            return file.read().strip()
    return None

def save_token_to_file(token):
    with open(TOKEN_FILE, "w") as file:
        file.write(token)

@st.cache_resource
def load_diarization_pipeline(token):
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    return diar_pipeline

@st.cache_resource
def load_whisper_pipeline(model_id):
    """
    Charge 1 seul modèle Whisper. On profite du cache pour éviter de le recharger
    si on revient sur le même modèle. Mais si on change de modèle, on force la
    libération du précédent.
    """
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    if device != -1:  # GPU
        model = model.cuda()

    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    return asr_pipeline

# Récupérer le token
hf_token = get_token_from_file()

if hf_token:
    st.success("Token loaded from .token file.")
    login(token=hf_token)
    st.session_state.hf_token = hf_token

    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"Whisper and Pyannote are running on: **{device}**")
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

# ----------------------------------------------------------------------
# Quand on change de modèle dans la sidebar, on veut libérer l'ancien
# ----------------------------------------------------------------------
def switch_whisper_model_if_needed(new_model_id):
    """
    Si l'utilisateur a changé de modèle, on supprime le pipeline précédent
    de la session_state pour forcer le cache Streamlit à recharger.
    On vide également la mémoire GPU (empty_cache).
    """
    if "current_whisper_model" not in st.session_state:
        st.session_state.current_whisper_model = new_model_id
    elif st.session_state.current_whisper_model != new_model_id:
        # Le modèle a changé
        st.session_state.current_whisper_model = new_model_id

        # Tenter de supprimer le pipeline précédent
        if "whisper_pipeline" in st.session_state:
            del st.session_state["whisper_pipeline"]

        # Vider la mémoire GPU au besoin
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ----------------------------------------------------------------------

if "hf_token" in st.session_state:
    st.header("🔊 Speaker Diarization and Transcription")

    # On charge la pipeline de diarisation
    diarization_pipeline = load_diarization_pipeline(st.session_state.hf_token)

    # --- Choix du modèle Whisper ---
    st.sidebar.subheader("🗣 Whisper Model")
    whisper_model_options = [
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
        "openai/whisper-tiny"
    ]
    if "whisper_model_choice" not in st.session_state:
        st.session_state.whisper_model_choice = "openai/whisper-large-v3-turbo"

    chosen_model = st.sidebar.selectbox(
        "Select a model for Whisper",
        whisper_model_options,
        index=0
    )

    # --- Choix de la langue pour Whisper (forced_decoder_ids) ---
    st.sidebar.subheader("Language Settings")
    whisper_lang_options = ["auto-detect", "en", "fr", "de", "es"]
    if "whisper_lang" not in st.session_state:
        st.session_state.whisper_lang = "auto-detect"

    st.session_state.whisper_lang = st.sidebar.selectbox(
        "Select language",
        whisper_lang_options,
        index=0
    )

    # Gérer le changement de modèle (libère le précédent si besoin)
    switch_whisper_model_if_needed(chosen_model)

    # Charger le pipeline dans st.session_state si pas déjà présent
    if "whisper_pipeline" not in st.session_state:
        # Charger le pipeline correspondant au modèle actuel
        st.session_state.whisper_pipeline = load_whisper_pipeline(st.session_state.current_whisper_model)
    whisper_pipeline = st.session_state.whisper_pipeline

    # Mettre à jour la variable du modèle choisi
    st.session_state.whisper_model_choice = chosen_model

    # Réglages dans la sidebar
    st.sidebar.subheader("⚙️ Diarization Settings")
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
        "Number of Speakers",
        speaker_options,
        key="num_speakers",
        on_change=update_num_speakers
    )
    min_speakers = st.sidebar.selectbox(
        "Minimum Speakers",
        speaker_options,
        key="min_speakers",
        on_change=update_min_max_speakers
    )
    max_speakers = st.sidebar.selectbox(
        "Maximum Speakers",
        speaker_options,
        key="max_speakers",
        on_change=update_min_max_speakers
    )

    # Initialiser le chemin audio et le mode path
    if "chosen_audio_path" not in st.session_state:
        st.session_state.chosen_audio_path = None
    if "path" not in st.session_state:
        st.session_state.path = False

    # Section 1 : Enregistrement audio
    st.subheader("Record Audio from Microphone")
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        format="wav"
    )

    if audio and "bytes" in audio:
        st.session_state.chosen_audio_path = "streamlit-records/output.wav"
        with open(st.session_state.chosen_audio_path, "wb") as f:
            f.write(audio["bytes"])
        st.audio(audio["bytes"], format='audio/wav')
        st.success(f"Audio saved as '{st.session_state.chosen_audio_path}'.")
        st.session_state.path = "recorded"
    else:
        st.warning("No audio recorded.")

    st.markdown("---")

    # Section 2 : Importer un fichier audio ou utiliser un existant
    st.subheader("Choose Audio Source")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.session_state.chosen_audio_path = "streamlit-records/uploaded_audio.wav"
        with open(st.session_state.chosen_audio_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.path = "uploaded"

    if st.button("Use existing audio"):
        st.session_state.chosen_audio_path = "audios/audio_DER.wav"
        st.session_state.path = "default"

    if st.session_state.chosen_audio_path:
        if st.session_state.path == "default":
            st.write(f"Using default audio: {st.session_state.chosen_audio_path}")
        elif st.session_state.path == "uploaded":
            st.success(f"Uploaded audio saved as '{st.session_state.chosen_audio_path}'.")
        elif st.session_state.path == "recorded":
            st.success(f"Audio recorded at '{st.session_state.chosen_audio_path}'.")

        st.audio(st.session_state.chosen_audio_path, format='audio/wav')

    st.markdown("---")

    # Exécution si on a un chemin audio
    if st.session_state.chosen_audio_path:

        if "run_transcription" not in st.session_state:
            st.session_state.run_transcription = False
        if "run_diarization" not in st.session_state:
            st.session_state.run_diarization = False
        if "transcription_result" not in st.session_state:
            st.session_state.transcription_result = None
        if "diarization_result" not in st.session_state:
            st.session_state.diarization_result = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Transcription"):
                st.session_state.run_transcription = True

        with col2:
            if st.button("▶️ Diarization"):
                st.session_state.run_diarization = True

        # --- TRANSCRIPTION ---
        if (
            st.session_state.run_transcription
            and st.session_state.chosen_audio_path
            and os.path.exists(st.session_state.chosen_audio_path)
        ):
            with st.spinner("Transcribing the audio..."):
                start_time = time.time()

                current_pipeline = st.session_state.whisper_pipeline

                # Forcer la langue si nécessaire via forced_decoder_ids
                if st.session_state.whisper_lang != "auto-detect":
                    forced_decoder_ids = current_pipeline.tokenizer.get_decoder_prompt_ids(
                        language=st.session_state.whisper_lang,
                        task="transcribe"
                    )
                    current_pipeline.model.config.forced_decoder_ids = forced_decoder_ids
                else:
                    current_pipeline.model.config.forced_decoder_ids = None

                transcription = current_pipeline(st.session_state.chosen_audio_path)

                end_time = time.time()
                st.session_state.transcription_result = transcription["text"]
                st.session_state.transcription_time = end_time - start_time
                st.session_state.run_transcription = False

        if st.session_state.transcription_result:
            st.markdown("### 📄 Transcription Result:")
            st.markdown(st.session_state.transcription_result)
            if "transcription_time" in st.session_state:
                st.info(f"Transcription completed in **{st.session_state.transcription_time:.2f} seconds**")

        st.markdown("---")

        # --- DIARIZATION ---
        if (
            st.session_state.run_diarization
            and st.session_state.chosen_audio_path
            and os.path.exists(st.session_state.chosen_audio_path)
        ):
            with st.spinner("Processing the audio with Pyannote..."):
                diarization_args = {}
                if st.session_state.num_speakers:
                    diarization_args["num_speakers"] = int(st.session_state.num_speakers)
                if st.session_state.min_speakers:
                    diarization_args["min_speakers"] = int(st.session_state.min_speakers)
                if st.session_state.max_speakers:
                    diarization_args["max_speakers"] = int(st.session_state.max_speakers)

                start_time = time.time()
                diarization_result = diarization_pipeline(
                    st.session_state.chosen_audio_path,
                    **diarization_args
                )
                end_time = time.time()
                st.session_state.diarization_result = diarization_result
                st.session_state.diarization_time = end_time - start_time
                st.session_state.run_diarization = False

        if st.session_state.diarization_result:
            st.markdown("### 🎤 Diarization Result:")
            st.write(st.session_state.diarization_result)
            if "diarization_time" in st.session_state:
                st.info(f"Diarization completed in **{st.session_state.diarization_time:.2f} seconds**")

else:
    st.warning("Please log in to Hugging Face to proceed.")
