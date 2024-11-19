import os
import json
from flask import Flask, render_template, request, jsonify, Response
from pyannote.audio import Pipeline
from tempfile import NamedTemporaryFile
import whisper
from pydub import AudioSegment
import torch

app = Flask(__name__, template_folder='static/html', static_url_path='/static')

hf_token = "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"

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=hf_token
)

# Cuda
if torch.cuda.is_available():
    diarization_pipeline.to("cuda")

whisper_model = whisper.load_model("base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': "Aucun fichier n'a été envoyé."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': "Aucun fichier sélectionné."}), 400

    is_chunk = request.form.get('is_chunk', 'false').lower() == 'true'

    # Sauvegarder temporairement le fichier audio
    with NamedTemporaryFile(delete=False) as temp_audio_file:
        file.save(temp_audio_file.name)
        temp_audio_path = temp_audio_file.name

    try:
        # Si c'est un segment audio en cours d'enregistrement
        if is_chunk:
            result = process_audio_chunk(temp_audio_path)
            return jsonify(result)
        else:
            # Si c'est le fichier complet ou le dernier segment
            response_data = process_audio(temp_audio_path)
            return jsonify(response_data)
    except Exception as e:
        # Supprimer le fichier temporaire en cas d'erreur
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return jsonify({'error': str(e)}), 500

@app.route('/process-default-audio', methods=['GET'])
def process_default_audio():
    default_audio_path = os.path.join(os.path.dirname(__file__), "audio.mp3")
    if not os.path.exists(default_audio_path):
        return jsonify({'error': "Le fichier audio par défaut (audio.mp3) n'existe pas."}), 400

    def generate():
        # Charger l'audio par défaut
        audio = AudioSegment.from_file(default_audio_path)
        chunk_length_ms = 7000

        for i, chunk in enumerate(audio[::chunk_length_ms]):
            if len(chunk) < 200:  # Ignorer les segments trop courts
                continue

            # Sauvegarder le segment temporairement
            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
                chunk.export(temp_wav_file.name, format="wav")
                temp_wav_path = temp_wav_file.name

            # Transcrire avec Whisper
            transcription_result = whisper_model.transcribe(temp_wav_path, language='en')

            # Diarisation avec PyAnnote
            diarization_result = diarization_pipeline(temp_wav_path)

            # Supprimer le fichier temporaire
            os.remove(temp_wav_path)

            # Associer transcription et locuteur
            if 'segments' in transcription_result and transcription_result['segments']:
                speaker_segments = match_transcription_with_speakers(transcription_result, diarization_result)
                for segment in speaker_segments:
                    yield f"data:{json.dumps(segment)}\n\n"
            else:
                # Aucun segment de transcription trouvé
                no_transcription = {
                    'speaker': 'Unknown',
                    'start': 0,
                    'end': len(chunk) / 1000.0,
                    'text': 'Aucune transcription disponible.'
                }
                yield f"data:{json.dumps(no_transcription)}\n\n"

        # indiquer la fin du traitement
        yield "event:end\n\n"

    return Response(generate(), mimetype='text/event-stream')

def process_audio_chunk(audio_path):
    """
    Traite un segment audio pendant l'enregistrement pour renvoyer des résultats partiels.
    """
    # Charger le segment audio actuel
    segment_audio = AudioSegment.from_file(audio_path)
    os.remove(audio_path)  # nettoyer le segment temporaire

    if len(segment_audio) < 200:  # moins de 200 ms
        return []

    # Exporte le segment audio dans un fichier WAV temporaire
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        segment_audio.export(temp_wav_file.name, format="wav")
        temp_wav_path = temp_wav_file.name

    # Effectuer la transcription du segment avec Whisper
    segment_transcription = whisper_model.transcribe(temp_wav_path, language='en')

    # Diarisation du segment avec PyAnnote
    diarization_result = diarization_pipeline(temp_wav_path)

    # Supprimer le fichier WAV temporaire
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    # Associer chaque segment de transcription à un locuteur
    if 'segments' not in segment_transcription or not segment_transcription['segments']:
        return []

    speaker_segments = match_transcription_with_speakers(segment_transcription, diarization_result)

    return speaker_segments

def process_audio(audio_path):
    """
    Traite le fichier audio spécifié pour la diarisation et la transcription.
    """
    # Convertir le fichier audio en WAV pour compatibilité avec les modèles
    audio = AudioSegment.from_file(audio_path)
    audio_duration_seconds = len(audio) / 1000.0  # Durée en secondes
    if audio_duration_seconds < 0.5:
        return [{
            'speaker': 'Unknown',
            'start': 0,
            'end': 0,
            'text': "L'audio est trop court pour être traité."
        }]

    wav_path = audio_path + ".wav"
    audio.export(wav_path, format="wav")

    # Exécuter la diarisation pour détecter les locuteurs
    diarization_result = diarization_pipeline(wav_path)

    # Vérifier si la diarisation a trouvé des segments de parole
    if not diarization_result or len(diarization_result.labels()) == 0:
        if os.path.exists(wav_path) and not audio_path.endswith(".wav"):
            os.remove(wav_path)
        return [{
            'speaker': 'Unknown',
            'start': 0,
            'end': audio_duration_seconds,
            'text': 'Silence ou parole non détectée.'
        }]

    # Transcrire l'audio complet avec Whisper
    transcription_result = whisper_model.transcribe(wav_path)

    # Vérifier que la transcription ne soit pas vide
    if 'segments' not in transcription_result or not transcription_result['segments']:
        if os.path.exists(wav_path) and not audio_path.endswith(".wav"):
            os.remove(wav_path)
        return [{
            'speaker': 'Unknown',
            'start': 0,
            'end': audio_duration_seconds,
            'text': 'Aucune transcription disponible.'
        }]

    # Associer chaque segment de transcription à un locuteur détecté via la pipeline
    speaker_segments = match_transcription_with_speakers(transcription_result, diarization_result)

    # Supprimer les fichiers audio temporaires
    if os.path.exists(wav_path) and not audio_path.endswith(".wav"):
        os.remove(wav_path)

    return speaker_segments

def match_transcription_with_speakers(transcription_result, diarization_result):
    """
    Associe les segments de transcription aux locuteurs détectés par PyAnnote.
    """
    speaker_segments = []
    for segment in transcription_result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        # Trouver le locuteur correspondant en fonction du chevauchement le plus important
        speaker = 'Unknown'
        max_overlap = 0.0
        for spk_segment, _, label in diarization_result.itertracks(yield_label=True):
            overlap_start = max(start_time, spk_segment.start)
            overlap_end = min(end_time, spk_segment.end)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > max_overlap and overlap_duration > 0:
                max_overlap = overlap_duration
                speaker = label

        speaker_segments.append({
            'speaker': speaker,
            'start': start_time,
            'end': end_time,
            'text': text.strip()
        })

    return speaker_segments

if __name__ == '__main__':
    # Démarrer le serveur Flask sur le port 5000
    app.run(debug=True, host='localhost', port=5000)
