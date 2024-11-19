# Transcription et Diarisation de Réunions

Ce projet utilise des bibliothèques comme `pyannote.audio` et `transformers` pour analyser des fichiers audio en identifiant "qui parle quand" (diarisation des locuteurs) et en transcrivant les conversations en texte (speech-to-text).

## Concepts clés

### 1. **Diarisation des locuteurs**
La diarisation consiste à analyser un fichier audio contenant plusieurs locuteurs pour identifier les segments correspondant à chaque locuteur. Cela permet de déterminer les intervalles de temps où chaque locuteur parle, sans produire de transcription.

#### Outils utilisés :
- **`pyannote.audio`** : Une bibliothèque spécialisée dans l'analyse de l'audio, utilisée pour séparer les locuteurs dans un enregistrement.

---

### 2. **Transcription automatique de la parole (ASR)**
La transcription automatique convertit un contenu audio en texte. Elle produit une transcription de ce qui a été dit, sans distinguer les différents locuteurs.

#### Outils utilisés :
- **`transformers`** : Une bibliothèque de modèles préentraînés de Hugging Face.
- **`Whisper`** : Un modèle développé par OpenAI, utilisé pour transcrire l'audio en texte avec des horodatages.

---

### 3. **Combinaison de la diarisation et de la transcription**
En fusionnant les résultats de la diarisation et de la transcription, il est possible de produire une transcription enrichie qui associe chaque segment de texte à un locuteur spécifique, avec les horodatages correspondants.

#### Outils utilisés :
- **`Speechbox`** : Une extension facilitant l'alignement des segments temporels entre la transcription et la diarisation.
