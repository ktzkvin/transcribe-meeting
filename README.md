# Projet de Transcription et Diarisation en Temps Réel

Ce projet utilise Whisper pour la transcription audio et PyAnnote pour la diarisation (détection des locuteurs). L'objectif est de réaliser une transcription diarisée en temps réel à partir d'enregistrements audio.

## Fonctionnalités
- **Transcription en temps réel** : Convertit l'audio en texte.
- **Diarisation des locuteurs** : Identifie les différents locuteurs et leur attribue les segments de transcription correspondants.
- **Détection de silence** : Arrête et relance l'enregistrement en fonction des silences détectés.

## Technologies Utilisées
- **Flask** : Pour créer le serveur backend.
- **Whisper (OpenAI)** : Pour la transcription de l'audio en texte.
- **PyAnnote** : Pour la segmentation des locuteurs (diarisation).
- **JavaScript et HTML** : Interface web pour lancer et arrêter l'enregistrement.

## Installation
1. Cloner le projet.
2. Installer les dépendances Python :
   ```bash
   pip install -r requirements.txt
