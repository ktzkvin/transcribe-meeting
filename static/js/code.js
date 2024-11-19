const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const transcriptionResult = document.getElementById('transcriptionResult');
const audioFileInput = document.getElementById('audioFileInput');
const uploadFileButton = document.getElementById('uploadFileButton');
const uploadDefaultFileButton = document.getElementById('uploadDefaultFileButton');

let mediaRecorder;
let audioContext;
let analyser;
let dataArray;
let silenceThreshold = 0.02;
let silenceDuration = 1500;
let silenceStart = 0;
let recordingStartTime = 0;
let segmentDuration = 7000;
let segmentInterval;
let recordingInterval;
let shouldRestartRecording = false;

/**
 * Gestion de l'enregistrement audio via le microphone
 */
recordButton.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        audioContext = new AudioContext();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        dataArray = new Float32Array(analyser.fftSize);

        const chunks = [];
        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                chunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            // Démarrer l'enregistrement segmenté
            chunks.length = 0; // vider le tableau des segments
            segmentInterval = setInterval(async () => {
                if (mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    await new Promise(resolve => setTimeout(resolve, 50)); // petite attente pour permettre ondataavailable d'ajouter les données
                    mediaRecorder.start();
                    await sendAudioChunkToServer(chunks);
                    chunks.length = 0;
                }
            }, segmentDuration);
        };

        mediaRecorder.onstop = async () => {
            if (shouldRestartRecording) {
                shouldRestartRecording = false;
                return;
            }

            clearInterval(segmentInterval);
            clearInterval(recordingInterval);

            // Envoyer les données restantes
            if (chunks.length > 0) {
                await sendAudioToServer(chunks);
                chunks.length = 0;
            }

            if (audioContext) {
                audioContext.close();
            }

            recordButton.disabled = false;
            stopButton.disabled = true;
        };

        mediaRecorder.start();
        recordingStartTime = Date.now();
        monitorSilence();
        recordButton.disabled = true;
        stopButton.disabled = false;
    } catch (error) {
        console.error('Error accessing microphone:', error);
    }
});

stopButton.addEventListener('click', () => {
    shouldRestartRecording = false;
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
});

audioFileInput.addEventListener('change', () => {
    // Activer le bouton de téléversement seulement si un fichier est sélectionné
    uploadFileButton.disabled = !audioFileInput.files.length;
});

uploadFileButton.addEventListener('click', async () => {
    if (audioFileInput.files.length === 0) {
        console.warn('Aucun fichier sélectionné.');
        return;
    }

    const file = audioFileInput.files[0];
    if (!file) {
        console.warn('Fichier invalide.');
        return;
    }

    await sendAudioFileToServer(file);
    // Réinitialiser le champ de fichier
    audioFileInput.value = '';
    uploadFileButton.disabled = true;
});

uploadDefaultFileButton.addEventListener('click', async () => {
    try {
        // Désactiver le bouton pour éviter les clics multiples
        uploadDefaultFileButton.disabled = true;

        // Réinitialiser les résultats précédents
        transcriptionResult.textContent = '';

        // Créer un EventSource pour recevoir les résultats via SSE
        const eventSource = new EventSource('/process-default-audio');

        eventSource.onmessage = function(event) {
            if (event.data) {
                try {
                    const segment = JSON.parse(event.data);
                    transcriptionResult.textContent += `Locuteur ${segment.speaker}: ${segment.text}\n`;
                } catch (e) {
                    console.error('Error parsing segment:', e);
                }
            }
        };

        eventSource.onerror = function(err) {
            if (eventSource.readyState === EventSource.CLOSED) {
                console.log('Connexion SSE fermée normalement.');
            } else {
                console.error('EventSource failed:', err);
                transcriptionResult.textContent += 'Error: Échec du traitement du fichier audio par défaut.\n';
            }
            eventSource.close();
            uploadDefaultFileButton.disabled = false;
        };

        eventSource.onopen = function() {
            console.log('Début du traitement du fichier audio par défaut.');
        };

        // Optionnel : gérer la fermeture de la connexion SSE
        eventSource.addEventListener('end', function() {
            console.log('Fin du traitement du fichier audio par défaut.');
            eventSource.close();
            uploadDefaultFileButton.disabled = false;
        });

    } catch (error) {
        console.error('Error processing default audio:', error);
        transcriptionResult.textContent += 'Error: ' + error.message + '\n\n';
        uploadDefaultFileButton.disabled = false;
    }
});

/**
 * Fonction pour envoyer un segment audio (depuis le microphone) au serveur en temps réel
 */
async function sendAudioChunkToServer(chunks) {
    const audioBlob = new Blob(chunks, { type: 'audio/wav' });
    if (audioBlob.size === 0) return;

    const formData = new FormData();
    formData.append('file', audioBlob, 'chunk.wav');
    formData.append('is_chunk', 'true'); // Indique au serveur qu'il s'agit d'un segment

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        handleServerChunkResponse(result);
    } catch (error) {
        console.error('Error sending audio chunk to server:', error);
        transcriptionResult.textContent += 'Error: ' + error.message + '\n\n';
    }
}

/**
 * Fonction pour envoyer l'audio complet (dernier segment ou tout l'audio restant) au serveur
 */
async function sendAudioToServer(chunks) {
    const audioBlob = new Blob(chunks, { type: 'audio/wav' });
    if (audioBlob.size === 0) return;

    const formData = new FormData();
    formData.append('file', audioBlob, 'file.wav');
    formData.append('is_chunk', 'false'); // Indique au serveur qu'il s'agit du dernier segment ou d'un fichier complet

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        handleServerResponse(result);
    } catch (error) {
        console.error('Error sending audio to server:', error);
        transcriptionResult.textContent += 'Error: ' + error.message + '\n\n';
    }
}

/**
 * Fonction pour envoyer un fichier audio (depuis l'ordinateur) au serveur
 */
async function sendAudioFileToServer(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('is_chunk', 'false');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        handleServerResponse(result);
    } catch (error) {
        console.error('Error sending audio file to server:', error);
        transcriptionResult.textContent += 'Error: ' + error.message + '\n\n';
    }
}

function handleServerChunkResponse(result) {
    // Affiche les résultats partiels pour chaque segment
    if (Array.isArray(result)) {
        displayTranscriptionResult(result);
    } else if (result.error) {
        transcriptionResult.textContent += 'Error: ' + result.error + '\n\n';
    }
}

function handleServerResponse(result) {
    if (Array.isArray(result)) {
        displayTranscriptionResult(result);
    } else if (result.error) {
        transcriptionResult.textContent += 'Error: ' + result.error + '\n\n';
    } else {
        transcriptionResult.textContent += 'Réponse inattendue du serveur\n\n';
    }
}

function displayTranscriptionResult(segments) {
    segments.forEach(segment => {
        transcriptionResult.textContent += `Locuteur ${segment.speaker}: ${segment.text}\n`;
    });
}

function monitorSilence() {
    recordingInterval = setInterval(() => {
        analyser.getFloatTimeDomainData(dataArray);
        let silence = true;
        for (let i = 0; i < dataArray.length; i++) {
            if (Math.abs(dataArray[i]) > silenceThreshold) {
                silence = false;
                silenceStart = 0;
                break;
            }
        }

        const currentTime = Date.now();

        // Si silence prolongé, on arrête l'enregistrement
        if (silence) {
            if (silenceStart === 0) {
                silenceStart = currentTime;
            } else if (currentTime - silenceStart > silenceDuration) {
                shouldRestartRecording = false;
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
                clearInterval(recordingInterval);
                if (audioContext) {
                    audioContext.close();
                }
            }
        }
    }, 100);
}
