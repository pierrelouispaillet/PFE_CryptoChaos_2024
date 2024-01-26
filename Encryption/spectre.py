from pydub import AudioSegment
from scipy.fft import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

def convert_mp3_to_wav(mp3_filepath, wav_filepath):
    # Charger le fichier MP3 et exporter en WAV
    audio = AudioSegment.from_mp3(mp3_filepath)
    audio.export(wav_filepath, format="wav")

def plot_frequency_spectrum(filepath, start_freq=0):
    # Lire le fichier audio
    sample_rate, data = wavfile.read(filepath)

    # Si les données sont stéréo, prendre seulement un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Appliquer FFT sur les données audio
    N = len(data)
    yf = fft(data)
    xf = np.linspace(0.0, sample_rate / 2.0, N // 2)

    # Filtrer les données pour afficher à partir de 'start_freq'
    indices = np.where(xf >= start_freq)[0]
    xf = xf[indices]
    yf = yf[indices]

    # Créer un graphique du spectre en fréquence
    plt.figure(figsize=(30, 6))
    plt.plot(xf, 2.0 / N * np.abs(yf[:len(xf)]))
    plt.title("Spectre en fréquence à partir de {} Hz".format(start_freq))
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# Exemple d'utilisation : Tracer le spectre en fréquence à partir de 1000 Hz



# Chemin du fichier MP3
mp3_file = "testOscillo.mp3"
# Chemin du fichier WAV à créer
wav_file = "testOscillo.wav"

# Convertir MP3 en WAV
convert_mp3_to_wav(mp3_file, wav_file)

# Appel de la fonction pour tracer le spectre en fréquence
plot_frequency_spectrum(wav_file, start_freq=50)
