from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Charger le fichier audio (en supposant que c'est un fichier WAV)
fs, data = wavfile.read('testOscillo.wav')

# Appliquer le filtre passe-bande
lowcut = 1000   # Fréquence de coupure basse en Hz
highcut = 20000 # Fréquence de coupure haute en Hz
filtered_data = bandpass_filter(data, lowcut, highcut, fs)

# Écrire les données filtrées dans un nouveau fichier WAV
wavfile.write('bandpass_filtered_output.wav', fs, filtered_data.astype(np.int16))
