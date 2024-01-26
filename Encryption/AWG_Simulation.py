import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np


#47.1ns


def plot_oscilloscope_data(relative_filepath,relative2):
    data_np = np.loadtxt(relative_filepath, delimiter=' ', skiprows=5)
    data_np2 = np.loadtxt(relative2, delimiter=' ', skiprows=5)

    time = data_np[:, 0]  # time column
    ampl = data_np[:, 1]  # amplitude column

    time2 = data_np2[:, 0]  # time column
    ampl2 = data_np2[:, 1]  # amplitude column

    ampl2 =    ampl2 -ampl
    print(len(ampl))
    print(len(ampl2))
    # Convertir les samples en bytes
    raw_data = ampl.tobytes()
    raw_data2 = ampl2.tobytes()


    # Créer un AudioSegment à partir des données brutes
    # Remarque : Vous devrez connaître la fréquence d'échantillonnage, les canaux et la profondeur des bits de vos données originales
    sample_rate = 650000 # Par exemple, 44100 Hz
    channels = 2         # 1 pour mono, 2 pour stéréo
    sample_width = 2     # En bytes, 2 pour 16 bits par échantillon

    audio_segment = AudioSegment(
        data=raw_data2,
        sample_width=sample_width,
        frame_rate=sample_rate,
        channels=channels
    )

    # Exporter en fichier MP3
    audio_segment.export("testBruitChaos.mp3", format="mp3")
    # create plot
    plt.figure(figsize=(30, 6))
    plt.plot(time2 , ampl2)
    plt.title('Waveform from Oscilloscope')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()




osc_file = "C1_modified.txt"
osc_file2 = "C2_modified.txt"
plot_oscilloscope_data(osc_file, osc_file2)

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def plot_oscilloscope_data(relative_filepath, relative2):
    data_np = np.loadtxt(relative_filepath, delimiter=' ', skiprows=5)
    data_np2 = np.loadtxt(relative2, delimiter=' ', skiprows=5)

    # separate columns
    time = data_np[:, 0]  # time column
    ampl = data_np[:, 1]  # amplitude column
    time2 = data_np2[:, 0]  # time column
    ampl2 = data_np2[:, 1]  # amplitude column

    # Calcul de la FFT
    N = len(ampl2)
    T = time2[1] - time2[0]  # intervalle de temps entre échantillons
    yf = fft(ampl2)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    # Créer le graphique de la FFT
    plt.figure(figsize=(30, 6))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title('Transformée de Fourier du signal')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

# Appel de la fonction
osc_file = "C2bruit.txt"
osc_file2 = "C2yy00008.txt"
#plot_oscilloscope_data(osc_file, osc_file2)
