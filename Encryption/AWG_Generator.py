

#__________________________________________V2 Modulation_______________________
import numpy as np

def text_to_binary(text):
    binary_message = ''.join(format(ord(char), '08b') for char in text)
    return np.array([int(bit) for bit in binary_message])

def generate_awg_file_custom_amplitude_modulation(message, file_path):
    # S'assurer que la longueur du message est suffisante pour atteindre 2400 échantillons
    while len(message) * 8 < 2500:
        message += message

    # Tronquer le message pour obtenir exactement 2500 bits
    binary_message = text_to_binary(message)[:2500]

    # Paramètres du signal
    fs = 1000  # Fréquence d'échantillonnage
    t = np.arange(0, len(binary_message) / fs, 1/fs)  # Temps d'échantillonnage
    f_carrier = 10  # Fréquence de la porteuse

    # Créer la porteuse
    carrier = np.sin(2 * np.pi * f_carrier * t)

    # Modulation d'amplitude personnalisée pour la séquence "01000001"
    modulated_signal = np.zeros_like(t)

    for i, bit in enumerate(binary_message):
        if bit == 1:
            modulated_signal += np.sin(2 * np.pi * f_carrier * t)  # Pic d'amplitude
        else:
            modulated_signal += np.zeros_like(t)  # Amplitude nulle

    # Créer un fichier texte compatible avec l'AWG
    with open(file_path, 'w') as file:
        for amp in modulated_signal:
            file.write(f'{amp:.9e},0,0\n')

# Exemple d'utilisation
message_text = "Bonjour"
awg_file_path = r'C:\Users\plpai\Desktop\Encryption\Bonjour.txt'
generate_awg_file_custom_amplitude_modulation(message_text, awg_file_path)
print(f'Fichier AWG généré avec succès : {awg_file_path}')


# # #___________________________Demodulation V2____________________________________
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert, find_peaks

# # Fonction de démodulation AM avec détection de pics
# def demodulate_am_signal_with_peaks(data):
#     time = data[:, 0]
#     amplitude = data[:, 1]

#     # Appliquer la transformée de Hilbert pour obtenir l'enveloppe
#     analytic_signal = hilbert(amplitude)
#     envelope = np.abs(analytic_signal)

#     # Utiliser find_peaks pour détecter les pics
#     peaks, _ = find_peaks(envelope, height=0.0075)  # Ajustez le paramètre de hauteur selon vos besoins

#     # Binariser en utilisant les pics détectés
#     binary_sequence = np.zeros_like(envelope)
#     binary_sequence[peaks] = 1

#     # Visualisation
#     plt.figure(figsize=(12, 8))

#     plt.subplot(2, 1, 1)
#     plt.plot(time, amplitude, label='Signal Modulé')
#     plt.title('Signal Modulé')
#     plt.xlabel('Temps')
#     plt.ylabel('Amplitude')
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(time, envelope, label='Enveloppe')
#     plt.plot(time[peaks], envelope[peaks], 'r.', label='Pics détectés')
#     plt.title('Enveloppe du Signal Modulé avec Pics détectés')
#     plt.xlabel('Temps')
#     plt.ylabel('Amplitude')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     return binary_sequence

# # Charger les données en ignorant les 5 premières lignes
# file_path = r'C:\Users\mercadieju\Documents\Manip\Yaya\2023_Modulation\Audio_generation_1\oscillo\C2cst00000.txt'
# data = np.loadtxt(file_path, skiprows=5)

# # Démoulation du signal modulé AM avec détection de pics
# binary_sequence = demodulate_am_signal_with_peaks(data)

# # Fonction pour décoder la séquence binaire en texte ASCII
# def decode_binary_sequence(binary_sequence):
#     # Convertir la séquence binaire en une chaîne de bits (en ignorant les points décimaux)
#     bit_string = ''.join(map(str, binary_sequence))

#     # Ignorer les points décimaux
#     bit_string = bit_string.replace('.', '')

#     # Décoder la chaîne de bits en texte ASCII
#     decoded_text = ''.join([chr(int(bit_string[i:i+8], 2)) for i in range(0, len(bit_string), 8)])

#     return decoded_text

# # Décoder la séquence binaire en texte
# decoded_text = decode_binary_sequence(binary_sequence)

# # Afficher le texte décodé
# print("Texte Décodé :", decoded_text)

#______________________________________________________________________________
#______________________________________________________________________________


import numpy as np
from scipy.signal import hilbert, find_peaks
import matplotlib.pyplot as plt

def modulate_amplitude(letter, fs, f_carrier, pulse_width, duration):
    # Convertir la lettre en binaire
    binary_message = format(ord(letter), '08b')

    # Paramètres temporels
    t = np.arange(0, duration, 1/fs)

    # Porteuse
    carrier = np.sin(2 * np.pi * f_carrier * t)

    # Signal modulé en amplitude
    modulated_signal = np.zeros_like(t)
    for i, bit in enumerate(binary_message):
        if bit == '1':
            pulse = np.ones(int(pulse_width * fs))
            modulated_signal[i*int(pulse_width * fs):(i+1)*int(pulse_width * fs)] = pulse

    modulated_signal *= carrier

    # Enregistrement du signal modulé dans un fichier texte
    awg_file_path = r'C:\Users\mercadieju\Documents\Manip\Yaya\2023_Modulation\Modulation_2\awg_file_modulation.txt'
    with open(awg_file_path, 'w') as file:
        for amp in modulated_signal:
            file.write(f'{amp:.9e},0,0\n')

    # Visualisation du signal modulé
    plt.plot(t, modulated_signal)
    plt.title('Signal Modulé en Amplitude')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.show()

    return t, modulated_signal

# Paramètres de modulation
fs = 1000  # Fréquence d'échantillonnage
f_carrier = 10  # Fréquence de la porteuse
pulse_width = 0.1  # Largeur de l'impulsion en secondes
duration = 8  # Durée totale du signal en secondes

# Modulation de la lettre "A"
letter_to_modulate = "A"

#time, modulated_signal = modulate_amplitude(letter_to_modulate, fs, f_carrier, pulse_width, duration)
#______________________________________________________________________________


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks

# Fonction de démodulation AM avec détection de pics
def demodulate_am_signal_with_peaks(file_path, fs, pulse_width):
    # Charger les données en ignorant les 5 premières lignes
    data = np.loadtxt(file_path, skiprows=5)

    time = data[:, 0]
    amplitude = data[:, 1]

    # Appliquer la transformée de Hilbert pour obtenir l'enveloppe
    analytic_signal = hilbert(amplitude)
    envelope = np.abs(analytic_signal)

    # Utiliser find_peaks pour détecter les pics
    peaks, _ = find_peaks(envelope, height=0.011)  # Ajustez le paramètre de hauteur selon vos besoins

    # Binariser en utilisant les pics détectés
    binary_sequence = np.zeros_like(envelope)
    binary_sequence[peaks] = 1

    # Visualisation
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time, amplitude, label='Signal Modulé')
    plt.title('Signal Modulé')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, envelope, label='Enveloppe')
    plt.plot(time[peaks], envelope[peaks], 'r.', label='Pics détectés')
    plt.title('Enveloppe du Signal Modulé avec Pics détectés')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return binary_sequence

# Charger les données en ignorant les 5 premières lignes
file_path = r'C:\Users\mercadieju\Documents\Manip\Yaya\2023_Modulation\Audio_generation_1\oscillo\C2testtexte300000.txt'

# Démoulation du signal modulé AM avec détection de pics
binary_sequence = demodulate_am_signal_with_peaks(file_path, fs, pulse_width)

# Fonction pour décoder la séquence binaire en texte ASCII
def decode_binary_sequence(binary_sequence):
    # Convertir la séquence binaire en une chaîne de bits (en ignorant les points décimaux)
    bit_string = ''.join(map(str, binary_sequence))

    # Ignorer les points décimaux
    bit_string = bit_string.replace('.', '')

    # Décoder la chaîne de bits en texte ASCII
    decoded_text = ''.join([chr(int(bit_string[i:i+8], 2)) for i in range(0, len(bit_string), 8)])

    return decoded_text

# Décoder la séquence binaire en texte
decoded_text = decode_binary_sequence(binary_sequence)

# Afficher le texte décodé
print("Texte Décodé :", decoded_text)



