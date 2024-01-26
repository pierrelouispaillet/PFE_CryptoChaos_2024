import matplotlib.pyplot as plt
import numpy as np

def file_to_binary_string(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return ''.join(format(byte, '08b') for byte in binary_data)

def formant(taille, surechantillonnage):
    PI = 3.1415926535
    taille_vecteur = taille * surechantillonnage + 1
    x = np.arange(1, taille_vecteur + 1)
    x = x - (taille_vecteur + 1) / 2
    x = x / surechantillonnage
    y = np.zeros(taille_vecteur)
    for i in range(taille_vecteur):
        if (x[i] * x[i] == 1/16):
            y[i] = PI / 4
        else:
            y[i] = np.cos(2 * PI * x[i]) / (1 - 16 * x[i] * x[i])
    return y

binary_string = file_to_binary_string('money.mp3')
print(binary_string[:20])

with open('binarySend.txt', 'w') as file:
    file.write(binary_string)

taille_message = len(binary_string)
taille_vecteur = 10 * 20 + 1  # Défini en fonction des paramètres de la fonction formant

formant = formant(10, 20)
amplitude1 = 10
amplitude2 = -10
signal = np.zeros(taille_message * taille_vecteur)

for i in range(taille_message):
    start = i * taille_vecteur
    end = start + taille_vecteur
    if binary_string[i] == '1':
        signal[start:end] = amplitude1 * formant
    else:
        signal[start:end] = amplitude2 * formant

x = np.arange(len(signal[:5000]))  

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x, signal[:5000])
plt.title('AWG simulation plot')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Sauvegarde de signal dans un fichier texte
np.savetxt("signal_data.txt", signal[:5000])

# Ouvrir le fichier original et lire les lignes
with open('signal_data.txt', 'r') as file:
    lines = file.readlines()

# Modifier chaque ligne
modified_lines = [line.strip() + ',0,0\n' for line in lines]

# Écrire les lignes modifiées dans un nouveau fichier (ou écraser l'ancien)
with open('signal_data_modified.txt', 'w') as file:
    file.writelines(modified_lines)
