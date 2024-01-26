from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np



# Charger le fichier MP3
audio = AudioSegment.from_mp3("12280.mp3")

# Convertir en échantillons audio
samples = np.array(audio.get_array_of_samples())

print(len(samples))
# Créer le graphique
plt.figure(figsize=(20, 4))
plt.plot(samples)
plt.title("Onde sonore de 'money.mp3'")
plt.xlabel("Échantillons")
plt.ylabel("Amplitude")
plt.show()

np.savetxt('samples.txt', samples, fmt='%d')

# Ouvrir le fichier original et lire les lignes
with open('samples.txt', 'r') as file:
    lines = file.readlines()

# Modifier chaque ligne
modified_lines = [line.strip() + ',0,0\n' for line in lines]

# Écrire les lignes modifiées dans un nouveau fichier (ou écraser l'ancien)
with open('12280.txt', 'w') as file:
    file.writelines(modified_lines)



# Convertir les samples en bytes
raw_data = samples.tobytes()

# Créer un AudioSegment à partir des données brutes
# Remarque : Vous devrez connaître la fréquence d'échantillonnage, les canaux et la profondeur des bits de vos données originales
sample_rate = 44100  # Par exemple, 44100 Hz
channels = 2         # 1 pour mono, 2 pour stéréo
sample_width = 2     # En bytes, 2 pour 16 bits par échantillon

audio_segment = AudioSegment(
    data=raw_data,
    sample_width=sample_width,
    frame_rate=sample_rate,
    channels=channels
)

# Exporter en fichier MP3
audio_segment.export("output.mp3", format="mp3")