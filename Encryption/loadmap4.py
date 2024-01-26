import numpy as np

signal = np.loadtxt("signal_data.txt")

binaire = ""
condition = 1
seuil = 5

for i in range(len(signal)):
    if (signal[i] > seuil) and (condition == 1):
        binaire += '1'  # Ajouter un caractère '1' à la chaîne
        condition = 0
    elif (signal[i] < -seuil) and (condition == 1):
        binaire += '0'  # Ajouter un caractère '0' à la chaîne
        condition = 0
    elif (-seuil < signal[i] < seuil) and (condition == 0):
        condition = 1

def binary_to_mp3(binary_data, output_file_path):
    # Convertir les données binaires en bytes
    mp3_data = int(binary_data, 2).to_bytes((len(binary_data) + 7) // 8, byteorder='big')

    # Écrire les données byte dans un fichier MP3
    with open(output_file_path, 'wb') as file:
        file.write(mp3_data)


print(binaire[:20])
# Utilisez cette fonction avec le chemin vers le fichier de sortie souhaité
output_file_path = 'moneyLoad.mp3'
binary_to_mp3(binaire, output_file_path)

# ... Votre code existant ...

# Sauvegarder les données binaires dans un fichier texte
with open('binaryload.txt', 'w') as file:
    file.write(binaire)
