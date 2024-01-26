import numpy as np

def binary_to_mp3(binary_data, output_file_path):
    with open(output_file_path, 'wb') as file:
        for binary_line in binary_data:
            mp3_data = int(binary_line, 2).to_bytes((len(binary_line) + 7) // 8, byteorder='big')
            file.write(mp3_data)

# Charger les données binaires, en s'assurant qu'elles sont toujours dans un tableau 1D
signal = np.loadtxt("binarySend.txt", dtype=str, ndmin=1)

output_file_path = 'moneyLoad.mp3'
binary_to_mp3(signal, output_file_path)
