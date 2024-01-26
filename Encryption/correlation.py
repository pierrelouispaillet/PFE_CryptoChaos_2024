import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

osc_file = "C1c1.txt"
osc_file2 = "C2c2.txt"

data_np = np.loadtxt(osc_file, delimiter=' ', skiprows=5)
data_np2 = np.loadtxt(osc_file2, delimiter=' ', skiprows=5)

time = data_np[:, 0]  # time column
ampl = data_np[:, 1]  # amplitude column

time2 = data_np2[:, 0]  # time column
ampl2 = data_np2[:, 1]

import numpy as np

# Supposons que vos données d'amplitude soient stockées dans ampl et ampl2
# ampl = data_np[:, 1]
# ampl2 = data_np2[:, 1]

import matplotlib.pyplot as plt

def plot_difference_at_max_correlation(ampl, ampl2):
    max_corr = -1
    best_lag = 0

    # Parcourir chaque décalage possible
    for lag in range(-len(ampl2) + 1, len(ampl)):
        if lag < 0:
            shifted_ampl = ampl[-lag:]
            shifted_ampl2 = ampl2[:lag]
        else:
            shifted_ampl = ampl[:-lag] if lag != 0 else ampl
            shifted_ampl2 = ampl2[lag:]

        # Assurer que les deux signaux ont la même longueur
        min_length = min(len(shifted_ampl), len(shifted_ampl2))
        shifted_ampl = shifted_ampl[:min_length]
        shifted_ampl2 = shifted_ampl2[:min_length]

        # Calculer le coefficient de corrélation de Pearson
        if len(shifted_ampl) > 0 and len(shifted_ampl2) > 0:
            corr = np.corrcoef(shifted_ampl, shifted_ampl2)[0, 1]
            if corr > max_corr:
                max_corr = corr
                best_lag = lag

    # Aligner les séries en fonction du meilleur décalage
    if best_lag > 0:
        ampl2_aligned = np.concatenate([np.zeros(best_lag), ampl2])
        ampl2_aligned = ampl2_aligned[:len(ampl)]
    else:
        ampl_aligned = np.concatenate([np.zeros(-best_lag), ampl])
        ampl = ampl_aligned[:len(ampl2)]

    
    # Calculer la différence
    if best_lag > 0:
        ampl_meaned = mean_of_chunks(ampl, 10)
        ampl2_meaned = mean_of_chunks(ampl2_aligned, 10)
        difference = ampl_meaned - ampl2_meaned
    else:
        ampl_meaned = mean_of_chunks(ampl, 10)
        ampl2_meaned = mean_of_chunks(ampl2, 10)
        difference =   ampl_meaned - ampl2_meaned

    # Tracer la différence
    plt.figure(figsize=(10, 5))
    plt.plot(difference, label='Ampl1 - Ampl2')
    plt.title('Différence entre Ampl1 et Ampl2 au décalage maximal de corrélation de Pearson')
    plt.xlabel('Index')
    plt.ylabel('Différence d\'amplitude')
    plt.legend()
    plt.show()

    return best_lag, max_corr

def mean_of_chunks(arr, chunk_size):
    # Calculer le nombre de chunks complets
    num_full_chunks = len(arr) // chunk_size

    # Initialiser le nouveau tableau
    averaged_arr = []

    # Calculer la moyenne pour chaque chunk complet
    for i in range(num_full_chunks):
        chunk = arr[i * chunk_size:(i + 1) * chunk_size]
        averaged_arr.append(np.mean(chunk))

    # Traiter le dernier chunk incomplet, si nécessaire
    remaining_elements = len(arr) % chunk_size
    if remaining_elements != 0:
        last_chunk = arr[-remaining_elements:]
        averaged_arr.append(np.mean(last_chunk))

    return np.array(averaged_arr)

# Appliquer la moyennisation sur les tableaux d'amplitude


def find_max_correlation_and_plot(ampl, ampl2):
    # Calcul de la corrélation croisée
    corr = np.correlate(ampl, ampl2, mode='full')
    
    # Recherche de l'indice de corrélation maximale
    max_corr_index = np.argmax(corr)
    
    # Calcul du décalage
    lag = max_corr_index - (len(ampl) - 1)
    

    plt.figure(figsize=(10, 5))
    plt.plot(range(-len(ampl) + 1, len(ampl2)), corr, label='Corrélation')
    plt.title('Corrélation en fonction du décalage')
    plt.xlabel('Décalage')
    plt.ylabel('Corrélation')
    plt.legend()
    #plt.show()

    return lag, corr[max_corr_index]

# Utiliser la fonction modifiée
lag, max_corr = find_max_correlation_and_plot(ampl, ampl2)


print("Le décalage pour lequel la corrélation est maximale est :", lag)
print("La valeur de la corrélation maximale est :", max_corr)

def apply_lag_and_save_with_time(time, ampl, time2, ampl2, lag, filename1, filename2):
    if lag > 0:
        # Décalage de ampl2 vers la droite
        ampl2_lagged = np.concatenate([np.zeros(lag), ampl2])
        time2_lagged = np.concatenate([time2[:lag], time2])  # Ajuster le temps en conséquence
        ampl2_lagged = ampl2_lagged[:len(ampl)]
        time2_lagged = time2_lagged[:len(time)]
    else:
        # Décalage de ampl vers la droite
        ampl_lagged = np.concatenate([np.zeros(-lag), ampl])
        time_lagged = np.concatenate([time[:-lag], time])  # Ajuster le temps en conséquence
        ampl = ampl_lagged[:len(ampl2)]
        time = time_lagged[:len(time2)]

    # Préparation des données à enregistrer
    data1 = np.column_stack((time, ampl))
    if lag > 0:
        data2 = np.column_stack((time2_lagged, ampl2_lagged))
    else:
        data2 = np.column_stack((time2, ampl2))

    # Enregistrer les données décalées
    np.savetxt(filename1, data1, fmt='%.10f %.10f')
    np.savetxt(filename2, data2, fmt='%.10f %.10f')



ampl_centered = ampl - np.mean(ampl)
ampl2_centered = ampl2 - np.mean(ampl2)

# Normaliser ampl et ampl2 après le centrage
ampl_normalized = ampl_centered / np.max(np.abs(ampl_centered))
ampl2_normalized = ampl2_centered / np.max(np.abs(ampl2_centered))

#plot_difference_at_max_correlation(ampl_normalized, ampl2_normalized)
best_lag, max_corr = plot_difference_at_max_correlation(ampl_normalized, ampl2_normalized)
print("Décalage pour lequel la corrélation de Pearson est maximale :", best_lag)
print("Corrélation de Pearson maximale :", max_corr)





plt.figure(figsize=(12, 6))

# Tracer ampl
plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, position 1
plt.plot(time, ampl_normalized, label='Ampl1')
plt.title('Ampl1')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()

# Tracer ampl2
plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, position 2
plt.plot(time2, ampl2_normalized, label='Ampl2')
plt.title('Ampl2')
plt.xlabel('Temps')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
#plt.show()
# Appliquer le décalage et enregistrer dans de nouveaux fichiers
#apply_lag_and_save_with_time(time, ampl, time2, ampl2, lag, 'Chaos1_modified.txt', 'Chaos2_modified.txt')


