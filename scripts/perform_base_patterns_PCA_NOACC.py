import sys
import os
sys.path.append(os.path.join(os.getcwd()))
from imports.training_data import generate_training_data, get_carlson_stats_of_perturbations,  generate_base_patterns
from imports.pca_dfa import perform_PCA_patterns_analysis

# generamos los patrones base
print(f'Generando patrones base...')
base_patterns = generate_base_patterns(num_of_bases=1500)

# los perturbamos en 5 y 10 milisegundos y sacamos sus estadísticas
print(f'Generando perturbaciones 5ms...')
vibrated_patterns_5ms_stats = get_carlson_stats_of_perturbations(
    new_patterns=generate_training_data(og_patterns=base_patterns, max_half_range=5, num_patterns=1)
)
print(f'Generando perturbaciones 10ms...')
vibrated_patterns_10ms_stats = get_carlson_stats_of_perturbations(
    new_patterns=generate_training_data(og_patterns=base_patterns, max_half_range=10, num_patterns=1)
)

# efectuamos PCA sobre cada uno
data, labels = [], []
for l, list_of_dicts_of_data in vibrated_patterns_5ms_stats.items():
    if l == 'acceleration':
        continue
    for d_of_data in list_of_dicts_of_data:
        labels.append(l)
        data.append([d_of_data[k] for k in ['Sf', 'Pf', 'Ef', 'R2f', 'R1f', 'F1f', 'F2f', 'R2d', 'R1d', 'F1d', 'F2d', 'St', 'Et', 'R2t', 'R1t', 'F1t', 'F2t', 'A1', 'A2', 'A3']])
perform_PCA_patterns_analysis(data=data, labels=labels, graph_label='_vibracion_5ms_NOACC')
data, labels = [], []
for l, list_of_dicts_of_data in vibrated_patterns_10ms_stats.items():
    if l == 'acceleration':
        continue
    for d_of_data in list_of_dicts_of_data:
        labels.append(l)
        data.append([d_of_data[k] for k in ['Sf', 'Pf', 'Ef', 'R2f', 'R1f', 'F1f', 'F2f', 'R2d', 'R1d', 'F1d', 'F2d', 'St', 'Et', 'R2t', 'R1t', 'F1t', 'F2t', 'A1', 'A2', 'A3']])
perform_PCA_patterns_analysis(data=data, labels=labels, graph_label='_vibracion_10ms_NOACC')