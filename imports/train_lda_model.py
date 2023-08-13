################################################################################
# Módulo train_lda_model
#
# Author: Alejandro Vázquez Huerta
# Descripción: Este módulo originariamente solo implementaba LDA. Actualmente,
#              implementa las clases, el entrenamiento y la aplicación de las
#              metodologías de detección y clasificación de SPIs en una
#              secuencia temporal de pulsos.
################################################################################

import os
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis , LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from elephant.spike_train_dissimilarity import victor_purpura_distance, van_rossum_distance, SpikeTrain
from numpy import array, arange, cov, linspace, mean, dot, ndarray, where, interp, diff, inf, std
from numpy.linalg import inv
from imports.support.utils import Color, PATTERNS, DEFINED_DISTANCES, miliseconds_to_seconds
from imports.training_data import generate_base_patterns, generate_training_data, get_carlson_stats_of_perturbations, get_center_of_base_patterns
from joblib import dump
import quantities as pq
from random import choices
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

################################################################################
# MACROS
################################################################################

# Cantidad de patrones test a generar
NUM_TEST_PATTERNS = 500

# Vibraciones a evaluar
VIBRATIONS = [5]

# cantidades de parametros a considerar en entrenamiento
NUM_TRAINING_PATTERNS_PER_TYPE = [5, 10, 60, 200]

# Cantidad de patrones de la base de datos para LDA
NUM_DB_PATTERNS_PER_TYPE = 500

# numero de hiperparametros para el estimador EN DISTANCIAS
NUM_BASE_PATTERNS_PER_TYPE = 1

# numero de patrones base de cada tipo pra el test EN DISTANCIAS
NUM_BASES_PER_TYPE_FOR_TEST = 50

################################################################################
# CLASES
################################################################################

#
# Clase de estimadores por distancia en general
#
class DistanceClassificator(BaseEstimator):

    def __init__(
        self,
        temporal_param: float = 0.01,
        distance: str = 'VP'
    ):
        self.temporal_param = temporal_param
        self.distance = distance
    
    def fit(self, X, y):
        # validamos la longitud de ambos y que haya el numero correcto de patrones base
        if len(X) != len(y):
            print(f'Error: longitudes de datos y labels no es la misma')
            return None
        # validamos el tipo de y
        if any(type(y_) != str for y_ in y):
            return None
        # guardamos las clases
        self.classes_ = unique_labels(y)
        # guardamos los datos
        self.X_ = [SpikeTrain(x, units='ms', t_stop = max(x)) for x in X]
        self.y_ = y
        return self
    
    def predict(self, X):
        # verificamos que estamos fit
        check_is_fitted(self)
        # verificamos el tipo de distancia
        if self.distance not in DEFINED_DISTANCES:
            print(f'Distancia {self.distance} no soportada')
            return None
        # convertimos a X en un SpikeTrain
        X_normalized = [SpikeTrain(x, units='ms', t_stop = max(x)) for x in X]
        # la clasificacion la hacemos mirando los dicotomicos
        return_list = []
        if self.distance == 'VP':
            for x in X_normalized:
                min_distance = inf
                min_type = None
                for i, x_ in zip(range(len(self.X_)), self.X_):
                    vp_distance = victor_purpura_distance(
                        [x, x_],
                        cost_factor=self.temporal_param/pq.ms,
                        sort=False
                    )[0,1]
                    if vp_distance < min_distance:
                        min_distance = vp_distance
                        min_type = self.y_[i]
                return_list.append(min_type)
        else:
            for x in X_normalized:
                min_distance = inf
                min_type = None
                for i, x_ in zip(range(len(self.X_)), self.X_):
                    vr_distance = van_rossum_distance(
                        [x, x_],
                        time_constant=self.temporal_param*pq.ms,
                        sort=False
                    )[0,1]
                    if vr_distance < min_distance:
                        min_distance = vr_distance
                        min_type = self.y_[i]
                return_list.append(min_type)
        
        return return_list
    
    # el metodo score no va a funcionar perfecto ya que puede dar no reconocido
    def score(self, X, y):
        # predecimos los valores de X
        y_pred = self.predict(X=X)

        # los contrastamos
        num_scored, num_total = 0, 0
        for y_, y_pred_ in zip(y, y_pred):
            if y_ == y_pred_:
                num_scored += 1
            num_total += 1
        
        return num_scored/num_total
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

#
# Clasificador Fitness
#
class FitnessClassificator(BaseEstimator):
    def __init__(self, setup_distance: int = 1000, num_points_interp: int = 50):
        self.setup_distance = setup_distance
        self.num_points_interp = num_points_interp

    def fit(self, X, y):
        # validamos la longitud de ambos y que haya el numero correcto de patrones base
        if len(X) != len(y):
            return None
        # validamos el tipo de y
        if any(type(y_) != str for y_ in y):
            return None
        # guardamos las clases
        self.classes_ = unique_labels(y)
        
        # guardamos los datos
        self.X_ = self.__normalize(X=X)
        self.y_ = y
        return self
    
    def predict(self, X):
        # verificamos que estamos fit
        check_is_fitted(self)
        # le aplicamos a X el proceso de normalizado del fit
        X = self.__normalize(X=X)
        # calculamos, para cada x en X, el x_ de X_ con el menor MSE
        Y_closest = []
        for x in X:
            min_y = None
            min_distance = inf
            for x_, y in zip(self.X_, self.y_):
                mse = ((x_ - x)**2).mean()
                if mse < min_distance:
                    min_distance = mse
                    min_y = y
            Y_closest.append(min_y)
        # devolvemos las clases de los tipos correspondientes
        return Y_closest
    
    def __normalize(self, X):
        # normalizamos los datos a longitud setup_distance
        X_spikes = [array(x)*self.setup_distance/max(x) for x in X]
        X_ipis = [[d[0]] + [d[i+1] - d[i] for i in range(len(d) - 1)] for d in X_spikes]
        # vamos a guardar sus interpolaciones derivadas
        interpolation_range = linspace(start=0, stop=self.setup_distance, num=self.num_points_interp)
        X = [interp(x=interpolation_range, xp=ds, fp=di) for ds, di in zip(X_spikes, X_ipis)]
        X = [diff(d)/diff(interpolation_range) for d in X]
        return X
    
    def score(self, X, y):
        # predecimos los valores de X
        y_pred = self.predict(X=X)

        # los contrastamos
        num_scored, num_total = 0, 0
        for y_, y_pred_ in zip(y, y_pred):
            if y_ == y_pred_:
                num_scored += 1
            num_total += 1
        
        return num_scored/num_total


################################################################################
# FUNCIONES PRIVADAS
################################################################################

#
# Funcion para obtener los coeficientes de las fronteras proyectadas en LDA, además de graficar con los datos de test
#
def __graph_lda_data(
    data_stats: ndarray,
    data_labels: list[str],
    test_stats: ndarray,
    test_labels: list[str],
    lda: LinearDiscriminantAnalysis,
    plot_label: str = ''
) -> None:
    fig, x_axis = plt.subplots()

    # la separamos por tipos
    data_class_index = {p : where(array(data_labels) == p)[0].tolist() for p in PATTERNS}
    data_stats_per_type = {p: data_stats[data_class_index[p]] for p in PATTERNS}
    data_means_per_type = {p: array([mean(data_stats_per_type[p][:,0]), mean(data_stats_per_type[p][:,1])]) for p in PATTERNS}
    test_class_index = {p : where(array(test_labels) == p)[0].tolist() for p in PATTERNS}
    test_stats_per_type = {p: test_stats[test_class_index[p]] for p in PATTERNS}

    # graficamos los datos test
    for p in PATTERNS:
        x_axis.scatter(test_stats_per_type[p][:,0], test_stats_per_type[p][:,1], label=f'{p}s', s=2)
    
    # guardamos los coeficientes de las rectas de interseccion y las graficamos
    matrix_for_lines = inv(cov([data_stats_per_type['scallop'][:,0], data_stats_per_type['scallop'][:,1]]))
    xmin, xmax = x_axis.get_xlim()
    ymin, ymax = x_axis.get_ylim()
    rectas_coefs = []
    for p_1, p_2 in [['scallop', 'acceleration'], ['scallop', 'rasp'], ['acceleration', 'rasp']]:
        # rango en el que graficar
        x_l = linspace(xmin, xmax, 100)
        # ecuacion de la recta
        A_B = dot(matrix_for_lines, (data_means_per_type[p_1] - data_means_per_type[p_2]).T)
        C_D = (data_means_per_type[p_1] + data_means_per_type[p_2]) / 2
        x_axis.plot(x_l, (-A_B[0]/A_B[1])*x_l + dot(A_B,C_D)/A_B[1], '--')
        rectas_coefs.append([(-A_B[0]/A_B[1]), dot(A_B,C_D)/A_B[1]])

    # guardamos el coeficiente de las rectas
    with open(os.path.join(os.getcwd(), f'scripts/auxiliar_results/LDA_coef_rectas_{plot_label}.dat'), 'w') as f:
        for r in rectas_coefs:
            f.write(f'{r[0]} {r[1]}\n')
    print(f'Archivo {Color.YELLOW}LDA_coef_rectas_{plot_label}.dat{Color.END} generado en scripts/auxiliar_results/')

    # completamos la grafica
    x_axis.set_ylim((ymin, ymax))
    x_axis.set_xlabel(f'Componente 1')
    x_axis.set_ylabel(f'Componente 2')
    x_axis.set_title(f'Separacion de los burst por LDA')
    x_axis.legend(fontsize=8)
    fig.savefig(os.path.join(os.getcwd(), f'graficas/proyeccion_tests_LDA_{plot_label}.jpg'), dpi=300)
    print(f'Archivo {Color.YELLOW}proyeccion_tests_LDA_{plot_label}.jpg{Color.END} generado en graficas/')
    x_axis.cla()

    # guardamos que es cada componente como combinacion de las estadisticas
    with open(os.path.join(os.getcwd(), f'scripts/auxiliar_results/pesos_componentes_LDA_{plot_label}.dat'), 'w') as f:
        f.write('# estadistica ')
        for i in range(2):
            f.write(f'# pesos componente {i+1} ')
        f.write('\n')
        for stat, peso_en_factor in zip(
            ['Sf', 'Pf', 'Ef', 'R2f', 'R1f', 'F1f', 'F2f', 'R2d', 'R1d', 'F1d', 'F2d', 'St', 'Et', 'R2t', 'R1t', 'F1t', 'F2t', 'A1', 'A2', 'A3'],
            [[round(s, 4) for s in lda.coef_[:2,j]] for j in range(lda.n_features_in_)]
        ):
            f.write(f'{stat} ')
            for peso in peso_en_factor:
                f.write(f'{peso} ')
            f.write('\n')
    print(f'Archivo {Color.YELLOW}pesos_componentes_LDA_{plot_label}.dat{Color.END} generado en scripts/auxiliar_results/')

################################################################################
# FUNCIONES DE GESTION DE DATOS
################################################################################

#
# Funcion para generar una base de datos de estadisticas
#
def generate_stats_database(data_size_per_type: int = 500) -> dict[str, list[list]]:
    database = {p : [] for p in PATTERNS}

    # generamos patrones base
    print(f'Generamos patrones de entrenamiento de cada tipo...')
    base_patterns = generate_base_patterns(num_of_bases=data_size_per_type)
    for p in PATTERNS:
        for _ in range(data_size_per_type):
            database[p].append([])
        
    # estadisticas de los datos base
    for v in VIBRATIONS:
        print(f'Sacamos sus estadisticas rizados a {v}ms...')
        vibrated_base_patterns_stats = get_carlson_stats_of_perturbations(
            new_patterns=generate_training_data(
                og_patterns=base_patterns,
                max_half_range=v,
                num_patterns=1
            )
        )
        print(f'Añadimos los stats rizados a {v}ms a la database...')
        for p in PATTERNS:
            L_p = len(database[p])
            for d, i in zip(vibrated_base_patterns_stats[p], range(L_p)):
                database[p][i].append(
                    [
                        d[k] for k in ['Sf', 'Pf', 'Ef', 'R2f', 'R1f', 'F1f', 'F2f', 'R2d', 'R1d', 'F1d', 'F2d', 'St', 'Et', 'R2t', 'R1t', 'F1t', 'F2t', 'A1', 'A2', 'A3']
                    ]
                )
    
    return database

#
# Funcion para generar una base de datos de patrones centrados
#
def generate_centered_patterns_database(data_size_per_type: int = 500) -> dict[str, list[list]]:
    database = {p : [] for p in PATTERNS}

    # generamos patrones base
    print(f'Generamos patrones de entrenamiento de cada tipo...')
    base_patterns = generate_base_patterns(num_of_bases=data_size_per_type)
    base_patterns = get_center_of_base_patterns(base_patterns=base_patterns)
    base_patterns = {p : [[s - sp[0] for s in sp] for sp in base_patterns[p]] for p in PATTERNS}
    for p in PATTERNS:
        for _ in range(data_size_per_type):
            database[p].append([])
        
    # estadisticas de los datos base
    for v in VIBRATIONS:
        print(f'Sacamos sus rizados a {v}ms...')
        test_patterns = generate_training_data(
            og_patterns=base_patterns,
            max_half_range=v,
            num_patterns=1
        )
        test_patterns = {p : [[s - sp[0] for s in sp[:-1]] for sp in test_patterns[p]] for p in PATTERNS}
        
        print(f'Añadimos los rizados a {v}ms a la database...')
        for p in PATTERNS:
            L_p = len(database[p])
            for SPIs, i in zip(test_patterns[p], range(L_p)):
                database[p][i].append(SPIs)
    
    return database

################################################################################
# FUNCIONES DEL MODULO LDA
################################################################################

def train_lda_model() -> list[LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis]:
    print(f'Sacamos la base de datos...')
    database = generate_stats_database(data_size_per_type=NUM_DB_PATTERNS_PER_TYPE)

    # vamos a efectuar RSF-fold 
    test_results_LDA = {num_t : {v : 0 for v in VIBRATIONS} for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    test_results_QDA = {num_t : {v : 0 for v in VIBRATIONS} for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    reg_param_LDA_results = {num_t : {v : 0 for v in VIBRATIONS} for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    reg_param_QDA_results = {num_t : {v : 0 for v in VIBRATIONS} for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    # para cada uno de estos patrones de entrenamiento y test de cada tipo y vibracion
    for num_p in NUM_TRAINING_PATTERNS_PER_TYPE:
        for vibration, vibration_idx in zip(VIBRATIONS, range(len(VIBRATIONS))):
            print(f'{Color.GREEN}{num_p} de entrenamiento, vibracion {vibration}ms{Color.END}')
            # el conjunto total de indices de la base de datos
            indexes_for_splitting = choices(range(len(database['scallop'])), k=num_p)
            y = ['scallop'] * len(indexes_for_splitting)

            # los partimos
            k_folder = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
            n_splits = k_folder.get_n_splits()
            for i, (train_index, test_index) in enumerate(k_folder.split(indexes_for_splitting, y)):
                print(f"Iteracion {i}")

                # formateamos los conjuntos de entrenamiento y test
                training_data = []
                training_labels = []
                test_data = []
                test_labels = []
                for p in PATTERNS:
                    # sacamos los de training
                    for data in [database[p][i] for i in train_index]:
                        training_data.append(data[vibration_idx])
                        training_labels.append(p)
                    # sacamos los de training
                    for data in [database[p][i] for i in test_index]:
                        test_data.append(data[vibration_idx])
                        test_labels.append(p)
                
                # entrenamos LDA
                print(f'Entrenando LDA...')
                search_lda = GridSearchCV(
                    estimator=LinearDiscriminantAnalysis(solver='lsqr', n_components = 2),
                    param_grid={'shrinkage' : arange(0, 1, 0.01)},
                    scoring='accuracy',
                    cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=10),
                    n_jobs=-1,
                    error_score='raise',
                ).fit(training_data, training_labels) # lda
                lda = search_lda.best_estimator_.fit(training_data, training_labels)

                # entrenamos QDA
                print(f'Entrenando QDA...')
                search_qda = GridSearchCV(
                    estimator=QuadraticDiscriminantAnalysis(),
                    param_grid={'reg_param' : arange(0, 1, 0.01)},
                    scoring='accuracy',
                    cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=10),
                    n_jobs=-1,
                    error_score='raise'
                ).fit(training_data, training_labels) # qda
                qda = search_qda.best_estimator_.fit(training_data, training_labels)

                # vamos sumando estos resultados
                test_results_LDA[num_p][vibration] += lda.score(test_data, test_labels)*100
                test_results_QDA[num_p][vibration] += qda.score(test_data, test_labels)*100
                reg_param_LDA_results[num_p][vibration] += search_lda.best_params_['shrinkage']
                reg_param_QDA_results[num_p][vibration] += search_qda.best_params_['reg_param']

            # estandarizamos todos
            test_results_LDA[num_p][vibration] /= n_splits
            test_results_QDA[num_p][vibration] /= n_splits
            reg_param_LDA_results[num_p][vibration] /= n_splits
            reg_param_QDA_results[num_p][vibration] /= n_splits
            print(f'Exito LDA {num_p} patrones {vibration}ms: {Color.CYAN}{test_results_LDA[num_p][vibration]}{Color.END}%')
            print(f'Exito QDA {num_p} patrones {vibration}ms: {Color.CYAN}{test_results_QDA[num_p][vibration]}{Color.END}%')
            print(f'Parametro LDA {num_p} patrones {vibration}ms: {Color.CYAN}{reg_param_LDA_results[num_p][vibration]}{Color.END}%')
            print(f'Parametro QDA {num_p} patrones {vibration}ms: {Color.CYAN}{reg_param_QDA_results[num_p][vibration]}{Color.END}%')
    
    # grafica vibracion_entrenamiento frente a vibracion_test para cada numero de parametros
    print(f'Graficamos...')
    for results, model_name in zip(
        [test_results_LDA, test_results_QDA, reg_param_LDA_results, reg_param_QDA_results],
        ['LDA tasa de exito', 'QDA tasa de exito', 'Parametro de regularizacion LDA', 'Parametro de regularizacion QDA']
    ):
        fig, ax = plt.subplots()
        ax.set_xlabel(f'Rizado de los patrones (ms)')
        ax.set_ylabel(model_name)
        for num_t in NUM_TRAINING_PATTERNS_PER_TYPE:
            ax.plot(VIBRATIONS, [results[num_t][v] for v in VIBRATIONS], '--o', label=f'{num_t} de entrenamiento')
        ax.legend(fontsize=8)
        fig.savefig(os.path.join(os.getcwd(), f'scripts/auxiliar_img/{model_name}.jpg'), dpi=300)
        print(f'Archivo {Color.YELLOW}{model_name}.jpg.pdf{Color.END} generado en scripts/auxiliar_img/')
        ax.cla()
        plt.close(fig=fig)
    
    # guardamos un LDA y un QDA con el maximo de patrones y vibracion 5
    for _, (train_index, test_index) in enumerate(
        RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=1
        ).split(choices(range(len(database['scallop'])), k=200))
    ):
        break
    training_data = []
    training_labels = []
    test_data = []
    test_labels = []
    for p in PATTERNS:
        # sacamos los de training
        for data in [database[p][i] for i in train_index]:
            training_data.append(data[vibration_idx])
            training_labels.append(p)
        # sacamos los de training
        for data in [database[p][i] for i in test_index]:
            test_data.append(data[vibration_idx])
            test_labels.append(p)
    
    # entrenamos los modelos LDA
    print(f'Entrenando LDA final...')
    search_lda = GridSearchCV(
        estimator=LinearDiscriminantAnalysis(solver='eigen', n_components = 2),
        param_grid={'shrinkage' : arange(0, 1, 0.01)},
        scoring='accuracy',
        cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=10),
        n_jobs=-1,
        error_score='raise'
    ).fit(training_data, training_labels) # qda
    lda = search_lda.best_estimator_.fit(training_data, training_labels)

    # entrenamos QDA
    print(f'Entrenando QDA final...')
    search_qda = GridSearchCV(
        estimator=QuadraticDiscriminantAnalysis(),
        param_grid={'reg_param' : arange(0, 1, 0.01)},
        scoring='accuracy',
        cv=RepeatedStratifiedKFold(n_splits=4, n_repeats=10),
        n_jobs=-1,
        error_score='raise'
    ).fit(training_data, training_labels) # qda
    qda = search_qda.best_estimator_.fit(training_data, training_labels)

    # lo guardamos
    dump(lda, os.path.join(os.getcwd(), f'resultados/classificator_LDA.jl'))
    dump(qda, os.path.join(os.getcwd(), f'resultados/classificator_QDA.jl'))
    print(f'Archivo {Color.YELLOW}classificator_LDA.jl{Color.END} generado en resultados/')
    print(f'Archivo {Color.YELLOW}classificator_QDA.jl{Color.END} generado en resultados/')

    __graph_lda_data(
        data_stats=training_data,
        test_stats=test_data,
        data_labels=training_labels,
        test_labels=test_labels,
        lda=lda
    )

    return lda, qda

#
# Funcion para graficar la proyeccion LDA
#
def graph_LDA_projections(training_amounts_per_type: list[int] = [200], plot_labels: list[str] = [''], reg_params: list[float] = [0]) -> None:
    print(f'Sacamos la base de datos...')
    num_patrones_test = 500
    database = generate_stats_database(data_size_per_type=max(training_amounts_per_type) + num_patrones_test)

    # guardamos un LDA y un QDA con el maximo de patrones y vibracion 5
    for tr_amount, plt_label, shr in zip(training_amounts_per_type, plot_labels, reg_params):
        # sacamos los idx de los patrones seleccionados
        r = list(range(len(database['scallop'])))
        patrones_entrenamiento_idx = list(set(choices(r, k=tr_amount)))
        # sacamos los idx de los patrones test
        for pe_idx in patrones_entrenamiento_idx:
            if pe_idx not in r:
                print(f'INDICE DE ERROR: {pe_idx}, {patrones_entrenamiento_idx}')
            r.remove(pe_idx)
        patrones_test_idx = choices(r, k=num_patrones_test)
        # sacamos los patrones de cada tipo
        training_data = []
        training_labels = []
        test_data = []
        test_labels = []
        for p in PATTERNS:
            # sacamos los de training
            for data in [database[p][i] for i in patrones_entrenamiento_idx]:
                training_data.append(data[0])
                training_labels.append(p)
            # sacamos los de training
            for data in [database[p][i] for i in patrones_test_idx]:
                test_data.append(data[0])
                test_labels.append(p)
        
        # entrenamos los modelos LDA
        print(f'Entrenando LDA para graficado...')
        lda=LinearDiscriminantAnalysis(solver='eigen', n_components = 2, shrinkage=shr).fit(training_data, training_labels)

        # graficamos
        training_data = lda.transform(training_data)
        test_data = lda.transform(test_data)
        print(f'Graficando proyeccion...')
        __graph_lda_data(
            data_stats=array(training_data),
            test_stats=array(test_data),
            data_labels=training_labels,
            test_labels=test_labels,
            lda=lda,
            plot_label=plt_label
        )

    return


################################################################################
# FUNCIONES DEL MODULO FM
################################################################################

#
# Funcion de entrenamiento del modelo
#
def train_fitness_model(setup_distance: int = 100, num_points_interp: int = 50) -> FitnessClassificator:
    print(f'Sacamos la base de datos...')
    database = generate_centered_patterns_database(data_size_per_type=NUM_DB_PATTERNS_PER_TYPE)

    # probamos para diferentes patrones de entrenamiento
    results_reg = {num_t : {v : 0 for v in VIBRATIONS} for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    # para cada uno de estos patrones de entrenamiento y test de cada tipo y vibracion
    for num_p in NUM_TRAINING_PATTERNS_PER_TYPE:
        for vibration, vibration_idx in zip(VIBRATIONS, range(len(VIBRATIONS))):
            print(f'{Color.GREEN}{num_p} de entrenamiento, vibracion {vibration}ms{Color.END}')
            # el conjunto total de indices de la base de datos
            indexes_for_splitting = choices(range(len(database['scallop'])), k=num_p)
            y = ['scallop'] * len(indexes_for_splitting)

            # los partimos
            k_folder = RepeatedStratifiedKFold(n_splits=5, n_repeats=20)
            n_splits = k_folder.get_n_splits()
            for i, (train_index, test_index) in enumerate(k_folder.split(indexes_for_splitting, y)):
                print(f"Iteracion {i}")

                # formateamos los conjuntos de entrenamiento y test
                training_spikes = []
                training_labels = []
                test_spikes = []
                test_labels = []
                for p in PATTERNS:
                    # sacamos los de training
                    for data in [database[p][i] for i in train_index]:
                        training_spikes.append(data[vibration_idx])
                        training_labels.append(p)
                    # sacamos los de training
                    for data in [database[p][i] for i in test_index]:
                        test_spikes.append(data[vibration_idx])
                        test_labels.append(p)
                
                # cargamos los patrones base en el modelo
                print(f'{num_p}: Cargando datos en el modelo vibracion {vibration}...')
                fm = FitnessClassificator(
                    setup_distance=setup_distance,
                    num_points_interp=num_points_interp
                ).fit(training_spikes, training_labels)

                # lo probamos y guardamos
                results_reg[num_p][vibration] += fm.score(test_spikes, test_labels) * 100
                
            # lo guardamos
            results_reg[num_p][vibration] /= n_splits
            print(f'Exito FM {num_p} patrones {vibration}ms: {Color.CYAN}{results_reg[num_p][vibration]}{Color.END}%')
    
    # lo graficamos
    print(f'Graficamos...')
    fig, ax = plt.subplots()
    ax.set_xlabel(f'Rizado de los patrones (ms)')
    ax.set_ylabel(f'porcentaje de exito en test')
    ax.set_title(f'Resultados exito')
    # grafica vibracion_entrenamiento frente a vibracion_test para cada numero de parametros
    for num_t in NUM_TRAINING_PATTERNS_PER_TYPE:
        ax.plot(VIBRATIONS, [results_reg[num_t][v] for v in VIBRATIONS], '--o', label = f'{num_t} por tipo')
    ax.legend(fontsize=7)
    fig.savefig(f'scripts/auxiliar_img/graficas_FM.jpg', dpi=300)
    print(f'Archivo {Color.YELLOW}graficas_FM.jpg{Color.END} generado en scripts/auxiliar_img/')

    # guardamos los modelos con 400 patrones a 10ms y graficamos
    indexes_for_splitting = choices(range(len(database['scallop'])), k=20)
    data_spikes = []
    data_labels = []
    for p in PATTERNS:
        # sacamos los de training
        for data in [database[p][i] for i in indexes_for_splitting]:
            training_spikes.append(data[VIBRATIONS.index(5)])
            training_labels.append(p)
    fm = FitnessClassificator(
        setup_distance=setup_distance,
        num_points_interp=num_points_interp
    ).fit(data_spikes, data_labels)
    # lo guardamos
    dump(fm, os.path.join(os.getcwd(), f'resultados/classificator_FM.jl'))
    print(f'Archivo {Color.YELLOW}classificator_FM.jl{Color.END} generado en resultados/')
    
    return fm

################################################################################
# FUNCIONES DE DISTANCIAS
################################################################################

def distance_discrimination_analysis(num_patterns_for_matrix: int = 30) -> None:
    print(f'Sacamos la base de datos...')
    database = generate_centered_patterns_database(data_size_per_type=num_patterns_for_matrix)

    # convertimos en spiketrains
    print(f'Formateando a SpikeTrains 5ms...')
    database = {p : [SpikeTrain(times=sp[0], t_stop=max(sp[0]), units='ms') for sp in database[p]] for p in PATTERNS}

    # parametros que probar
    q_params_to_try_VP = linspace(0, 0.05, 20)
    q_params_to_try_VR = linspace(0, 10000, 20)

    # para cada coste vamos a hacer esto:
    VP_discriminant_result = []
    for q in q_params_to_try_VP:
        # sacamos la matriz de distancias
        VP_matrix_means = {
            ('scallop', 'scallop') : 0,
            ('scallop', 'acceleration') : 0,
            ('scallop', 'rasp') : 0,
            ('acceleration', 'acceleration') : 0,
            ('acceleration', 'rasp') : 0,
            ('rasp', 'rasp') : 0
        }
        VP_matrix_covariances = {
            ('scallop', 'scallop') : 0,
            ('scallop', 'acceleration') : 0,
            ('scallop', 'rasp') : 0,
            ('acceleration', 'acceleration') : 0,
            ('acceleration', 'rasp') : 0,
            ('rasp', 'rasp') : 0
        }
        # sacamos las diagonales de la matriz de distancias
        print(f'(q={q}) Sacamos las diagonales de la matriz de distancias...')
        for p in PATTERNS:
            list_of_VP_distances = []
            for Sp1, Sp1_idx in zip(database[p], range(len(database[p]))):
                if Sp1_idx+1 < len(database[p]):
                    for Sp2 in database[p][Sp1_idx+1:]:
                        list_of_VP_distances.append(victor_purpura_distance(
                            spiketrains=[Sp1, Sp2],
                            cost_factor=q/pq.ms,
                            sort=False
                        )[0, 1])
            
            VP_matrix_means[(p, p)] = mean(list_of_VP_distances)
            VP_matrix_covariances[(p, p)] = std(list_of_VP_distances)
            
        
        # sacamos las no diagonales
        print(f'(q={q}) Sacamos la superior de la matriz de  de distancias...')
        for p1, i in zip(PATTERNS, range(len(PATTERNS))):
            if i+1 < len(PATTERNS):
                for p2 in PATTERNS[i+1:]:
                    list_of_VP_distances = []
                    for Sp1 in database[p1]:
                        for Sp2 in database[p2]:
                            list_of_VP_distances.append(victor_purpura_distance(
                                spiketrains=[Sp1, Sp2],
                                cost_factor=q/pq.ms,
                                sort=False
                            )[0, 1])
                    VP_matrix_means[(p1, p2)] = mean(list_of_VP_distances)
                    VP_matrix_covariances[(p1, p2)] = std(list_of_VP_distances)
        
        # imprimimos los resultados:
        print(f'(q={q}) Matriz de distancias VP:')
        for p1, i in zip(PATTERNS, range(len(PATTERNS))):
            things_to_print = ''
            for p2 in PATTERNS[i:]:
                things_to_print += f'{round(VP_matrix_means[(p1, p2)], 4)} +- {round(VP_matrix_covariances[(p, p)], 4)}'.ljust(20, ' ')
            print(' ' * 20 * i + things_to_print)
        
        # sacamos el discriminante total
        D_VP = 0
        for p1, i in zip(PATTERNS, range(len(PATTERNS))):
            if i+1 < len(PATTERNS):
                for p2 in PATTERNS[i+1:]:
                    D_VP += VP_matrix_means[(p1, p2)]
        for p in PATTERNS:
            D_VP -= VP_matrix_means[(p, p)]
        print(f'(q={q}) Discriminante VP: {D_VP}')

        # los guardamos
        VP_discriminant_result.append(D_VP)
    
    # ahora para VR
    VR_discriminant_result = []
    for q in q_params_to_try_VR:
        VR_matrix_means = {
            ('scallop', 'scallop') : 0,
            ('scallop', 'acceleration') : 0,
            ('scallop', 'rasp') : 0,
            ('acceleration', 'acceleration') : 0,
            ('acceleration', 'rasp') : 0,
            ('rasp', 'rasp') : 0
        }
        VR_matrix_covariances = {
            ('scallop', 'scallop') : 0,
            ('scallop', 'acceleration') : 0,
            ('scallop', 'rasp') : 0,
            ('acceleration', 'acceleration') : 0,
            ('acceleration', 'rasp') : 0,
            ('rasp', 'rasp') : 0
        }

        # sacamos las diagonales de la matriz de distancias
        print(f'(q={q}) Sacamos las diagonales de la matriz de distancias...')
        for p in PATTERNS:
            list_of_VR_distances = []
            for Sp1, Sp1_idx in zip(database[p], range(len(database[p]))):
                if Sp1_idx+1 < len(database[p]):
                    for Sp2 in database[p][Sp1_idx+1:]:
                        list_of_VR_distances.append(van_rossum_distance(
                            spiketrains=[Sp1, Sp2],
                            time_constant=q*pq.ms,
                            sort=False
                        )[0, 1])
            VR_matrix_means[(p, p)] = mean(list_of_VR_distances)
            VR_matrix_covariances[(p, p)] = std(list_of_VR_distances)
        
        # sacamos las no diagonales
        print(f'(q={q}) Sacamos la superior de la matriz de  de distancias...')
        for p1, i in zip(PATTERNS, range(len(PATTERNS))):
            if i+1 < len(PATTERNS):
                for p2 in PATTERNS[i+1:]:
                    list_of_VR_distances = []
                    for Sp1 in database[p1]:
                        for Sp2 in database[p2]:
                            list_of_VR_distances.append(van_rossum_distance(
                                spiketrains=[Sp1, Sp2],
                                time_constant=q*pq.ms,
                                sort=False
                            )[0, 1])
                    VR_matrix_means[(p1, p2)] = mean(list_of_VR_distances)
                    VR_matrix_covariances[(p1, p2)] = std(list_of_VR_distances)
        
        print(f'(q={q}) Matriz de distancias VR:')
        for p1, i in zip(PATTERNS, range(len(PATTERNS))):
            things_to_print = ''
            for p2 in PATTERNS[i:]:
                things_to_print += f'{round(VR_matrix_means[(p1, p2)], 4)} +- {round(VR_matrix_covariances[(p, p)], 4)} '.ljust(20, ' ')
            print(' ' * 20 * i + things_to_print)
        
        # sacamos el discriminante total
        D_VR = 0
        for p1, i in zip(PATTERNS, range(len(PATTERNS))):
            if i+1 < len(PATTERNS):
                for p2 in PATTERNS[i+1:]:
                    D_VR += VR_matrix_means[(p1, p2)]
        for p in PATTERNS:
            D_VR -= VR_matrix_means[(p, p)]
        print(f'(q={q}) Discriminante VR: {D_VR}')

        # los guardamos
        VR_discriminant_result.append(D_VR)

    # graficamos VP
    print(f'Graficamos...')
    fig, ax = plt.subplots()
    ax.set_xlabel(f'q (kHz)')
    ax.set_ylabel(f'Valores del discriminante')
    ax.set_title(f'Resultados del discriminante Victor Purpura')
    # grafica vibracion_entrenamiento frente a vibracion_test para cada numero de parametros
    ax.plot(q_params_to_try_VP, VP_discriminant_result, '--o')
    fig.savefig(f'scripts/auxiliar_img/graficas_discriminantes_VP.jpg', dpi=300)
    print(f'Archivo {Color.YELLOW}graficas_discriminantes_VP.jpg{Color.END} generado en scripts/auxiliar_img/')
    # graficamos ahora VR
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\tau_{r}$' + f' (ms)')
    ax.set_ylabel(f'Valores del discriminante')
    ax.set_title(f'Resultados del discriminante van Rossum')
    # grafica vibracion_entrenamiento frente a vibracion_test para cada numero de parametros
    ax.plot(q_params_to_try_VR, VR_discriminant_result, '--o')
    fig.savefig(f'scripts/auxiliar_img/graficas_discriminantes_VR.jpg', dpi=300)
    print(f'Archivo {Color.YELLOW}graficas_discriminantes_VR.jpg{Color.END} generado en scripts/auxiliar_img/')

    return

#
# Funcion de entrenamiento de los modelos
#
def train_distances_model(vp_parameter: float = 0.01, vr_parameter: float = 1000, graph_database: bool = False) -> list[DistanceClassificator]:
    print(f'Sacamos la base de datos...')
    database = generate_centered_patterns_database(data_size_per_type=NUM_DB_PATTERNS_PER_TYPE)

    # la graficamos
    if graph_database:
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        pdf_debug_file = PdfPages(f'scripts/auxiliar_img/database_Distancias.pdf')
        for p in PATTERNS:
            fig, x_axis = plt.subplots()
            x_axis.set_xlabel('Tiempo (s)')
            x_axis.set_ylabel('IPI (ms)')
            x_axis.xaxis.set_major_formatter(miliseconds_to_seconds)
            # ajuste de dimensiones
            if p == 'scallop':
                x_axis.set_xlim(left=420, right=1600)
                x_axis.set_ylim(bottom=0, top=200)
            elif p == 'acceleration':
                x_axis.set_xlim(left=420, right=2500)
            else:
                x_axis.set_xlim(left=420, right=2000)
            # graficamos
            for pattern in database[p]:
                pattern = pattern[0]
                ipis_base = [pattern[0]] + [pattern[i+1] - pattern[i] for i in range(len(pattern)-1)]
                x_axis.plot(pattern, ipis_base, alpha=0.3, color='blue')
            # guardamos
            fig.savefig(pdf_debug_file, format='pdf')
            x_axis.cla()
            plt.close(fig=fig)
        # guardamos
        pdf_debug_file.close()
        print(f'Archivo {Color.YELLOW}database_Distancias.pdf{Color.END} generado en scripts/auxiliar_img/')

    # probamos para diferentes patrones de entrenamiento
    results_vp = {num_t : 0 for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    results_vr = {num_t : 0 for num_t in NUM_TRAINING_PATTERNS_PER_TYPE}
    # para cada uno de estos patrones de entrenamiento y test de cada tipo y vibracion
    for num_p in NUM_TRAINING_PATTERNS_PER_TYPE:
        print(f'{Color.GREEN}{num_p} de entrenamiento{Color.END}')
        # el conjunto total de indices de la base de datos
        indexes_for_splitting = choices(range(len(database['scallop'])), k=num_p)
        y = ['scallop'] * len(indexes_for_splitting)

        # los partimos
        k_folder = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
        n_splits = k_folder.get_n_splits()
        for i, (train_index, test_index) in enumerate(k_folder.split(indexes_for_splitting, y)):
            print(f"Iteracion {i}")

            # formateamos los conjuntos de entrenamiento y test
            training_spikes = []
            training_labels = []
            test_spikes = []
            test_labels = []
            for p in PATTERNS:
                # sacamos los de training
                for data in [database[p][i] for i in train_index]:
                    training_spikes.append(data[VIBRATIONS.index(5)])
                    training_labels.append(p)
                # sacamos los de training
                for data in [database[p][i] for i in test_index]:
                    test_spikes.append(data[VIBRATIONS.index(5)])
                    test_labels.append(p)
            
            # cargamos los patrones base en el modelo
            print(f'{num_p}: Cargando datos en el modelo VP...')
            vp_classificator = DistanceClassificator(
                temporal_param=vp_parameter,
                distance='VP'
            ).fit(training_spikes, training_labels)
            print(f'{num_p}: Cargando datos en el modelo VP...')
            vr_classificator = DistanceClassificator(
                temporal_param=vr_parameter,
                distance='VR'
            ).fit(training_spikes, training_labels)

            # lo probamos y guardamos
            results_vp[num_p] += vp_classificator.score(test_spikes, test_labels) * 100
            results_vr[num_p] += vr_classificator.score(test_spikes, test_labels) * 100
            
        # lo guardamos
        results_vp[num_p] /= n_splits
        results_vr[num_p] /= n_splits
        print(f'Exito {num_p} Victor-Purpura: {Color.CYAN}{results_vp[num_p]}{Color.END}%')
        print(f'Exito {num_p} van Rossum: {Color.CYAN}{results_vr[num_p]}{Color.END}%')
    
    # lo graficamos
    print(f'Graficamos...')
    fig, ax = plt.subplots()
    ax.set_xlabel(f'Numero de patrones de entrenamiento')
    ax.set_ylabel(f'porcentaje de exito en test')
    # grafica vibracion_entrenamiento frente a vibracion_test para cada numero de parametros
    ax.plot(NUM_TRAINING_PATTERNS_PER_TYPE, [results_vp[num_t] for num_t in NUM_TRAINING_PATTERNS_PER_TYPE], '--o', label = f'Victor-Purpura')
    ax.plot(NUM_TRAINING_PATTERNS_PER_TYPE, [results_vr[num_t] for num_t in NUM_TRAINING_PATTERNS_PER_TYPE], '--o', label = f'van Rossum')
    ax.legend(fontsize=7)
    fig.savefig(f'scripts/auxiliar_img/graficas_VP_VR.jpg', dpi=300)
    print(f'Archivo {Color.YELLOW}graficas_VP_VR.jpg{Color.END} generado en scripts/auxiliar_img/')

    # guardamos los modelos con 400 patrones a 10ms y graficamos
    indexes_for_splitting = choices(range(len(database['scallop'])), k=20)
    data_spikes = []
    data_labels = []
    for p in PATTERNS:
        # sacamos los de training
        for data in [database[p][i] for i in indexes_for_splitting]:
            training_spikes.append(data[VIBRATIONS.index(5)])
            training_labels.append(p)
    print(f'{num_p}: Cargando datos en el modelo VP final...')
    vp_classificator = DistanceClassificator(
        temporal_param=vp_parameter,
        distance='VP'
    ).fit(data_spikes, data_labels)
    print(f'{num_p}: Cargando datos en el modelo VR final...')
    vr_classificator = DistanceClassificator(
        temporal_param=vr_parameter,
        distance='VR'
    ).fit(data_spikes, data_labels)
    # lo guardamos
    dump(vp_classificator, os.path.join(os.getcwd(), f'resultados/classificator_VP.jl'))
    dump(vr_classificator, os.path.join(os.getcwd(), f'resultados/classificator_VR.jl'))
    print(f'Archivo {Color.YELLOW}classificator_VP.jl{Color.END} generado en resultados/')
    print(f'Archivo {Color.YELLOW}classificator_VR.jl{Color.END} generado en resultados/')
    
    return vp_classificator, vr_classificator
