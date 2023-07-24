import os
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from elephant.spike_train_dissimilarity import victor_purpura_distance, van_rossum_distance, SpikeTrain
from imports.support.utils import DEFINED_DISTANCES, PATTERNS, Color, error_print
from imports.training_data import generate_base_patterns, generate_training_data, get_center_of_base_patterns
from numpy import inf, linspace
from itertools import product
from joblib import dump
import quantities as pq

################################################################################
# MACROS
################################################################################

# numero de hiperparametros para el estimador
NUM_BASE_PATTERNS_PER_TYPE = 1

# numero de patrones base de cada tipo pra el test
NUM_BASES_PER_TYPE_FOR_TEST = 50

################################################################################
# CLASES
################################################################################

#
# Clase para clasificar patrones por tipo dicotomicamente
#
class DicotomicPatternDistanceClassificator(BaseEstimator):

    def __init__(
        self,
        temporal_param: list = [0]*NUM_BASE_PATTERNS_PER_TYPE,
        threshold: float = inf,
        distance: str = 'VP'
    ):
        self.temporal_param = temporal_param
        self.threshold = threshold
        self.distance = distance
    
    def fit(self, X, y):
        # validamos la longitud de ambos y que haya el numero correcto de patrones base
        if len(X) != len(y):
            return None
        if len(X) != NUM_BASE_PATTERNS_PER_TYPE:
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
        # sacamos su set de distancias con cada uno de los patrones base
        return_list = []
        if self.distance == 'VP':
            for x in X_normalized:
                min_distance = inf
                for i, x_ in zip(range(len(self.X_)), self.X_):
                    vp_distance = victor_purpura_distance(
                        [x, x_],
                        cost_factor=self.temporal_param[i]/pq.ms,
                        sort=False
                    )[0,1]
                    if vp_distance < min_distance:
                        min_distance = vp_distance
                if min_distance < self.threshold:
                    return_list.append(self.y_[0])
                else:
                    return_list.append(f'not {self.y_[0]}')
        else:
            for x in X_normalized:
                min_distance = inf
                for i, x_ in zip(range(len(self.X_)), self.X_):
                    vp_distance = van_rossum_distance(
                        [x, x_],
                        cost_factor=self.temporal_param[i]*pq.ms,
                        sort=False
                    )[0,1]
                    if vp_distance < min_distance:
                        min_distance = vp_distance
                if min_distance < self.threshold:
                    return_list.append(self.y_[0])
                else:
                    return_list.append(f'not {self.y_[0]}')

        # devolvemos si se clasifican en este tipo o no
        return return_list
    
    def score(self, X, y):
        # predecimos los valores de X
        y_pred = self.predict(X=X)

        # los contrastamos
        num_scored, num_total = 0, 0
        for y_, y_pred_ in zip(y, y_pred):
            if y_ == self.y_[0] and y_pred_ == self.y_[0]:
                num_scored += 1
            elif y_ != self.y_[0] and y_pred_ != self.y_[0]:
                num_scored += 1
            num_total += 1
        
        return num_scored/num_total
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    
    # añadimos esta funcion para el clasificador multitipo
    def get_min_distance(self, x):
        min_distance = inf
        if self.distance == 'VP':
            for i, x_ in zip(range(len(self.X_)), self.X_):
                vp_distance = victor_purpura_distance(
                    [x, x_],
                    cost_factor=self.reg_params[i]/pq.ms,
                    sort=False
                )[0,1]
                if vp_distance < min_distance:
                    min_distance = vp_distance
        else:
            for i, x_ in zip(range(len(self.X_)), self.X_):
                vp_distance = van_rossum_distance(
                    [x, x_],
                    cost_factor=self.reg_params[i]/pq.ms,
                    sort=False
                )[0,1]
                if vp_distance < min_distance:
                    min_distance = vp_distance
        if min_distance < self.threshold:
            return min_distance
        else:
            return None
    
    # esta tambiem la añadimos para el general
    def get_param_type(self):
        return self.classes_[0]
    
    def get_threshold(self):
        return self.threshold

#
# Clase de estimadores por distancia en general
#
class DistanceClassificator(BaseEstimator):

    def __init__(
        self,
        dicotomic_classifiers: list = [DicotomicPatternDistanceClassificator()]*(NUM_BASE_PATTERNS_PER_TYPE*len(PATTERNS)),
    ):
        self.dicotomic_classifiers = dicotomic_classifiers
    
    def fit(self, X, y):
        # validamos la longitud de ambos y que haya el numero correcto de patrones base
        if len(X) != len(y):
            return None
        if len(X) != NUM_BASE_PATTERNS_PER_TYPE*len(PATTERNS):
            return None
        # validamos el tipo de y
        if any(type(y_) != str for y_ in y):
            return None
        # guardamos las clases
        self.classes_ = unique_labels(y)
        # cargamos los clasificadores
        data_per_classifier = {p : [[x, y_] for x, y_ in zip(X, y) if y_ == p] for p in PATTERNS}
        for dc in self.dicotomic_classifiers:
            for p in PATTERNS:
                if dc.get_param_type() == p:
                    dc.fit([x for x, y_ in data_per_classifier[p]], [y_ for x, y_ in data_per_classifier[p]])
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
        for x in X_normalized:
            min_distance = inf
            current_class = 'Not recognized'
            # sacamos su clasificacion por cada uno de los dicotomicos
            for dc in self.dicotomic_classifiers:
                current_distance = dc.get_min_distance(x)
                # la distancia tiene que estar en el rango del threshold y menor que la actual
                if current_distance < dc.get_threshold() and current_distance < min_distance:
                    min_distance = current_distance
                    current_class = dc.get_param_type()
            return_list.append(current_class)
        
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
    

################################################################################
# FUNCIONES
################################################################################

#
# Funcion de entrenamiento de los modelos
#
def train_distances_model(save: bool = False) -> DistanceClassificator:
    # generamos patrones base
    print(f'Generando patrones base...')
    base_patterns = generate_base_patterns(num_of_bases=NUM_BASE_PATTERNS_PER_TYPE)

    # normalizamos los patrones al 0
    print(f'Extrayendo centros y llevando a 0...')
    base_patterns = get_center_of_base_patterns(base_patterns=base_patterns)
    base_patterns = {p : [[s - sp[0] for s in sp] for sp in base_patterns[p]] for p in PATTERNS}
    # los preparamos para el clasificador y labels
    base_labels = []
    base_stats = []
    params_count = 0
    for p in PATTERNS:
        for base in base_patterns[p]:
            params_count += 1
            base_stats.append(base)
            base_labels.append(p)

    # los separamos por tipos para los clasificadores individuales
    base_patterns_per_type = {p : [b for b, y in zip(base_stats, base_labels) if y == p] for p in PATTERNS}
    L_base_patterns_per_type = {p : len(base_patterns_per_type[p]) for p in PATTERNS}
    labels_patterns_per_type = {p : [p] * L_base_patterns_per_type[p] for p in PATTERNS}

    # generamos nuestros patrones de entrenamiento
    print(f'Generando patrones de entrenamiento...')
    test_patterns = generate_training_data(
        og_patterns=get_center_of_base_patterns(
            base_patterns=generate_base_patterns(
                num_of_bases=NUM_BASES_PER_TYPE_FOR_TEST
            )
        ),
        max_half_range=10,
        num_patterns=1
    )
    test_patterns = {p : [[s - sp[0] for s in sp[:-1]] for sp in test_patterns[p]] for p in PATTERNS}
    test_labels = []
    test_stats = []
    for p in PATTERNS:
        for new in test_patterns[p]:
            test_stats.append(new)
            test_labels.append(p)
    L_test = len(test_stats)

    # conjunto de parametros a probar
    print(f'Generando conjuntos de parametros...')
    param_grid = {
        'temporal_param' : [[a for a in s] for s in product(linspace(0, 1, 20))],
        'threshold' : [20*s for s in range(1, 20)]
    }
    L_temporal_param = len(param_grid['temporal_param'])
    print(f'Cantidad de parametros a probar: {L_temporal_param}')
    L_thresholds = len(param_grid['threshold'])
    print(f'Cantidad de thresholds a probar: {L_thresholds}')

    # entrenamos Victor Purpura para cada tipo
    dicotomic_classifiers_VP = {}
    for p in PATTERNS:
        print(f'Entrenando VP para {p}...')
        search_VP_p = GridSearchCV(
            estimator=DicotomicPatternDistanceClassificator(distance='VP'),
            param_grid=param_grid,
            scoring='accuracy',
            cv=[
                ([s for s in range(0, L_base_patterns_per_type[p])],
                [x + L_base_patterns_per_type[p] for x in range(L_test)])
            ],
            n_jobs=-1,
            error_score='raise',
            verbose=3
        ).fit(
            base_patterns_per_type[p] + test_stats,
            labels_patterns_per_type[p] + test_labels
        )
        dicotomic_classifiers_VP[p] = search_VP_p.best_estimator_
        print(f'Calculando rendimiento VP para {p}...')
        print(f'Resultados VP en {p}: {round(search_VP_p.best_estimator_.fit(base_patterns_per_type[p], labels_patterns_per_type[p]).score(test_stats, test_patterns)*100, 2)}')
    
    # entrenamos van Rossum para cada tipo
    dicotomic_classifiers_VR = {}
    for p in PATTERNS:
        print(f'Entrenando VR para {p}...')
        search_VR_p = GridSearchCV(
            estimator=DicotomicPatternDistanceClassificator(distance='VR'),
            param_grid=param_grid,
            scoring='accuracy',
            cv=[
                ([s for s in range(0, L_base_patterns_per_type[p])],
                [x + L_base_patterns_per_type[p] for x in range(L_test)])
            ],
            n_jobs=-1,
            error_score='raise',
            verbose=3
        ).fit(
            base_patterns_per_type[p] + test_stats,
            labels_patterns_per_type[p] + test_patterns
        )
        dicotomic_classifiers_VR[p] = search_VR_p.best_estimator_
        print(f'Calculando rendimiento VR para {p}...')
        print(f'Resultados VR en {p}: {round(search_VR_p.best_estimator_.fit(base_patterns_per_type[p], labels_patterns_per_type[p]).score(test_stats, test_patterns)*100, 2)}')
    
    # cargamos los estimadores globales, guardamos y volvemos
    vp_multi = DistanceClassificator(
        dicotomic_classifiers=[dicotomic_classifiers_VP[p] for p in PATTERNS]
    ).fit(base_stats, base_labels)
    vr_multi = DistanceClassificator(
        dicotomic_classifiers=[dicotomic_classifiers_VR[p] for p in PATTERNS]
    ).fit(base_stats, base_labels)

    if save:
        dump(vp_multi, os.path.join(os.getcwd(), f'resultados/classificator_VP.jl'))
        dump(vr_multi, os.path.join(os.getcwd(), f'resultados/classificator_VR.jl'))
        print(f'Archivo {Color.YELLOW}classificator_VP.jl{Color.END} generado en resultados/')
        print(f'Archivo {Color.YELLOW}classificator_VR.jl{Color.END} generado en resultados/')

    return vp_multi, vr_multi