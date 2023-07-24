from imports.support.utils import MINIUM_RANGE

################################################################################
# MACROS
################################################################################

# MACRO para generar o no las graficas de la SDF y la SDD
GENERATE_GRAPHS = True

################################################################################
# FUNCIONES PRIVADAS
################################################################################

#
# Funcion para extraer los maximos de los bursts
#
def __get_bursts_maximum_times(SDF_data: list, bursts_times_ranges: list[list]) -> list[int]:
    maxs = []
    for min_time, max_time in bursts_times_ranges:
        mx = SDF_data[min_time]
        mx_time = min_time
        for s in range(min_time, max_time+1):
            if SDF_data[s] > mx:
                mx = SDF_data[s]
                mx_time = s
        maxs.append(mx_time)

    return maxs

#
# Funcion privada para sacar las estadisticas Sf, St, Ef y Et de un burst
#
# Parametros:
#   burst_max_time: tiempo de deteccion de un maximo de un burst (P)
#   SDF_data: funcion SDF
#
# Output:
#   St: ubicación del Sf respecto a P
#   Et: ubicación del Ef respecto a P
#
def __SDF_measures(burst_max_time, burst_ranges,  SDF_data):
    if (SDF_data[burst_max_time] - SDF_data[burst_ranges[1]] < 0):
        print(f'caso derecha mayor que centro: valor centro = {round(SDF_data[burst_max_time], 7)}, valor maximo real = {round(max(SDF_data), 7)}')

    # calculamos el punto de la izquierda mas cercano a la desviacion pico a pico al 5% desde el minimo
    left_standarized_deviation = (SDF_data[burst_max_time] - SDF_data[burst_ranges[0]]) * 0.05 + SDF_data[burst_ranges[0]]
    St = burst_ranges[0]
    while St < burst_max_time:
        if SDF_data[St] <= left_standarized_deviation and SDF_data[St+1] >= left_standarized_deviation:
            if abs(SDF_data[St] - left_standarized_deviation) > abs(SDF_data[St+1] - left_standarized_deviation):
                St += 1
            break
        St += 1
    
    # igual para la derecha
    right_standarized_deviation = (SDF_data[burst_max_time] - SDF_data[burst_ranges[1]]) * 0.05 + SDF_data[burst_ranges[1]]
    Et = burst_ranges[1]
    while Et > burst_max_time:
        if SDF_data[Et] <= right_standarized_deviation and SDF_data[Et-1] >= right_standarized_deviation:
            if abs(SDF_data[Et] - right_standarized_deviation) > abs(SDF_data[Et-1] - right_standarized_deviation):
                Et -= 1
            break
        Et -= 1

    # tenemos ya las estadisticas St y Et
    return St - burst_max_time, Et - burst_max_time

#
# Funcion privada para obtener las medidas de la SDD
#
# Parametros:
#   burst_max_time: tiempos de maximo de un burst (P)
#   SDD_data: datos de la SDD
#   St: ubicacion de la marca S respecto a P
#   Et: ubicacion de la marca E respecto a P
#
# Output:
#   R2t: Ubicacion de R2 relativa a P
#   R1t: Ubicacion de R1 relativa a P
#   F1t: Ubicacion de F1 relativa a P
#   F2t: Ubicacion de F2 relativa a P
#
def __SDD_measures(burst_max_time, SDD_data, St, Et):
    # Sacamos las ubicacion de R2, recorremos de izquierda a derecha desde el inicio del burst hasta antes del final
    #x = St + burst_max_time
    for x in range(St + burst_max_time + 1, burst_max_time, 1):
        for y in range(max(0, x - MINIUM_RANGE), min(len(SDD_data)-1, x + MINIUM_RANGE)):
            if SDD_data[x] < SDD_data[y]:
                break
        else:
            break
    R2t = x - burst_max_time
    
    # Sacamos el ultimo maximo yendo de adelante hacia atras
    #x = burst_max_time
    for x in reversed(range(St + burst_max_time + 1, burst_max_time, 1)):
        for y in range(max(0, x - MINIUM_RANGE), min(len(SDD_data)-1, x + MINIUM_RANGE)):
            if SDD_data[x] < SDD_data[y]:
                break
        else:
            break
    R1t = x - burst_max_time
    
    # ahora pasamos a F1t, es igual que R2 desde P hasta antes de E
    #x = burst_max_time
    for x in range(burst_max_time + 1, burst_max_time + Et - 1, 1):
        for y in range(max(0, x - MINIUM_RANGE), min(len(SDD_data)-1, x + MINIUM_RANGE)):
            if SDD_data[x] > SDD_data[y]:
                break
        else:
            break
    F1t = x - burst_max_time
    
    # hacemos lo mismo para F2t que en R1t
    #x = burst_max_time + Et - 1
    for x in reversed(range(burst_max_time + 1, burst_max_time + Et - 1, 1)):
        for y in range(max(0, x - MINIUM_RANGE), min(len(SDD_data)-1, x + MINIUM_RANGE)):
            if SDD_data[x] > SDD_data[y]:
                break
        else:
            break
    F2t = x - burst_max_time

    return R2t, R1t, F1t, F2t

#
# Funcion privada para extraer estadisticas de areas de la SDF entre picos de la SDD
#
# Parametros:
#   time_value_burst_data: par (tiempo, valor) de un burst
#   SDF_data: valores de la SDF
#   R2t: Ubicacion de R2 relativa a P
#   R1t: Ubicacion de R1 relativa a P
#   F1t: Ubicacion de F1 relativa a P
#   F2t: Ubicacion de F2 relativa a P
#
# Output:
#   A1: Area bajo la SDF entre R2t y R1t
#   A2: Area bajo la SDF entre R1t y F1t
#   A3: Area bajo la SDF entre F1t y F2t
#
def __measures_areas_SDF(burst_max_time, SDF_data, R2t, R1t, F1t, F2t):
    timestamps = [R2t, R1t, F1t, F2t]
    A = []
    
    # en cada area hay que ahcer lo mismo, sumar el numero de medidas y dividir entre el total de medidas
    for i in range(0, 3):
        N = timestamps[i+1] - timestamps[i]
        if N > 0:
            A.append(sum([SDF_data[t] for t in range(timestamps[i] + burst_max_time, timestamps[i+1] + burst_max_time, 1)])/N)
        else:
            A.append(0)

    return A[0], A[1], A[2]

#
# Funcion privada para juntar los datos completos de los bursts
#
# Parametros:
#   SDF_data: datos de la SDF
#   SDD_data: datos de la SDD
#   bursts_maximum: lista de tiempos de deteccion de maximos
#
def __burst_data_extraction(SDF_data, SDD_data, bursts_maximum_times, bursts_times_ranges) -> list:
    burst_data = []

    for burst_max_time, burst_ranges in zip(bursts_maximum_times, bursts_times_ranges):
        # estadisticas de la SDF, la SDD y las areas de los bursts en la SDF
        St, Et = __SDF_measures(burst_max_time, burst_ranges, SDF_data)
        R2t, R1t, F1t, F2t = __SDD_measures(burst_max_time, SDD_data, St, Et)
        A1, A2, A3 = __measures_areas_SDF(burst_max_time, SDF_data, R2t, R1t, F1t, F2t)
        burst_data.append(
            {
                'St' : St,
                'Et' : Et,
                'R2t' : R2t,
                'R1t' : R1t,
                'F1t' : F1t,
                'F2t' : F2t,
                'A1' : A1,
                'A2' : A2,
                'A3' : A3,
                'centro' : burst_max_time
            }
        )
    return burst_data
        

################################################################################
# FUNCION PUBLICA DEL MODULO
################################################################################

#
# funcion principal del modulo
# Se presupone que se han extraido los spikes de los datos de voltaje
#
def carlson(
    SDF_data: list,
    SDD_data: list,
    bursts_times_ranges: list,
) -> list[dict]:
    
    # sacamos los maximos de cada burst
    bursts_maximum_times = __get_bursts_maximum_times(SDF_data=SDF_data, bursts_times_ranges=bursts_times_ranges)

    # obtenemos las estadisticas de cada burst
    burst_stats_location = __burst_data_extraction(SDF_data, SDD_data, bursts_maximum_times, bursts_times_ranges)

    # vamos a meterlo todo en una lista con los valores calculados
    carlson_stats =[
        {
            'Sf' : SDF_data[burst_d["St"] + burst_d["centro"]],
            'Pf' : SDF_data[burst_d["centro"]],
            'Ef' : SDF_data[burst_d["Et"] + burst_d["centro"]],
            'R2f' : SDF_data[burst_d["R2t"] + burst_d["centro"]],
            'R1f' : SDF_data[burst_d["R1t"] + burst_d["centro"]],
            'F1f' : SDF_data[burst_d["F1t"] + burst_d["centro"]],
            'F2f' : SDF_data[burst_d["F2t"] + burst_d["centro"]],
            'R2d' : SDD_data[burst_d["R2t"] + burst_d["centro"]],
            'R1d' : SDD_data[burst_d["R1t"] + burst_d["centro"]],
            'F1d' : SDD_data[burst_d["F1t"] + burst_d["centro"]],
            'F2d' : SDD_data[burst_d["F2t"] + burst_d["centro"]],
            'St' : burst_d["St"],
            'Et' : burst_d["Et"],
            'R2t' : burst_d["R2t"],
            'R1t' : burst_d["R1t"],
            'F1t' : burst_d["F1t"],
            'F2t' : burst_d["F2t"],
            'A1' : burst_d["A1"],
            'A2' : burst_d["A2"],
            'A3' : burst_d["A3"],
            'centro' : burst_d["centro"]
        }
        for burst_d in burst_stats_location
    ]
    
    return carlson_stats
    