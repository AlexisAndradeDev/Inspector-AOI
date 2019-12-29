from timeit import default_timer as timer
from cv2 import imread
import threading
import os
import codecs
import copy

import sys
# Hacer esto para importar módulos y paquetes externos
sys.path.append('C:/Dexill/Inspector/Alpha-Premium/x64/plibs/inspector_package/')
import math_functions, ins_func, operations

def read_file(path):
    with open(path) as f:
        data = f.read()
        f.close()
    return data

def read_image(path):
    img = imread(path)
    while img is None:
        # si no se almacenó correctamente la imagen, volver a intentarlo
        img = imread(path)
    return img

def file_exists(path):
    # Retorna verdadera si existe el archivo
    return os.path.isfile(path)

def write_results(results):
    """Escribe los resultados de los tableros."""
    file = codecs.open("C:/Dexill/Inspector/Alpha-Premium/x64/pd/dbg_results.do", "w", encoding='utf8')
    file.write(str(results))
    file.close()


def run_threads(threads):
    # Correr los procesos de los hilos
    [thread.start() for thread in threads]
    # Matar los procesos que vayan terminando
    [thread.join() for thread in threads]

def create_threads(func, threads_num, targets_num, func_args):
    """
    Retorna una lista de multihilos.
    Si hay menos targets que número de hilos que se desean crear, se asignará un hilo cada target.

    func: Función que utilizarán los multihilos.
    func_args: Argumentos que necesita la función func
    Targets: Son los elementos que procesará la función, por ejemplo:
        Los tableros son procesados por la función inspect_boards.
        Los puntos de inspección son procesados por la función inspect_inspection_points.
    """
    # Si hay menos targets que número de hilos que se desean crear, se asignará un hilo cada target.
    if targets_num < threads_num:
        targets_per_thread = math_functions.elements_per_partition(
        number_of_elements=targets_num,
        number_of_partitions=targets_num)
    else:
        targets_per_thread = math_functions.elements_per_partition(
            number_of_elements=targets_num,
            number_of_partitions=threads_num)

    threads = []
    for thread_targets in targets_per_thread:
        last_target = thread_targets[1]
        func_args.insert(0, last_target) # añadir el índice del último target como argumento para func
        first_target = thread_targets[0]
        func_args.insert(0, first_target) # añadir el índice del primer target como argumento para func

        thread = threading.Thread(target=func,args=copy.copy(func_args))
        threads.append(thread)

        # eliminar first_target y last_target de la lista de argumentos
        del func_args[0]
        del func_args[0]

    return threads

def inspect(first_board, last_board, inspection_points, settings):
    global results

    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_index in range(first_board, last_board+1):

        board = operations.ObjectInspected(board_number=board_index)

        aligned_board_image = imread("{0}{1}-white-board_aligned.bmp".format(settings["images_path"], board.get_number()))
        if aligned_board_image is None:
            # No existe la imagen del tablero
            board.set_status("error", code="IMG_DOESNT_EXIST")
            # resultados del tablero: número, status, código (OPCIONAL, puede o no estar), tiempo de registro, tiempo de inspección
            board.set_board_results(registration_time=0, inspection_time=0, stage="debug")

            results += board.get_results() # resultados de los puntos de inspección y resultados del tablero (tiempos, status)
            continue


        # Inspeccionar puntos de inspección con multihilos
        if settings["uv_inspection"] == "uv_inspection:True":
            aligned_board_image_ultraviolet = imread("{0}{1}-ultraviolet-board_aligned.bmp".format(settings["images_path"], board.get_number()))
            if aligned_board_image_ultraviolet is None:
                # No existe la imagen del tablero tomada con luz ultravioleta
                board.set_status("error", code="UV_IMG_DOESNT_EXIST")
                # resultados del tablero: número, status, tiempo de tiempo de inspección
                board.set_board_results(registration_time=0, inspection_time=0, stage="debug")

                results += board.get_results() # resultados de los puntos de inspección y resultados del tablero (tiempos, status)
                continue

            threads = create_threads(
                func=ins_func.inspect_inspection_points,
                threads_num=settings["threads_num_for_inspection_points"],
                targets_num=len(inspection_points),
                func_args=[aligned_board_image,board,inspection_points,"debug","check:total",aligned_board_image_ultraviolet]
            )
        else:
            threads = create_threads(
                func=ins_func.inspect_inspection_points,
                threads_num=settings["threads_num_for_inspection_points"],
                targets_num=len(inspection_points),
                func_args=[aligned_board_image,board,inspection_points,"debug","check:total",None]
            )

        start = timer()
        run_threads(threads)
        end = timer()
        inspection_time = end-start

        if not board.get_inspection_points_results():
            # si no se obtuvo resultados de ningún punto de inspección
            board.set_status("error", code="NO_RESULTS")

        # resultados del tablero: número, status, código (OPCIONAL, puede o no estar), tiempo de registro, tiempo de inspección
        board.set_board_results(registration_time=0, inspection_time=inspection_time, stage="debug")

        results += board.get_results() # resultados de los puntos de inspección y resultados del tablero (tiempos, status)


def start_inspection(inspection_points, settings):
    global results

    # Inspeccionar los tableros con multihilos
    threads = create_threads(
        func=inspect,
        threads_num=settings["threads_num_for_boards"],
        targets_num=settings["boards_num"],
        func_args=[inspection_points, settings],
    )

    start = timer()
    run_threads(threads)
    end = timer()
    inspection_total_time = end-start
    # agregar tiempo de inspección total al final de los resultados
    results += "%{0}".format(inspection_total_time)

    write_results(results)


def wait_for_image_or_exit_file():
    while True:
        if file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/pd/photo.bmp"):
            # si existe la imagen de los tableros se da la orden de inspeccionar
            return "inspect"
        elif file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/pd/exit.ii"):
            # si existe el archivo exit.ii se da la orden de salir de la inspección
            return "exit"


if __name__ == '__main__':
    data = read_file(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd\dbg_dt.di")
    # Convertir string data a una lista de python
    data = eval(data)
    [settings_data, _, inspection_points_data] = data


    # Datos de configuración
    [images_path, uv_inspection, boards_num, threads_num_for_boards, threads_num_for_inspection_points] = settings_data

    settings = { # diccionario con datos de configuración
        "images_path":images_path,
        "uv_inspection":uv_inspection,
        "boards_num":boards_num,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_inspection_points":threads_num_for_inspection_points,
    }

    # Puntos de inspección
    inspection_points = ins_func.create_inspection_points(inspection_points_data)

    # Iniciar el debugeo de los tableros
    results = "" # crear variable global para resultados
    start_inspection(inspection_points, settings)
