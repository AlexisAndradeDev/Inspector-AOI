import os
import copy
import codecs
import threading
from inspector_package import math_functions
from cv2 import imread, imwrite

class StaticVar:
    def __init__(self, val):
        self.val = val

def read_file(path):
    with open(path) as f:
        data = f.read()
        f.close()
    return data

def force_read_image(path):
    img = imread(path)
    while img is None:
        # si no se almacenó correctamente la imagen, volver a intentarlo
        img = imread(path)
    return img

def file_exists(path):
    # Retorna verdadera si existe el archivo
    return os.path.isfile(path)

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
        last_target = thread_targets[1] # añadir el índice del último target como argumento para func
        func_args.insert(0, last_target)
        first_target = thread_targets[0] # añadir el índice del primer target como argumento para func
        func_args.insert(0, first_target)


        thread = threading.Thread(target=func,args=copy.copy(func_args))
        threads.append(thread)

        # eliminar first_target y last_target de la lista de argumentos
        del func_args[0]
        del func_args[0]

    return threads


def export_registration_images(images, name, light, images_path, check_mode, registration_fail):
    if check_mode == "check:total" or (check_mode == "check:yes" and registration_fail):
        for image_name, image in images:
            imwrite("{0}{1}-{2}-{3}.bmp".format(images_path, name, light, image_name), image)

def export_algorithm_images(images, board_number, reference_name, inspection_point_name, algorithm_name, light, images_path):
    # exportar imágenes de un algoritmo
    for image_name, image in images:
        imwrite("{0}{1}-{2}-{3}-{4}-{5}-{6}.bmp".format(images_path, board_number, reference_name, inspection_point_name, algorithm_name, light, image_name), image)

def export_reference_images(reference_images, board_number, reference_name, images_path):
    if reference_images is None:
        return
    # exportar imágenes de una referencia
    # iterar por cada punto de inspección
    for inspection_point_images in reference_images:
        ip_name, ip_images = inspection_point_images
        # iterar por cada algoritmo
        for algorithm_images in ip_images:
            algorithm_name, algorithm_light, algorithm_images = algorithm_images
            export_algorithm_images(algorithm_images, board_number, reference_name, ip_name, algorithm_name, algorithm_light, images_path)


def add_to_images_name(images, str_):
    """
    Es utilizado para agregar una cadena de texto al nombre de todas las
    imágenes que son retornadas por funciones de inspección y métodos de registro
    para ser exportadas.
    """
    for image_index in range(len(images)):
        image_name, image = images[image_index]
        new_name = image_name + str_
        # actualizar nombre
        images[image_index][0] = new_name

    return images

def write_results(results, stage):
    """Escribe los resultados de los tableros."""
    if stage == "inspection":
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/results.io"
    elif stage == "debug":
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/dbg_results.do"
    elif stage == "registration":
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_results.do"
    file = codecs.open(path, "w", encoding='utf8')
    file.write(str(results))
    file.close()


def get_first_last_boards_in_photo(boards_per_photo, photo_number):
    first_board = boards_per_photo * (photo_number-1) + 1
    last_board = boards_per_photo * (photo_number)
    return first_board, last_board

def get_first_last_boards_in_thread(first_board_position, last_board_position, boards_per_photo, photo_number):
    first_board_in_photo_number = boards_per_photo * (photo_number-1) + 1
    last_board_in_photo_number = boards_per_photo * (photo_number)

    first_board = first_board_in_photo_number + (first_board_position - 1)
    last_board = last_board_in_photo_number - (boards_per_photo - last_board_position)

    return first_board, last_board


def read_photos_for_registration(settings, photo_number):
    path = "{0}{1}.bmp".format(settings["read_images_path"], photo_number)
    photo = imread(path)

    if photo is None:
        fail = "IMG_DOESNT_EXIST" # !GENERAL_FAIL
        return fail, None, None

    if settings["uv_inspection"] == "uv_inspection:True":
        path = "{0}{1}-ultraviolet.bmp".format(settings["images_path"], photo_number)
        photo_ultraviolet = imread(path)

        if photo_ultraviolet is None:
            fail = "UV_IMG_DOESNT_EXIST" # !GENERAL_FAIL
            return fail, None, None

    else:
        photo_ultraviolet = None

    return None, photo, photo_ultraviolet

def get_board_position_in_photo(board_number, boards_per_photo):
    """
    Retorna la posición del tablero en el panel.
    Por ejemplo, si es el tablero 20 y hay 5 tableros por foto, su posición será
    la 5; si es el tablero 18, será la 3.
    Si hay 3 tableros por foto y es el 7, su posición será 1.
    """
    position = board_number%boards_per_photo

    if position == 0:
        position = boards_per_photo # última posición

    return position
