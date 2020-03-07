"""
Sistema PyV Alpha
Script para registrar (alinear) imágenes de tableros desalineados.

Hecho por: Martín Alexis Martínez Andrade.
"""

from inspector_package import math_functions, cv_func, reg_methods_func

import threading
import codecs
import copy
from cv2 import imwrite, imread

def read_file(path):
    with open(path) as f:
        data = f.read()
        f.close()
    return data

def write_file_error(text):
    """Escribe errores del sistema (error de sintaxis, apertura de cámara)"""
    file = codecs.open("C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_error.do", "a", encoding='utf8')
    file.write(text)
    file.close()

def write_results(text):
    """Escribe los detalles de los puntos de inspección."""
    file = codecs.open("C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_results.do", "w", encoding='utf8')
    file.write(text)
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


def register_boards(first_photo, last_photo, registration_settings, settings):
    global results, all_images

    # Resultados de los tableros registrados por este hilo
    results_of_this_thread = ""


    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for photo_index in range(first_photo, last_photo+1):
        first_board = settings["boards_per_photo"] * (photo_index-1) + 1
        last_board = settings["boards_per_photo"] * (photo_index)

        # leer imagen del tablero
        photo_path = settings["images_path"] + str(photo_index) + ".bmp"
        photo = imread(photo_path)
        if photo is None:
            # si no existe la foto, escribir el status de los tableros de la foto
            # como "failed" y el código de error IMG_DOESNT_EXIST
            for board_number in range(first_board, last_board+1):
                results_of_this_thread += "{0};{1};{2}#".format(board_number, "failed", "IMG_DOESNT_EXIST")
            continue

        # leer imagen con luz UV en caso de usarse
        if settings["uv_inspection"] == "uv_inspection:True":
            photo_path = settings["images_path"] + str(photo_index) + "-ultraviolet.bmp"
            photo_ultraviolet = imread(photo_path)
            if photo_ultraviolet is None:
                # si no existe la foto UV, escribir el status de los tableros de la foto
                # como "failed" y el código de error UV_IMG_DOESNT_EXIST
                for board_number in range(first_board, last_board+1):
                    results_of_this_thread += "{0};{1};{2}#".format(board_number, "failed", "UV_IMG_DOESNT_EXIST")
                continue


        # Iterar por cada tablero de la foto
        boards_resulting_images = []
        for board_number in range(first_board, last_board+1):

            # Recortar imagen de la región del tablero
            board_position_in_photo = board_number%settings["boards_per_photo"]-1
            [x1, y1, x2, y2] = boards_coordinates[board_position_in_photo]
            board_image = photo[y1:y2, x1:x2].copy() # usar .copy() para recortar de una copia y no corromper la foto original
            # Recortar imagen UV en caso de usarse
            if settings["uv_inspection"] == "uv_inspection:True":
                board_image_ultraviolet = photo_ultraviolet[y1:y2, x1:x2].copy()

            # Alinear imagen del tablero con luz blanca
            fail_code, resulting_images, _, rotation, translation = reg_methods_func.align_board_image(
                board_image, registration_settings
            )
            # Agregar imágenes a la lista para exportarlas
            all_images.append(["{0}-{1}".format(board_number, "white"), resulting_images])

            if fail_code:
                # Agregar error a los resultados de los tableros de esta foto
                results_of_this_thread += "{0};{1};{2}#".format(board_number, "failed", fail_code)
                continue

            if settings["uv_inspection"] == "uv_inspection:True":
                # Alinear imagen del tablero con luz ultravioleta con los datos de la imagen de luz blanca
                aligned_board_image_ultraviolet, _ = cv_func.rotate(board_image_ultraviolet, rotation)
                [x_translation, y_translation] = translation
                aligned_board_image_ultraviolet = cv_func.translate(aligned_board_image_ultraviolet, x_translation, y_translation)
                # Agregar imagen del tablero con luz UV rotado a la lista de imágenes a exportar
                resulting_images = [["board_aligned", aligned_board_image_ultraviolet]]
                all_images.append(["{0}-{1}".format(board_number, "ultraviolet"), resulting_images])

            # Agregar resultados del tablero a los resultados de este hilo
            results_of_this_thread += "{0};{1}#".format(board_number, "ok")

    # Agregar resultados de este hilo a los resultados de todos los tableros
    results += results_of_this_thread


def export_resulting_images(first_board, last_board, all_images):
    # el índice de los tableros en la lista de imágenes es el número del tablero menos uno.
    images = all_images[first_board-1:last_board]
    for board_images in images:
        # Cada "board_images tiene el número del tablero y tipo de luz en la primera posición
        # La segunda posición contiene la lista de imágenes que se exportarán
        board_name = board_images[0]
        images_to_export = board_images[1]
        for image_name, image in images_to_export:
            imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/pd/{0}-{1}.bmp".format(board_name, image_name), image)

def start_boards_registration(registration_settings, settings):
    global results, all_images

    # Inspeccionar los tableros con multihilos
    threads = create_threads(
        func=register_boards,
        threads_num=settings["threads_number_for_photos"],
        targets_num=settings["photos_num"],
        func_args=[registration_settings, settings],
    )

    # Correr multihilos
    run_threads(threads)
    if not results:
        write_file_error("NO_RESULTS")
        sys.exit()

    if all_images:
        # Exportar imágenes con multihilos
        threads = create_threads(
        func=export_resulting_images,
        threads_num=settings["threads_number_for_photos"],
        targets_num=len(all_images),
        func_args=[all_images],
        )

        # Correr multihilos
        run_threads(threads)
    else:
        write_file_error("NO_RESULTING_IMAGES")

    # Escribir archivo de resultados al final
    write_results(results)


if __name__ == '__main__':
    # Cargar datos
    data = read_file("C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_dt.di")
    data = eval(data) # Convertir string "data" a una lista de python

    [settings_data, registration_data] = data


    # Datos sobre la configuración
    [uv_inspection, images_path, photos_num, boards_per_photo,
    threads_number_for_photos, boards_coordinates] = settings_data

    settings = { # diccionario con datos de configuración
        "uv_inspection":uv_inspection,
        "images_path":images_path,
        "photos_num":photos_num,
        "boards_per_photo":boards_per_photo,
        "threads_number_for_photos":threads_number_for_photos,
        "boards_coordinates":boards_coordinates,
    }


    # Datos de registro del tablero (alineación de imagen)
    [registration_method, method_data] = registration_data

    if registration_method == "circular_fiducials":
        [fiducials_windows, min_diameters, max_diameters, min_circle_perfections,
        max_circle_perfections, objective_angle, [objective_x, objective_y],
        fiducials_filters] = method_data

        # Crear 2 objetos con los datos de los fiduciales 1 y 2
        fiducial_1 = reg_methods_func.Fiducial(
            1, fiducials_windows[0], min_diameters[0],
            max_diameters[0], min_circle_perfections[0],
            max_circle_perfections[0], fiducials_filters[0])

        fiducial_2 = reg_methods_func.Fiducial(
            2, fiducials_windows[1], min_diameters[1],
            max_diameters[1], min_circle_perfections[1],
            max_circle_perfections[1], fiducials_filters[1])

        registration_settings = {
            "method":"circular_fiducials",
            "fiducial_1":fiducial_1,
            "fiducial_2":fiducial_2,
            "objective_x":objective_x,
            "objective_y":objective_y,
            "objective_angle":objective_angle,
        }

    if registration_method == "rotation_points_and_translation_point":
        [rotation_point1_data, rotation_point2_data, translation_point_data,
        objective_angle, [objective_x, objective_y], rotation_iterations] = method_data

        # Punto de rotación 1
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = rotation_point1_data

        rotation_point1 = cv_func.create_reference_point(
            rp_type=rp_type, name="ROTATION_POINT1", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        # Punto de rotación 2
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = rotation_point2_data

        rotation_point2 = cv_func.create_reference_point(
            rp_type=rp_type, name="ROTATION_POINT2", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        # Punto de traslación
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = translation_point_data

        translation_point = cv_func.create_reference_point(
            rp_type=rp_type, name="TRANSLATION_POINT", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        registration_settings = {
            "method":"rotation_points_and_translation_point",
            "rotation_point1":rotation_point1,
            "rotation_point2":rotation_point2,
            "translation_point":translation_point,
            "objective_x":objective_x,
            "objective_y":objective_y,
            "objective_angle":objective_angle,
            "rotation_iterations":rotation_iterations,
        }


    # Iniciar el registro de los tableros
    results = "" # crear variable global para resultados
    all_images = [] # lista con todas las imágenes a exportar
    start_boards_registration(registration_settings, settings)
