from timeit import default_timer as timer
from os import remove as delete_file
from cv2 import imread, imwrite
import threading
import os
import codecs
import copy

import sys
# Hacer esto para importar módulos y paquetes externos
sys.path.append('C:/Dexill/Inspector/Alpha-Premium/x64/plibs/pyv_functions/')
import cv_functions, math_functions

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

def write_file_error(error):
    """Escribe los resultados de los tableros."""
    file = codecs.open("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/error.io", "w", encoding='utf8')
    file.write(str(error))
    file.close()

def write_results(results):
    """Escribe los resultados de los tableros."""
    file = codecs.open("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/results.io", "w", encoding='utf8')
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

def use_skip_function(board_image, skip_function):
    coordinates = skip_function["coordinates"]
    inspection_image = cv_functions.crop_image(board_image,coordinates)
    # Aplicar filtros secundarios al punto de inspección
    inspection_image_filt = cv_functions.apply_filters(
        img=inspection_image,
        filters=skip_function["filters"]
    )

    fails, results, status, resulting_images = cv_functions.inspect_point(inspection_image_filt, skip_function)

    # añadir la imagen sin filtrar a la lista de imágenes
    resulting_images.insert(0, ["rgb", inspection_image])

    if status != "good":
        # skippear
        return True, status, results, resulting_images, fails
    else:
        return False, status, results, resulting_images, fails


def inspect_boards(first_board, last_board, photo, inspection_points, registration_settings, settings, photo_ultraviolet):
    global results

    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_number in range(first_board, last_board+1):
        board = cv_functions.ObjectInspected(board_number=board_number)

        # Recortar imagen de la región del tablero
        coordinates = settings["boards_coordinates"][board.get_index()]
        board_image = cv_functions.crop_image(photo, coordinates)
        if settings["uv_inspection"] == "uv_inspection:True":
            board_image_ultraviolet = cv_functions.crop_image(photo_ultraviolet, coordinates)


        skip, skip_status, skip_results, skip_images, skip_fails = use_skip_function(board_image, settings["skip_function"])
        # Si no pasó la función skip, saltar al siguiente tablero
        if skip:
            # volver a intentarlo con la imagen rotada 180°
            board_image_180, _ = cv_functions.rotate(board_image, 180)
            skip, _, _, _, _ = use_skip_function(board_image_180, settings["skip_function"])
            if skip:
                # si no pasó con el tablero a 180°, skippear
                board.set_status("skip")
                # agregar resultados de la función skip como punto de inspección
                board.add_inspection_point_results(settings["skip_function"]["name"], settings["skip_function"]["light"], skip_status, skip_results, skip_fails)
                # resultados del tablero: número, status, tiempo de tiempo de inspección
                board.set_board_results(registration_time=0, inspection_time=0, stage="inspection")

                results += board.get_results()

                # si está activado el check mode (low, advanced o total), exportar todas las imágenes
                if settings["check_mode"] != "check:no":
                    cv_functions.export_images(skip_images, board.get_number(), settings["skip_function"]["name"], settings["skip_function"]["light"], images_path=settings["images_path"])

                continue


        # Alinear imagen del tablero con las ventanas y guardar tiempo de registro
        start = timer()
        fail, _, aligned_board_image, rotation, translation = cv_functions.align_board_image(
            board_image, registration_settings
        )
        end = timer()
        registration_time = end-start

        if fail:
            # Volver a intentar el registro con la imagen a 180°
            board_image, _ = cv_functions.rotate(board_image, 180)
            start = timer()
            fail, _, aligned_board_image, rotation, translation = cv_functions.align_board_image(
                board_image, registration_settings
            )
            end = timer()
            registration_time += end-start
            if fail:
                # Si falló con el tablero a 180°, se aborta la inspección del tablero y se continúa con el siguiente
                board.set_status("registration_failed", code=fail)
                # resultados del tablero: número, status, tiempo de tiempo de inspección
                board.set_board_results(registration_time=registration_time, inspection_time=0, stage="inspection")
                results += board.get_board_results()
                continue
            # si pasó, rotar 180° la imagen con luz UV tambien
            board_image_ultraviolet, _ = cv_functions.rotate(board_image_ultraviolet, 180)

        # Escribir imagen del tablero alineado con luz blanca
        imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/board-{0}.bmp".format(board.get_number()), aligned_board_image)

        if settings["uv_inspection"] == "uv_inspection:True":
            # Alinear imagen del tablero con luz ultravioleta con los datos de la imagen de luz blanca
            aligned_board_image_ultraviolet, _ = cv_functions.rotate(board_image_ultraviolet, rotation)
            [x_translation, y_translation] = translation
            aligned_board_image_ultraviolet = cv_functions.translate(aligned_board_image_ultraviolet, x_translation, y_translation)

            # Escribir imagen del tablero alineado con luz ultravioleta
            imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/board-{0}-ultraviolet.bmp".format(board.get_number()), aligned_board_image_ultraviolet)

        # Inspeccionar puntos de inspección con multihilos
        if settings["uv_inspection"] == "uv_inspection:True":
            threads = create_threads(
                func=cv_functions.inspect_inspection_points,
                threads_num=settings["threads_num_for_inspection_points"],
                targets_num=len(inspection_points),
                func_args=[aligned_board_image,board,inspection_points,"inspection",settings["check_mode"],aligned_board_image_ultraviolet]
            )
        else:
            threads = create_threads(
                func=cv_functions.inspect_inspection_points,
                threads_num=settings["threads_num_for_inspection_points"],
                targets_num=len(inspection_points),
                func_args=[aligned_board_image,board,inspection_points,"inspection",settings["check_mode"],None]
            )

        start = timer()
        run_threads(threads)
        end = timer()
        inspection_time = end-start

        if not board.get_inspection_points_results():
            # si no se obtuvo resultados de ningún punto de inspección
            board.set_status("error", code="NO_RESULTS")

        # resultados del tablero: número, status, código (OPCIONAL, puede o no estar), tiempo de registro, tiempo de inspección
        board.set_board_results(registration_time, inspection_time, stage="inspection")

        results += board.get_results() # resultados de los puntos de inspección y resultados del tablero (tiempos, status)

def inspect(photo, inspection_points, registration_settings, settings, photo_ultraviolet=None):
    global results

    # Inspeccionar los tableros con multihilos
    threads = create_threads(
        func=inspect_boards,
        threads_num=settings["threads_num_for_boards"],
        targets_num=settings["boards_num"],
        func_args=[photo, inspection_points, registration_settings, settings, photo_ultraviolet]
    )

    start = timer()
    run_threads(threads)
    end = timer()
    total_time = end-start
    # agregar tiempo de inspección total al final de los resultados
    results += "%{0}".format(total_time)

def start_inspection_loop(inspection_points, registration_settings, settings):
    global results
    while True:
        # esperar a que exista la imagen de los tableros o el archivo exit.ii para salir de la inspección
        instruction = wait_for_image_or_exit_file()

        if instruction == "inspect":
            photo = force_read_image("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp")
            delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp")

            if settings["uv_inspection"] == "uv_inspection:True":
                photo_ultraviolet = force_read_image("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-ultraviolet.bmp")
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-ultraviolet.bmp")
                inspect(photo, inspection_points, registration_settings, settings, photo_ultraviolet=photo_ultraviolet)
            else:
                inspect(photo, inspection_points, registration_settings, settings)

            if not results:
                write_file_error("NO_RESULTS")
            write_results(results)
            results = "" # vaciar los resultados

        elif instruction == "exit":
            delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
            break # salir del bucle de inspección

def wait_for_image_or_exit_file():
    while True:
        if file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp"):
            # si existe la imagen de los tableros se da la orden de inspeccionar
            return "inspect"
        elif file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii"):
            # si existe el archivo exit.ii se da la orden de salir de la inspección
            return "exit"


if __name__ == '__main__':
    data = read_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/dt.ii")
    # Convertir string data a una lista de python
    data = eval(data)
    [settings_data, registration_data, inspection_points_data] = data


    # Datos de configuración
    [uv_inspection, boards_num, threads_num_for_boards, threads_num_for_inspection_points,
    check_mode, boards_coordinates, skip_function_data] = settings_data

    # Función de inspección para verificar que el tablero N esté en la imagen, si no pasa
    # la función, no se inspecciona el tablero
    skip_function = cv_functions.create_inspection_point(skip_function_data)

    settings = { # diccionario con datos de configuración
        "images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/inspections/bad_windows_results/",
        "uv_inspection":uv_inspection,
        "boards_num":boards_num,
        "boards_coordinates":boards_coordinates,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_inspection_points":threads_num_for_inspection_points,
        "check_mode":check_mode,
        "skip_function":skip_function,
    }


    # Datos de registro del tablero (alineación de imagen)
    [registration_method, method_data] = registration_data

    if registration_method == "circular_fiducials":
        [fiducials_windows, min_diameters, max_diameters, min_circle_perfections,
        max_circle_perfections, objective_angle, [objective_x, objective_y],
        fiducials_filters] = method_data

        # Crear 2 objetos con los datos de los fiduciales 1 y 2
        fiducial_1 = cv_functions.Fiducial(
            1, fiducials_windows[0], min_diameters[0],
            max_diameters[0], min_circle_perfections[0],
            max_circle_perfections[0], fiducials_filters[0])

        fiducial_2 = cv_functions.Fiducial(
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

        rotation_point1 = cv_functions.create_reference_point(
            rp_type=rp_type, name="ROTATION_POINT1", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        # Punto de rotación 2
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = rotation_point2_data

        rotation_point2 = cv_functions.create_reference_point(
            rp_type=rp_type, name="ROTATION_POINT2", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        # Punto de traslación
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = translation_point_data

        translation_point = cv_functions.create_reference_point(
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


    # Puntos de inspección
    inspection_points = cv_functions.create_inspection_points(inspection_points_data)

    # Iniciar el bucle de inspección
    results = "" # crear variable global para resultados
    start_inspection_loop(inspection_points, registration_settings, settings)
