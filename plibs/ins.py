from timeit import default_timer as timer
from os import remove as delete_file
from cv2 import imread, imwrite
import threading
import os
import codecs
import copy

from inspector_package import math_functions, cv_func, ins_func, operations, reg_methods_func

class StaticVar:
    def __init__(self, val):
        self.value = val

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
    inspection_image = cv_func.crop_image(board_image,coordinates)
    # Aplicar filtros secundarios al punto de inspección
    inspection_image_filt = cv_func.apply_filters(
        img=inspection_image,
        filters=skip_function["filters"]
    )

    fails, results, status, resulting_images = ins_func.inspect_point(inspection_image_filt, skip_function)

    # añadir la imagen sin filtrar a la lista de imágenes
    resulting_images.insert(0, ["rgb", inspection_image])

    if status != "good":
        # skippear
        return True, status, results, resulting_images, fails
    else:
        return False, status, results, resulting_images, fails

def inspect_board(results, board, photo_number, board_image, inspection_points, registration_settings, settings, board_image_ultraviolet):
    # Alinear imagen del tablero con las ventanas y guardar tiempo de registro
    start = timer()
    fail, _, aligned_board_image, rotation, translation = reg_methods_func.align_board_image(
        board_image, registration_settings
    )
    end = timer()
    registration_time = end-start

    if fail:
        # Volver a intentar el registro con la imagen a 180°
        board_image, _ = cv_func.rotate(board_image, 180)
        start = timer()
        fail, _, aligned_board_image, rotation, translation = reg_methods_func.align_board_image(
            board_image, registration_settings
        )
        end = timer()
        registration_time += end-start
        if fail:
            # Si falló con el tablero a 180°, se aborta la inspección del tablero y se continúa con el siguiente
            board.set_status("registration_failed", code=fail)
            board.set_board_results()

            results.value += board.get_results() # resultados de los puntos de inspección y resultados del tablero (tiempos, status)
            return
        # si pasó, rotar 180° la imagen con luz UV tambien
        board_image_ultraviolet, _ = cv_func.rotate(board_image_ultraviolet, 180)

    # Escribir imagen del tablero alineado con luz blanca
    imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/{0}-board-{1}.bmp".format(photo_number, board.get_board_number()), aligned_board_image)

    if settings["uv_inspection"] == "uv_inspection:True":
        # Alinear imagen del tablero con luz ultravioleta con los datos de la imagen de luz blanca
        aligned_board_image_ultraviolet, _ = cv_func.rotate(board_image_ultraviolet, rotation)
        [x_translation, y_translation] = translation
        aligned_board_image_ultraviolet = cv_func.translate(aligned_board_image_ultraviolet, x_translation, y_translation)

        # Escribir imagen del tablero alineado con luz ultravioleta
        imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/{0}-board-{1}-ultraviolet.bmp".format(photo_number, board.get_board_number()), aligned_board_image_ultraviolet)

    # Inspeccionar puntos de inspección con multihilos
    if settings["uv_inspection"] == "uv_inspection:True":
        threads = create_threads(
            func=ins_func.inspect_inspection_points,
            threads_num=settings["threads_num_for_inspection_points"],
            targets_num=len(inspection_points),
            func_args=[photo_number,board,aligned_board_image,inspection_points,"inspection",settings["check_mode"],aligned_board_image_ultraviolet]
        )
    else:
        threads = create_threads(
            func=ins_func.inspect_inspection_points,
            threads_num=settings["threads_num_for_inspection_points"],
            targets_num=len(inspection_points),
            func_args=[photo_number,board,aligned_board_image,inspection_points,"inspection",settings["check_mode"],None]
        )

    start = timer()
    run_threads(threads)
    end = timer()
    inspection_time = end-start

    if not board.get_inspection_points_results():
        # si no se obtuvo resultados de ningún punto de inspección
        board.set_status("error", code="NO_RESULTS")

    board.set_board_results()
    results.value += board.get_results() # resultados de los puntos de inspección y resultados del tablero (tiempos, status)

def inspect_boards(first_board, last_board, results, photo_number, photo, inspection_points, registration_settings, settings, photo_ultraviolet):
    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_number in range(first_board, last_board+1):
        board = operations.ObjectInspected(photo_number=photo_number, board_number=board_number)

        # Recortar imagen de la región del tablero
        coordinates = settings["boards_coordinates"][board.get_board_index()]
        board_image = cv_func.crop_image(photo, coordinates)
        if settings["uv_inspection"] == "uv_inspection:True":
            board_image_ultraviolet = cv_func.crop_image(photo_ultraviolet, coordinates)


        skip, skip_status, skip_results, skip_images, skip_fails = use_skip_function(board_image, settings["skip_function"])
        # Si no pasó la función skip, saltar al siguiente tablero
        if skip:
            # volver a intentarlo con la imagen rotada 180°
            board_image_180, _ = cv_func.rotate(board_image, 180)
            skip, _, _, _, _ = use_skip_function(board_image_180, settings["skip_function"])
            if skip:
                # agregar resultados de la función skip como punto de inspección
                board.add_inspection_point_results(settings["skip_function"]["name"], settings["skip_function"]["light"], skip_status, skip_results, skip_fails)
                # si no pasó con el tablero a 180°, skippear
                board.set_status("skip")
                board.set_board_results()

                results.value += board.get_results()

                # si está activado el check mode (low, advanced o total), exportar todas las imágenes
                if settings["check_mode"] != "check:no":
                    operations.export_images(skip_images, photo_number, board.get_board_number(), settings["skip_function"]["name"], settings["skip_function"]["light"], images_path=general_settings["images_path"])

                continue

        inspect_board(results, board, photo_number, board_image, inspection_points, registration_settings, settings, photo_ultraviolet)

def inspect_photo(photo_number, photo, inspection_points, registration_settings, settings, photo_ultraviolet=None):
    # crear string para almacenar resultados
    results = StaticVar("")

    if settings["boards_num"] == 1:
        board = operations.ObjectInspected(photo_number=photo_number, board_number=1)
        start = timer()
        inspect_board(results, board, photo_number, photo, inspection_points, registration_settings, settings, photo_ultraviolet)
        end = timer()
        photo_time = end-start

    else:
        # Inspeccionar los tableros con multihilos
        threads = create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_num"],
            func_args=[results, photo_number, photo, inspection_points, registration_settings, settings, photo_ultraviolet]
        )

        start = timer()
        run_threads(threads)
        end = timer()
        photo_time = end-start

    return results.value, photo_time

def start_inspection_loop(general_settings, photos_settings):
    while True:
        results = "" # resultados de todos los tableros
        inspection_time = 0
        for photo_settings in photos_settings:
            # esperar a que exista la foto, o el archivo exit.ii para salir de la inspección
            instruction = wait_for_image_or_exit_file(photo_settings["settings"]["photo_index"]+1)

            if instruction == "inspect":
                photo = force_read_image("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-{0}.bmp".format(photo_settings["settings"]["photo_index"]+1))
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-{0}.bmp".format(photo_settings["settings"]["photo_index"]+1))

                if photo_settings["settings"]["uv_inspection"] == "uv_inspection:True":
                    photo_ultraviolet = force_read_image("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-{0}-ultraviolet.bmp".format(photo_settings["settings"]["photo_index"]+1))
                    delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-{0}-ultraviolet.bmp".format(photo_settings["settings"]["photo_index"]+1))
                    photo_results, photo_time = inspect_photo(photo_settings["settings"]["photo_index"]+1, photo, photo_settings["inspection_points"], photo_settings["registration_settings"], photo_settings["settings"], photo_ultraviolet=photo_ultraviolet)
                else:
                    photo_results, photo_time = inspect_photo(photo_settings["settings"]["photo_index"]+1, photo, photo_settings["inspection_points"], photo_settings["registration_settings"], photo_settings["settings"])

                if not photo_results:
                    write_file_error("NO_RESULTS")

                inspection_time += photo_time
                # indica el número de la foto al final de los resultados de la foto
                photo_results += "&&{0}&&".format(photo_settings["settings"]["photo_index"]+1)
                # separador de fotos
                photo_results += "%"
                # agregar resultados del tablero a los resultados
                results += photo_results

            elif instruction == "exit":
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
                break # salir del bucle de inspección

        # tiempo total al final de las fotos al final de los resultados
        results += "##{0}##".format(inspection_time)
        write_results(results)
        results = "" # vaciar los resultados

def wait_for_image_or_exit_file(photo_number):
    while True:
        if file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-{0}.bmp".format(photo_number)):
            # si existe la imagen de los tableros se da la orden de inspeccionar
            return "inspect"
        elif file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii"):
            # si existe el archivo exit.ii se da la orden de salir de la inspección
            return "exit"


if __name__ == '__main__':
    data = read_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/dt.ii")
    # Convertir string data a una lista de python
    data = eval(data)
    [general_settings_data, photos_data] = data


    # datos generales a todas las fotos
    [photos_num] = general_settings_data
    general_settings = {
        "images_path": "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/bad_windows_results/",
        "photos_num": photos_num,
    }


    # lista con los datos de las fotografías (p. de ins., métodos de registro, configuraciones)
    photos_settings = []

    for index, photo_data in enumerate(photos_data):
        [settings_data, registration_data, inspection_points_data] = photo_data


        # Datos de configuración
        boards_num = settings_data[1]

        # un tablero en la foto
        if boards_num == 1:
            [uv_inspection, boards_num, threads_num_for_inspection_points,
            check_mode] = settings_data

            settings = { # diccionario con datos de configuración
                "uv_inspection":uv_inspection,
                "boards_num":boards_num,
                "threads_num_for_inspection_points":threads_num_for_inspection_points,
                "check_mode":check_mode,
            }

        # varios tableros en la foto
        else:
            [uv_inspection, boards_num, threads_num_for_boards, threads_num_for_inspection_points,
            check_mode, boards_coordinates, skip_function_data] = settings_data

            # Función de inspección para verificar que el tablero X esté en la imagen, si no pasa
            # la función, no se inspecciona el tablero
            skip_function = ins_func.create_inspection_point(skip_function_data)

            settings = { # diccionario con datos de configuración
                "uv_inspection":uv_inspection,
                "boards_num":boards_num,
                "boards_coordinates":boards_coordinates,
                "threads_num_for_boards":threads_num_for_boards,
                "threads_num_for_inspection_points":threads_num_for_inspection_points,
                "check_mode":check_mode,
                "skip_function":skip_function,
            }

        settings["photo_index"] = index


        # Datos de registro del tablero (alineación de imagen con ventanas)
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


        # Puntos de inspección
        if inspection_points_data[0] == "shares_inspection_points_of_another_photo":
            index_of_photo_that_shares = inspection_points_data[1]-1
            inspection_points_data = photos_data[index_of_photo_that_shares][2]

        inspection_points = ins_func.create_inspection_points(inspection_points_data)

        photo_settings = {
            "settings": settings,
            "registration_settings": registration_settings,
            "inspection_points": inspection_points,
        }
        photos_settings.append(photo_settings)

    # Iniciar el bucle de inspección
    start_inspection_loop(general_settings, photos_settings)
