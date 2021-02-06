"""Contiene las funciones para iniciar los procesos de inspección, debugeo 
y registro pre-debugeo."""

from inspector_packageOptimizandoNuevo import (cv_func, ins_ref,
    threads_operations, inspection_objects, results_management, 
    reg_methods_func, images_operations, files_management,)

from os import remove as delete_file
from timeit import default_timer as timer
from cv2 import imread

class StaticVar:
    def __init__(self, val):
        self.val = val

def get_first_last_boards_in_photo(boards_per_panel, panel_number):
    first_board = boards_per_panel * (panel_number-1) + 1
    last_board = boards_per_panel * (panel_number)
    return first_board, last_board

def get_first_last_boards_in_thread(first_board_position, last_board_position, boards_per_panel, panel_number):
    first_board_in_panel_number = boards_per_panel * (panel_number-1) + 1
    last_board_in_panel_number = boards_per_panel * (panel_number)

    first_board = first_board_in_panel_number + (first_board_position - 1)
    last_board = last_board_in_panel_number - (boards_per_panel - last_board_position)

    return first_board, last_board

def crop_board_image(board, settings, photo, photo_ultraviolet):
    """
    Recorta la región del tablero.

    Args:
        board (inspection_objects.Board): Almacena información sobre el tablero.
        settings, photo, photo_ultraviolet - 
            Ver ins_loop_func.start_inspection_loop()

    Returns:
        board_image (numpy.ndarray): Fotografía del panel.
        board_image_ultraviolet (numpy.ndarray)): Fotografía UV del panel.
    """
    board_index = board.get_position_in_panel()-1
    coordinates = settings["boards_coordinates"][board_index]

    board_image = cv_func.crop_image(photo, coordinates)
    if settings["uv_inspection"]:
        board_image_ultraviolet = cv_func.crop_image(photo_ultraviolet, coordinates)
    else:
        board_image_ultraviolet = None

    return board_image, board_image_ultraviolet

def add_boards_in_panel_to_results_as_registration_failed(results, panel, boards_per_panel, 
        code):
    """
    Asigna a todos los tableros de un panel el estado de fallo de registro con
    el código del fallo; luego, añade los resultados de cada tablero al string
    de resultados.

    Args:
        results (ins_loop_func.StaticVar): Almacena el string de resultados.
        panel (inspection_objects.Panel): Panel que contiene a estos tableros.
        boards_per_panel (int): Número de tableros que cada panel contiene.
        code (str): Código del fallo de registro.
    """
    for board_position in range(1, boards_per_panel+1):
        board = inspection_objects.Board(panel, board_position)
        board.set_as_registration_failed(code)
        results.val += board.get_results_string()

def use_skip_function(board_image, board_image_ultraviolet, skip_function):
    """
    Retorna los resultados de la función de skip.

    Args:
        board_image (numpy.ndarray): Fotografía del tablero.
        board_image_ultraviolet (numpy.ndarray): Fotografía UV del tablero.
        skip_function (dict): Contiene los datos de la función de skip
            como un diccionario de algoritmo.

    Returns:
        skip (bool): False si el status es 'good'; True si no lo es.
        status, results, resulting_images, fails - Ver ins_ref.execute_inspection_function()
    """    
    coordinates = skip_function["coordinates"]

    if skip_function["light"] == "ultraviolet":
        inspection_image = cv_func.crop_image(board_image_ultraviolet, coordinates)
    else:
        inspection_image = cv_func.crop_image(board_image, coordinates)

    # Aplicar filtros secundarios al punto de inspección
    inspection_image_filt = cv_func.apply_filters(
        img=inspection_image,
        filters=skip_function["filters"]
    )

    fails, location, results, status, resulting_images = ins_ref.execute_inspection_function(
        inspection_image_filt, skip_function
    )

    if status != "good":
        skip = True
    else:
        skip = False
    return skip, status, results, resulting_images, fails

def inspect_boards(first_board_position, last_board_position, results, panel, 
                   settings, references, registration_settings, stage, photo, 
                   photo_ultraviolet):
    """
    Inspecciona tableros. Añade sus resultados al string de resultados.

    Args:
        first_board_position (int): Posición del primer tablero que se 
            inspeccionará.
        last_board_position (int): Posición del último tablero que se 
            inspeccionará.
        results (ins_loop_func.StaticVar): Almacena el string de resultados.
        panel (inspection_objects.Panel): Panel que contiene a estos tableros.
        photo (numpy.ndarray or None): Fotografía del panel.
            Si se está en etapa de debug, será None.
        photo_ultraviolet (numpy.ndarray or None): Fotografía UV del panel.
            Si no hay fotografía ultravioleta, será None.
            Si se está en etapa de debug, será None.
        settings, references, registration_settings, stage -
            Ver ins_loop_func.start_inspection_loop()
    """
    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_position in range(first_board_position, last_board_position+1):
        board = inspection_objects.Board(
            panel, position_in_panel=board_position,
        )

        if stage in ['inspection', 'registration']:
            board_image, board_image_ultraviolet = crop_board_image(
                board, settings, photo, photo_ultraviolet
            )


        # Skip function
        if stage in ['inspection']:
            skip, skip_status, skip_results, skip_images, skip_fails = \
                use_skip_function(
                    board_image, board_image_ultraviolet, 
                    settings["skip_function"],
                )

            if (settings["check_mode"] == "check:total" or
                    (settings["check_mode"] == "check:yes" and skip)
                ):
                images_operations.export_skip_function_images(
                    skip_images, settings["skip_function"], board,
                    settings["check_mode_images_path"],
                )

            if skip:
                # el único algoritmo del tablero será la función skip
                skip_results = \
                    results_management.create_skip_function_results_string(
                        settings["skip_function"], board, skip_status, 
                        skip_results, skip_fails,
                    )
                results.val += skip_results

                board.set_as_skip()
                results.val += board.get_results_string()

                # abortar inspección del tablero
                continue


        # Registro
        if stage in ['inspection', 'registration']:
            if settings["registration_mode"] == "registration_mode:local":
                registration_fail, registration_images, aligned_board_image, \
                aligned_board_image_ultraviolet, rotation, translation = \
                    reg_methods_func.register_image(
                        board_image, board_image_ultraviolet, 
                        registration_settings,
                    )

                images_operations.export_local_registration_images(
                    registration_images, board, "white", settings, 
                    registration_fail,
                )

                if registration_fail:
                    board.set_as_registration_failed(code=registration_fail)
                    results.val += board.get_results_string()
                    # abortar inspección del tablero
                    continue

            elif settings["registration_mode"] == "registration_mode:global":
                # la imagen ya está registrada
                aligned_board_image = board_image
                aligned_board_image_ultraviolet = board_image_ultraviolet

            # escribir imagen del tablero registrado
            images_operations.export_registered_board_image(
                aligned_board_image, aligned_board_image_ultraviolet, 
                settings, board, stage,
            )

        elif stage == 'debug':
            # debugeo lee las imágenes de los tableros ya alineadas, no registra
            # luz blanca
            aligned_board_image = imread(
                "{0}{1}-{2}-white-registered.bmp".format(
                    settings["read_images_path"], 
                    panel.get_number(), board.get_position_in_panel()
                )
            )

            if aligned_board_image is None:
                board.set_as_error(code="IMG_DOESNT_EXIST") # !BOARD_ERROR
                results.val += board.get_results_string()
                # abortar inspección del tablero
                continue

            # luz ultravioleta
            if settings["uv_inspection"]:
                aligned_board_image_ultraviolet = imread(
                    "{0}{1}-{2}-ultraviolet-registered.bmp".format(
                        settings["read_images_path"], 
                        panel.get_number(), board.get_position_in_panel()
                    )
                )

                if aligned_board_image_ultraviolet is None:
                    board.set_as_error(code="UV_IMG_DOESNT_EXIST") # !BOARD_ERROR
                    results.val += board.get_results_string()
                    continue
            else:
                aligned_board_image_ultraviolet = None


        # Inspeccionar referencias con multihilos
        if stage in ['inspection', 'debug']:
            threads = threads_operations.create_threads(
                func=ins_ref.inspect_references,
                threads_num=settings["threads_num_for_references"],
                targets_num=len(references),
                func_args=[results, panel, board,
                settings, references, stage, aligned_board_image, 
                aligned_board_image_ultraviolet],
            )

            threads_operations.run_threads(threads)

        board.create_results_string()
        results.val += board.get_results_string()

def run_registration(first_panel, last_panel, results, settings, 
                     registration_settings, stage):
    """
    (Debug) Crea e inicia los multihilos para inspeccionar tableros de un
    panel.

    Args:
        first_panel, last_panel, results, settings, registration_settings, stage -
            Ver ins_loop_func.run_inspection()
    """
    references = None
    photos_results = StaticVar("")

    for panel_number in range(first_panel, last_panel+1):
        first_board, last_board = get_first_last_boards_in_photo(
            settings["boards_per_panel"], panel_number
        )

        fail, photo, photo_ultraviolet = \
            images_operations.read_photos_for_registration(settings, panel_number)

        if fail:
            # asignar resultados de fallo general a cada tablero
            for board_number in range(first_board, last_board+1):
                board = inspection_objects.Board(board_number=board_number, panel_number=panel_number,
                    stage='registration',
                    position_in_panel=get_board_position_in_panel(board_number, settings["boards_per_panel"]),
                )

                board.set_status("general_failed", code=fail)
                board.process_results()
                photos_results.val += board.get_results_string()

            # abortar registro de la foto
            continue


        # registro global
        if settings["registration_mode"] == "registration_mode:global":
            # Registro de todo el panel (registra la fotografía completa)
            registration_fail, images, photo, photo_ultraviolet, rotation, 
            translation = reg_methods_func.register_image(
                photo, photo_ultraviolet, registration_settings,
            )

            images_operations.export_local_registration_images(
                images, board, "white", settings, registration_fail,
            )

            if registration_fail:
                # asignar resultados a cada tablero como si fuera fallo de registro local
                for board_number in range(first_board, last_board+1):
                    board = inspection_objects.Board(board_number=board_number, panel_number=panel_number,
                        stage='registration',
                        position_in_panel=get_board_position_in_panel(board_number, settings["boards_per_panel"]),
                    )

                    board.set_status("registration_failed", code=registration_fail)
                    board.process_results()
                    photos_results.val += board.get_results_string()

                # abortar registro de la foto
                continue

        # Procesar los tableros con multihilos
        panel_results = StaticVar("")
        threads = threads_operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_per_panel"],
            func_args=[results, panel, settings, references, registration_settings, 
            stage, photo, photo_ultraviolet],
        )

        threads_operations.run_threads(threads)
        # agregar resultados de la foto
        photos_results.val += panel_results.val

    # agregar resultados de todas las fotos
    results.val += photos_results.val

    return results

def run_debug(first_panel, last_panel, results, settings, references, stage):
    """
    (Debug) Crea e inicia los multihilos para inspeccionar tableros de un
    panel.

    Args:
        first_panel, last_panel, results, settings, references, stage -
            Ver ins_loop_func.run_inspection()
    """
    registration_settings, photo, photo_ultraviolet = None, None, None

    for panel_number in range(first_panel, last_panel+1):
        panel = inspection_objects.Panel(panel_number, stage)

        # multihilos para tableros
        threads = threads_operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_per_panel"],
            func_args=[results, panel, settings, references, 
            registration_settings, stage, photo, photo_ultraviolet],
        )

        threads_operations.run_threads(threads)
        # añadir resultados del panel
        results.val += panel.get_results_string()

def read_inspection_photo(settings):
    """
    Retorna la fotografía de luz blanca y la fotografía ultravioleta.

    Args:
        settings - Ver ins_loop_func.start_inspection_loop()

    Returns:
        photo (numpy.ndarray): Fotografía del panel.
        photo_ultraviolet (numpy.ndarray)): Fotografía UV del panel.
    """
    photo = images_operations.force_read_image(
        settings["read_images_path"] + "photo.bmp"
    )
    delete_file(settings["read_images_path"] + "photo.bmp")

    if settings["uv_inspection"]:
        photo_ultraviolet = images_operations.force_read_image(
            settings["read_images_path"] + "photo-ultraviolet.bmp"
        )
        delete_file(settings["read_images_path"] + "photo-ultraviolet.bmp")
    else:
        photo_ultraviolet = None
 
    return photo, photo_ultraviolet

def run_inspection(first_panel, last_panel, results, settings, references, 
                   registration_settings, stage):
    """
    (Inspección) Crea e inicia los multihilos para inspeccionar tableros de un 
    panel.

    Args:
        first_panel (int): Número del primer panel que se inspeccionará.
        last_panel (int): Número del último panel que se inspeccionará.
        results (ins_loop_func.StaticVar): Almacena el string de resultados.
        settings, references, registration_settings, stage - 
            Ver ins_loop_func.start_inspection_loop()
    """
    photo, photo_ultraviolet = read_inspection_photo(settings)

    for panel_number in range(first_panel, last_panel+1):
        panel = inspection_objects.Panel(panel_number)

        # Registro global
        if settings["registration_mode"] == "registration_mode:global":
            # registrar la fotografía completa
            registration_fail, images, photo, photo_ultraviolet, rotation, \
            translation = reg_methods_func.register_image(
                photo, photo_ultraviolet, registration_settings,
            )

            images_operations.export_global_registration_images(
                images, panel, "white", settings, registration_fail,
            )

            if registration_fail:
                add_boards_in_panel_to_results_as_registration_failed(
                    results, panel, settings["boards_per_panel"], 
                    registration_fail,
                )

                panel.set_as_registration_failed(code=registration_fail)
                results.val += panel.get_results_string()
                # abortar la inspección de este panel
                continue

        # multihilos para tableros
        threads = threads_operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_per_panel"],
            func_args=[results, panel, settings, references, 
            registration_settings, stage, photo, photo_ultraviolet],
        )

        threads_operations.run_threads(threads)

        panel.create_results_string()
        results.val += panel.get_results_string()

def run(settings, references, registration_settings, stage):
    """
    Crea e inicia los multihilos para inspeccionar paneles.

    Args:
        settings, references, registration_settings, stage - 
            Ver ins_loop_func.start_inspection_loop().

    Returns:
        results (ins_loop_func.StaticVar): Almacena el string de resultados.
    """
    results = StaticVar("")
    total_time = 0

    # Crear multihilos
    if stage == 'inspection':
        threads = threads_operations.create_threads(
            func=run_inspection,
            threads_num=settings["threads_num_for_panels"],
            targets_num=settings["panels_num"],
            func_args=[results, settings, references, registration_settings, stage]
        )

    elif stage == 'debug':
        threads = threads_operations.create_threads(
            func=run_debug,
            threads_num=settings["threads_num_for_panels"],
            targets_num=settings["panels_num"],
            func_args=[results, settings, references, stage]
        )

    elif stage == 'registration':
        threads = threads_operations.create_threads(
            func=run_registration,
            threads_num=settings["threads_num_for_panels"],
            targets_num=settings["panels_num"],
            func_args=[results, settings, registration_settings, stage]
        )

    # Correr multihilos
    start = timer()
    threads_operations.run_threads(threads)
    end = timer()
    total_time += end-start

    if not results.val:
        results.val = "%NO_RESULTS" # !FATAL_ERROR
    else:
        # agregar tiempo de inspección total al final de los resultados
        results.val += "$&${0}".format(total_time)

    return results

def wait_for_image_or_exit_file():
    while True:
        if files_management.file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp"):
            # si existe la imagen de los tableros se da la orden de inspeccionar
            return "inspect"
        elif files_management.file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii"):
            # si existe el archivo exit.ii se da la orden de salir de la inspección
            return "exit"

def start_inspection_loop(settings, references, registration_settings, stage):
    """
    Inicia los procesos de inspección, debugeo o registro pre-debugeo.

    Args:
        settings (dict): Diccionario con la configuración general.
        references (list): Lista con los diccionarios de las referencias
            creados con 'create_reference'.
        registration_settings (dict): Diccionario con la configuración
            de registro.
        stage (str): Etapa que se ejecutará. 
            'debug', 'inspection', 'registration'.
    """
    if stage == 'inspection':
        while True:
            instruction = wait_for_image_or_exit_file()

            if instruction == "inspect":
                results = run(settings, references, registration_settings, stage)

                results_management.write_results(results.val, stage)
                # vaciar los resultados
                results.val = ""

            elif instruction == "exit":
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
                break # salir del bucle de inspección

    elif stage == 'debug' or stage == 'registration':
        results = run(settings, references, registration_settings, stage)

        results_management.write_results(results.val, stage)
        # vaciar los resultados
        results.val = ""
