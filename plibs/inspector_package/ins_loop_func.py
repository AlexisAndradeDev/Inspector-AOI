from inspector_package import cv_func, ins_func, operations, results_management, reg_methods_func

from os import remove as delete_file
from timeit import default_timer as timer
from cv2 import imread

def use_skip_function(board_image, skip_function):
    coordinates = skip_function["coordinates"]
    inspection_image = cv_func.crop_image(board_image,coordinates)
    # Aplicar filtros secundarios al punto de inspección
    inspection_image_filt = cv_func.apply_filters(
        img=inspection_image,
        filters=skip_function["filters"]
    )

    fails, location, results, status, resulting_images = ins_func.execute_algorithm(inspection_image_filt, skip_function)

    if status != "good":
        # skippear
        return True, status, results, resulting_images, fails
    else:
        return False, status, results, resulting_images, fails

def inspect_boards(first_board, last_board, results, photo_number, references, registration_settings, settings, stage, photo=None, photo_ultraviolet=None):
    if stage == "registration" or stage == "inspection":
        # Calcular el número del primer tablero y del último tablero para el registro pre-debugeo

        # en registro, first_board es en realidad la posición dentro del panel
        # del primer tablero que inspeccionará, y last_board es la posición en el panel del último
        first_board_position = first_board
        last_board_position = last_board

        first_board, last_board = operations.get_first_last_boards_in_thread(first_board_position, last_board_position, settings["boards_per_photo"], photo_number)


    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_number in range(first_board, last_board+1):
        if stage == "debug":
            # en debugeo, actualmente el número de tablero es el número de fotografía (luego habrá que cambiarlo a que sea el número correcto)
            photo_number = board_number


        board = results_management.ObjectInspected(board_number=board_number, photo_number=photo_number, stage=stage, position_in_photo=operations.get_board_position_in_photo(board_number, settings["boards_per_photo"]))

        # Recortar región del tablero
        if stage == "inspection" or stage == "registration":
            coordinates = settings["boards_coordinates"][board.get_position_in_photo()-1]
            board_image = cv_func.crop_image(photo, coordinates)
            if settings["uv_inspection"] == "uv_inspection:True":
                board_image_ultraviolet = cv_func.crop_image(photo_ultraviolet, coordinates)
            else:
                board_image_ultraviolet = None

        # Skip
        if stage == "inspection":
            # función skip
            skip, skip_status, skip_results, skip_images, skip_fails = use_skip_function(board_image, settings["skip_function"])

            # exportar imágenes de la función de skip si el modo de revisión es total o si no pasó el skip
            if settings["check_mode"] == "check:total" or (settings["check_mode"] == "check:yes" and skip):
                operations.export_algorithm_images(
                    images=skip_images, photo_number=board.get_photo_number(), board_number=board.get_number(),
                    reference_name="skip_function", inspection_point_name="skip_function",
                    algorithm_name=settings["skip_function"]["name"],
                    light=settings["skip_function"]["light"],
                    images_path=settings["check_mode_images_path"]
                )

            if skip:
                board.set_status("skip")

                # agregar resultados de la función skip con &%&
                skip_function_results = results_management.create_algorithm_results_string(
                    settings["skip_function"]["name"], settings["skip_function"]["light"],
                    skip_status, skip_results, skip_fails
                )
                board.add_references_results(references_results="")
                board.process_results()

                results.val += board.get_results()

                # abortar inspección del tablero
                continue


        # Registro
        if stage == "inspection" or stage == "registration":
            if settings["registration_mode"] == "registration_mode:local":
                # Registro local
                registration_fail, registration_images, aligned_board_image, \
                aligned_board_image_ultraviolet, rotation, translation = reg_methods_func.register_image(
                    board_image, board_image_ultraviolet, registration_settings,
                )

                operations.export_registration_images(registration_images, board.get_photo_number(), board.get_number(), "white", settings["check_mode_images_path"], settings["check_mode"], registration_fail)

                if registration_fail:
                    # si falló, se aborta la inspección del tablero y se continúa con el siguiente
                    board.set_status("registration_failed", code=registration_fail)

                    if stage == "inspection":
                        board.add_references_results(references_results="")

                    board.process_results()
                    results.val += board.get_results()

                    continue

            elif settings["registration_mode"] == "registration_mode:global":
                # la imagen ya está alineada
                aligned_board_image = board_image
                aligned_board_image_ultraviolet = board_image_ultraviolet


            # escribir imagen del tablero alineado
            operations.export_aligned_board_image(aligned_board_image, aligned_board_image_ultraviolet,
                stage, board.get_photo_number(), board.get_position_in_photo()
            )


        elif stage == "debug": # ! CHECAR SI NO HAY ERRORES <<<-------------------------------------------0
            # debugeo lee las imágenes de los tableros ya alineadas, no registra
            aligned_board_image = imread("{0}{1}-white-board_aligned.bmp".format(settings["read_images_path"], board_number))
            if aligned_board_image is None:
                board.set_status("general_failed", code="IMG_DOESNT_EXIST") # !GENERAL_FAIL
                board.add_references_results(references_results="")
                board.process_results()
                results.val += board.get_results()
                continue

            if settings["uv_inspection"] == "uv_inspection:True":
                aligned_board_image_ultraviolet = imread("{0}{1}-ultraviolet-board_aligned.bmp".format(settings["read_images_path"], board_number))
                if aligned_board_image_ultraviolet is None:
                    board.set_status("general_failed", code="UV_IMG_DOESNT_EXIST") # !GENERAL_FAIL
                    board.add_references_results(references_results="")
                    board.process_results()
                    results.val += board.get_results()
                    continue


        # Inspeccionar referencias con multihilos
        if stage == "inspection" or stage == "debug":
            if settings["uv_inspection"] == "uv_inspection:True":
                threads = operations.create_threads(
                    func=ins_func.inspect_references,
                    threads_num=settings["threads_num_for_references"],
                    targets_num=len(references),
                    func_args=[aligned_board_image,board,references,stage,settings["check_mode"],aligned_board_image_ultraviolet]
                )
            else:
                threads = operations.create_threads(
                    func=ins_func.inspect_references,
                    threads_num=settings["threads_num_for_references"],
                    targets_num=len(references),
                    func_args=[aligned_board_image,board,references,stage,settings["check_mode"],None]
                )

            operations.run_threads(threads)

        board.process_results()

        if not board.get_results():
            # si no se obtuvo resultados de ninguna referencia
            board.set_status("general_error", code="NO_RESULTS") # !GENERAL_ERROR

        # agregar resultados del tablero
        results.val += board.get_results()


def run_registration(first_photo, last_photo, results, registration_settings, settings):
    """Exclusivo para la etapa de registro pre-debugeo."""
    photos_results = operations.StaticVar("")

    for photo_number in range(first_photo, last_photo+1):
        first_board, last_board = operations.get_first_last_boards_in_photo(settings["boards_per_photo"], photo_number)
        fail, photo, photo_ultraviolet = operations.read_photos_for_registration(settings, photo_number)

        if fail:
            # asignar resultados de fallo general a cada tablero
            for board_number in range(first_board, last_board+1):
                board = results_management.ObjectInspected(board_number=board_number, photo_number=photo_number,
                    stage="registration",
                    position_in_photo=operations.get_board_position_in_photo(board_number, settings["boards_per_photo"]),
                )

                board.set_status("general_failed", code=fail)
                board.process_results()
                photos_results.val += board.get_results()

            # abortar registro de la foto
            continue


        # Registro global
        if settings["registration_mode"] == "registration_mode:global":
            # Registro de todo el panel (registra la fotografía completa)
            registration_fail, images, photo, photo_ultraviolet, rotation, translation = reg_methods_func.register_image(
                photo, photo_ultraviolet, registration_settings,
            )

            operations.export_registration_images(images, photo_number, "global_registration", "white", settings["check_mode_images_path"], settings["check_mode"], registration_fail)

            if registration_fail:
                # asignar resultados a cada tablero como si fuera fallo de registro local
                for board_number in range(first_board, last_board+1):
                    board = results_management.ObjectInspected(board_number=board_number, photo_number=photo_number,
                        stage="registration",
                        position_in_photo=operations.get_board_position_in_photo(board_number, settings["boards_per_photo"]),
                    )

                    board.set_status("registration_failed", code=registration_fail)
                    board.process_results()
                    photos_results.val += board.get_results()

                # abortar registro de la foto
                continue

        # Procesar los tableros con multihilos
        threads = operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_per_photo"],
            func_args=[photos_results, photo_number, None, registration_settings, settings, "registration", photo, photo_ultraviolet],
        )

        operations.run_threads(threads)

    results.val += photos_results.val

    return results

def run_debug(first_board, last_board, results, references, settings, stage):
    photo_number, registration_settings, photo, photo_ultraviolet = None, None, None, None

    # multihilos para tableros
    threads = operations.create_threads(
        func=inspect_boards,
        threads_num=settings["threads_num_for_boards"],
        targets_num=settings["boards_num"],
        func_args=[results, photo_number, references, registration_settings, settings, stage, photo, photo_ultraviolet]
    )

def run(references, registration_settings, settings, stage, photo=None, photo_ultraviolet=None):
    results = operations.StaticVar("")
    total_time = 0

    # registro global
    if stage == "inspection":
        photo_number = 1 # sólo hay una fotografía por inspección

        if settings["registration_mode"] == "registration_mode:global":
            # Registro de todo el panel (registra la fotografía completa)
            start = timer()

            registration_fail, images, photo, photo_ultraviolet, rotation, translation = reg_methods_func.register_image(
                photo, photo_ultraviolet, registration_settings,
            )

            # exportar imágenes de registro si el modo de revisión es total o si falló
            operations.export_registration_images(images, photo_number, "global_registration", "white", settings["check_mode_images_path"], settings["check_mode"], registration_fail)

            if registration_fail:
                # abortar la inspección
                results.val += "%%{}".format(registration_fail)
                return results


            end = timer()
            # agregar tiempo de registro al tiempo de inspección
            total_time += end-start

    # Inspeccionar con multihilos
    if stage == "inspection":
        # multihilos para tableros
        threads = operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_per_photo"],
            func_args=[results, photo_number, references, registration_settings, settings, stage, photo, photo_ultraviolet]
        )

    elif stage == "debug":
        # multihilos para tableros
        threads = operations.create_threads(
            func=run_debug,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_num"],
            func_args=[results, references, settings, stage]
        )

    elif stage == "registration":
        # multihilos para tableros
        threads = operations.create_threads(
            func=run_registration,
            threads_num=settings["threads_num_for_photos"],
            targets_num=settings["photos_num"],
            func_args=[results, registration_settings, settings]
        )

    start = timer()
    operations.run_threads(threads)
    end = timer()
    total_time += end-start

    if not results.val:
        results.val = "%NO_RESULTS" # !FATAL_ERROR
    else:
        # agregar tiempo de inspección total al final de los resultados
        results.val += "$&${0}".format(total_time)

    return results

def start_inspection_loop(references, registration_settings, settings, stage):
    if stage == "inspection":
        while True:
            instruction = wait_for_image_or_exit_file()

            if instruction == "inspect":
                photo = operations.force_read_image("{0}photo.bmp".format(settings["read_images_path"]))
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp")

                if settings["uv_inspection"] == "uv_inspection:True":
                    photo_ultraviolet = operations.force_read_image("{0}photo-ultraviolet.bmp".format(settings["read_images_path"]))
                    delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-ultraviolet.bmp")
                    results = run(references, registration_settings, settings, stage, photo=photo, photo_ultraviolet=photo_ultraviolet)
                else:
                    results = run(references, registration_settings, settings, stage, photo=photo)

                operations.write_results(results.val, stage)
                # vaciar los resultados
                results.val = ""

            elif instruction == "exit":
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
                break # salir del bucle de inspección

    elif stage == "debug" or stage == "registration":
        results = run(references, registration_settings, settings, stage)

        operations.write_results(results.val, stage)
        # vaciar los resultados
        results.val = ""


def wait_for_image_or_exit_file():
    while True:
        if operations.file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp"):
            # si existe la imagen de los tableros se da la orden de inspeccionar
            return "inspect"
        elif operations.file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii"):
            # si existe el archivo exit.ii se da la orden de salir de la inspección
            return "exit"
