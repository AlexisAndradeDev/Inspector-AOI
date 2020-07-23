from inspector_package import cv_func, ins_func, operations, results_management, reg_methods_func

from os import remove as delete_file
from timeit import default_timer as timer
from cv2 import imread, imwrite

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

def inspect_boards(first_board, last_board, results, references, registration_settings, settings, stage, photo=None, photo_ultraviolet=None, photo_number=None):
    if stage == "registration":
        # Calcular el número del primer tablero y del último tablero para el registro pre-debugeo

        # en registro, first_board es en realidad la posición dentro del panel
        # del primer tablero que inspeccionará, y last_board es la posición en el panel del último
        first_board_position = first_board
        last_board_position = last_board

        first_board, last_board = operations.get_first_last_boards_in_thread(first_board_position, last_board_position, settings["boards_per_photo"], photo_number)


    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_number in range(first_board, last_board+1):
        board = results_management.ObjectInspected(board_number=board_number, stage=stage, position_in_photo=operations.get_board_position_in_photo(board_number, settings["boards_per_photo"]))

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
                    images=skip_images, board_number=board.get_number(),
                    reference_name="skip_function", inspection_point_name="skip_function",
                    algorithm_name=settings["skip_function"]["name"],
                    light=settings["skip_function"]["light"],
                    check_mode_images_path=settings["check_mode_images_path"]
                )

            if skip:
                board.set_status("skip")

                # agregar resultados de la función skip con &%&
                skip_function_results = results_management.create_algorithm_results_string(
                    settings["skip_function"]["name"], settings["skip_function"]["light"],
                    skip_status, skip_results, skip_fails
                )
                board.add_references_results(references_results="")
                board.set_results()

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

                operations.export_registration_images(registration_images, board.get_number(), "white", settings["check_mode_images_path"], settings["check_mode"], registration_fail)

                if registration_fail:
                    # si falló, se aborta la inspección del tablero y se continúa con el siguiente
                    board.set_status("registration_failed", code=registration_fail)

                    if stage == "inspection":
                        board.add_references_results(references_results="")

                    board.set_results()
                    results.val += board.get_results()

                    continue

            elif settings["registration_mode"] == "registration_mode:global":
                # la imagen ya está alineada
                aligned_board_image = board_image
                aligned_board_image_ultraviolet = board_image_ultraviolet


            # escribir imagen del tablero alineado con luz blanca
            if stage == "inspection":
                dir = "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/"
            elif stage == "registration":
                dir = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/"

            imwrite(dir+"board-{0}.bmp".format(board.get_number()), aligned_board_image)

            if settings["uv_inspection"] == "uv_inspection:True":
                # escribir imagen del tablero alineado con luz ultravioleta
                imwrite(dir+"board-{0}-ultraviolet.bmp".format(board.get_number()), aligned_board_image_ultraviolet)


        elif stage == "debug":
            # debugeo lee las imágenes de los tableros ya alineadas, no registra
            aligned_board_image = imread("{0}{1}-white-board_aligned.bmp".format(settings["read_images_path"], board_number))
            if aligned_board_image is None:
                board.set_status("general_failed", code="IMG_DOESNT_EXIST") # !GENERAL_FAIL
                board.add_references_results(references_results="")
                board.set_results()
                results.val += board.get_results()
                continue

            if settings["uv_inspection"] == "uv_inspection:True":
                aligned_board_image_ultraviolet = imread("{0}{1}-ultraviolet-board_aligned.bmp".format(settings["read_images_path"], board_number))
                if aligned_board_image_ultraviolet is None:
                    board.set_status("general_failed", code="UV_IMG_DOESNT_EXIST") # !GENERAL_FAIL
                    board.add_references_results(references_results="")
                    board.set_results()
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

        board.set_results()

        if not board.get_results():
            # si no se obtuvo resultados de ninguna referencia
            board.set_status("general_error", code="NO_RESULTS") # !GENERAL_ERROR

        # agregar resultados del tablero
        results.val += board.get_results()


def register_photos(first_photo, last_photo, results, total_time, registration_settings, settings):
    """Exclusivo para la etapa de registro pre-debugeo."""
    photos_results = operations.StaticVar("")

    for photo_number in range(first_photo, last_photo+1):
        first_board, last_board = operations.get_first_last_boards_in_photo(settings["boards_per_photo"], photo_number)
        fail, photo, photo_ultraviolet = operations.read_photos_for_registration(settings, photo_number)

        if fail:
            # asignar resultados de fallo general a cada tablero
            for board_number in range(first_board, last_board+1):
                board = results_management.ObjectInspected(board_number=board_number, stage="registration", position_in_photo=None)

                board.set_status("general_failed", code=fail)
                board.set_results()
                photos_results.val += board.get_results()

            # abortar registro de la foto
            continue


        # Registro global
        if settings["registration_mode"] == "registration_mode:global":
            # Registro de todo el panel (registra la fotografía completa)
            start = timer()

            registration_fail, images, photo, photo_ultraviolet, rotation, translation = reg_methods_func.register_image(
                photo, photo_ultraviolet, registration_settings,
            )

            operations.export_registration_images(images, "global_registration", "white", settings["check_mode_images_path"], settings["check_mode"], registration_fail)

            if registration_fail:
                # asignar resultados a cada tablero como si fuera fallo de registro local
                for board_number in range(first_board, last_board+1):
                    board = results_management.ObjectInspected(board_number=board_number, stage="registration", position_in_photo=None)

                    board.set_status("registration_failed", code=registration_fail)
                    board.set_results()
                    photos_results.val += board.get_results()

                # abortar registro de la foto
                continue

            end = timer()
            # agregar tiempo de registro al tiempo de inspección
            total_time += end-start

        # Procesar los tableros con multihilos
        threads = operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_per_photo"],
            func_args=[photos_results, None, registration_settings, settings, "registration", photo, photo_ultraviolet, photo_number],
        )

        operations.run_threads(threads)

    results.val += photos_results.val

    return results

def inspect(references, registration_settings, settings, stage, photo=None, photo_ultraviolet=None):
    results = operations.StaticVar("")
    total_time = 0

    if stage == "inspection":
        if settings["registration_mode"] == "registration_mode:global":
            # Registro de todo el panel (registra la fotografía completa)
            start = timer()

            registration_fail, images, photo, photo_ultraviolet, rotation, translation = reg_methods_func.register_image(
                photo, photo_ultraviolet, registration_settings,
            )

            # exportar imágenes de registro si el modo de revisión es total o si falló
            operations.export_registration_images(images, "global_registration", "white", settings["check_mode_images_path"], settings["check_mode"], registration_fail)

            if registration_fail:
                # abortar la inspección
                results.val += "%%{}".format(registration_fail)
                return results


            end = timer()
            # agregar tiempo de registro al tiempo de inspección
            total_time += end-start

    # Inspeccionar con multihilos
    if stage == "inspection" or stage == "debug":
        # multihilos para tableros
        threads = operations.create_threads(
            func=inspect_boards,
            threads_num=settings["threads_num_for_boards"],
            targets_num=settings["boards_num"],
            func_args=[results, references, registration_settings, settings, stage, photo, photo_ultraviolet]
        )

    elif stage == "registration":
        # multihilos para tableros
        threads = operations.create_threads(
            func=register_photos,
            threads_num=settings["threads_num_for_photos"],
            targets_num=settings["photos_num"],
            func_args=[results, total_time, registration_settings, settings]
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
                    results = inspect(references, registration_settings, settings, stage, photo=photo, photo_ultraviolet=photo_ultraviolet)
                else:
                    results = inspect(references, registration_settings, settings, stage, photo=photo)

                operations.write_results(results.val, stage)
                # vaciar los resultados
                results.val = ""

            elif instruction == "exit":
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
                break # salir del bucle de inspección

    elif stage == "debug" or stage == "registration":
        results = inspect(references, registration_settings, settings, stage)

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
