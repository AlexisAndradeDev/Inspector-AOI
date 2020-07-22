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

def inspect_boards(first_board, last_board, results, references, registration_settings, settings, stage, photo=None, photo_ultraviolet=None):
    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    for board_number in range(first_board, last_board+1):
        board = results_management.ObjectInspected(board_number=board_number)

        # Registrar tablero si se está en la fase de inspección
        if stage == "inspection":
            # recortar imagen de la región del tablero si se está en inspección
            coordinates = settings["boards_coordinates"][board.get_index()]
            board_image = cv_func.crop_image(photo, coordinates)
            if settings["uv_inspection"] == "uv_inspection:True":
                board_image_ultraviolet = cv_func.crop_image(photo_ultraviolet, coordinates)

            # Utilizar la función skip
            skip, skip_status, skip_results, skip_images, skip_fails = use_skip_function(board_image, settings["skip_function"])
            if skip:
                # si no pasó, saltar al siguiente tablero
                board.set_status("skip")

                # agregar resultados de la función skip con &%&
                skip_function_results = results_management.create_algorithm_results_string(
                    settings["skip_function"]["name"], settings["skip_function"]["light"],
                    skip_status, skip_results, skip_fails
                )
                board.add_references_results(references_results="")
                board.set_results()

                results.val += board.get_results()


            # exportar imágenes de la función de skip si el modo de revisión es total o si no pasó el skip
            if settings["check_mode"] == "check:total" or (settings["check_mode"] == "check:yes" and skip):
                operations.export_algorithm_images(
                    images=skip_images, board_number=board.get_number(),
                    reference_name="skip_function", inspection_point_name="skip_function",
                    algorithm_name=settings["skip_function"]["name"],
                    light=settings["skip_function"]["light"],
                    images_path=settings["images_path"]
                )

            if skip:
                # abortar inspección del tablero
                continue


            if settings["registration_mode"] == "registration_mode:local":
                # Registro local
                registration_fail, registration_images, aligned_board_image, rotation, translation = reg_methods_func.align_board_image(
                    board_image, registration_settings
                )

                if registration_fail:
                    # si falló, se aborta la inspección del tablero y se continúa con el siguiente
                    board.set_status("registration_failed", code=registration_fail)
                    board.add_references_results(references_results="")
                    board.set_results()
                    results.val += board.get_results()

                    # exportar imágenes de registro
                    if settings["check_mode"] == "check:yes" or settings["check_mode"] == "check:total":
                        operations.export_local_registration_images(registration_images, board.get_number(), "white", settings["images_path"])

                    continue

                # exportar imágenes de registro si el modo de revisión es total
                if settings["check_mode"] == "check:total":
                    operations.export_local_registration_images(registration_images, board.get_number(), "white", settings["images_path"])

                if settings["uv_inspection"] == "uv_inspection:True":
                    # alinear imagen del tablero con luz ultravioleta con las transformaciones de la imagen de luz blanca
                    aligned_board_image_ultraviolet, _ = cv_func.rotate(board_image_ultraviolet, rotation)
                    [x_translation, y_translation] = translation
                    aligned_board_image_ultraviolet = cv_func.translate(aligned_board_image_ultraviolet, x_translation, y_translation)


            elif settings["registration_mode"] == "registration_mode:global":
                # la imagen ya está alineada
                aligned_board_image = board_image
                if settings["uv_inspection"] == "uv_inspection:True":
                    aligned_board_image_ultraviolet = board_image_ultraviolet


            # escribir imagen del tablero alineado con luz blanca
            imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/board-{0}.bmp".format(board.get_number()), aligned_board_image)

            if settings["uv_inspection"] == "uv_inspection:True":
                # escribir imagen del tablero alineado con luz ultravioleta
                imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/board-{0}-ultraviolet.bmp".format(board.get_number()), aligned_board_image_ultraviolet)


        # debugeo lee las imágenes de los tableros ya alineadas, no registra
        elif stage == "debug":
            photo = imread(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/{0}-white-board_aligned.bmp".format(board_number))
            if photo is None:
                board.set_status("general_failed", code="IMG_DOESNT_EXIST") # !GENERAL_FAIL
                board.add_references_results(references_results="")
                board.set_results()
                results.val += board.get_results()
                continue

            aligned_board_image = photo
            if settings["uv_inspection"] == "uv_inspection:True":
                photo_ultraviolet = imread(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/{0}-ultraviolet-board_aligned.bmp".format(board_number))
                if photo_ultraviolet is None:
                    board.set_status("general_failed", code="UV_IMG_DOESNT_EXIST") # !GENERAL_FAIL
                    board.add_references_results(references_results="")
                    board.set_results()
                    results.val += board.get_results()
                    continue

                aligned_board_image_ultraviolet = photo_ultraviolet


        # Inspeccionar referencias con multihilos
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
            board.set_status("error", code="NO_RESULTS") # !ERROR

        # agregar resultados del tablero
        results.val += board.get_results()

def inspect(references, registration_settings, settings, stage, photo, photo_ultraviolet=None):
    results = operations.StaticVar("")
    total_time = 0

    if settings["registration_mode"] == "registration_mode:global":
        # Registro de todo el panel (registra la fotografía completa)
        start = timer()

        registration_fail, images, photo, rotation, translation = reg_methods_func.align_board_image(
            photo, registration_settings
        )

        if registration_fail:
            results.val += "%%{}".format(registration_fail) # !GLOBAL_REGISTRATION_FAIL

        # exportar imágenes de registro si el modo de revisión es total o si falló
        if settings["check_mode"] == "check:total" or (settings["check_mode"] == "check:yes" and registration_fail):
            operations.export_global_registration_images(images, "white", settings["images_path"])

        if registration_fail:
            # abortar la inspección
            return results


        if not registration_fail:
            # alinear foto con luz ultravioleta con la rotación y traslación de la foto de luz blanca
            if settings["uv_inspection"] == "uv_inspection:True":
                photo_ultraviolet, _ = cv_func.rotate(photo_ultraviolet, rotation)
                [x_translation, y_translation] = translation
                photo_ultraviolet = cv_func.translate(photo_ultraviolet, x_translation, y_translation)

        end = timer()
        # agregar tiempo de registro al tiempo de inspección
        total_time += end-start


    # Inspeccionar los tableros con multihilos
    threads = operations.create_threads(
        func=inspect_boards,
        threads_num=settings["threads_num_for_boards"],
        targets_num=settings["boards_num"],
        func_args=[results, references, registration_settings, settings, stage, photo, photo_ultraviolet]
    )

    start = timer()
    operations.run_threads(threads)
    end = timer()
    total_time += end-start
    # agregar tiempo de inspección total al final de los resultados
    results.val += "$&${0}".format(total_time)
    return results

def start_inspection_loop(references, registration_settings, settings, stage):
    if stage == "inspection":
        while True:
            instruction = wait_for_image_or_exit_file()

            if instruction == "inspect":
                photo = operations.force_read_image("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp")
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp")

                if settings["uv_inspection"] == "uv_inspection:True":
                    photo_ultraviolet = operations.force_read_image("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-ultraviolet.bmp")
                    delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo-ultraviolet.bmp")
                    results = inspect(references, registration_settings, settings, stage, photo=photo, photo_ultraviolet=photo_ultraviolet)
                else:
                    results = inspect(references, registration_settings, settings, stage, photo=photo)

                if not results.val:
                    results.val = "%NO_RESULTS" # !FATAL_ERROR
                operations.write_results(results.val, stage)
                results.val = "" # vaciar los resultados

            elif instruction == "exit":
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
                break # salir del bucle de inspección

    elif stage == "debug":
        results = inspect(references, registration_settings, settings, stage)
        if not results.val:
            results.val = "%NO_RESULTS" # !FATAL_ERROR

        operations.write_results(results.val, stage)
        results.val = "" # vaciar los resultados


def wait_for_image_or_exit_file():
    while True:
        if operations.file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/photo.bmp"):
            # si existe la imagen de los tableros se da la orden de inspeccionar
            return "inspect"
        elif operations.file_exists("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii"):
            # si existe el archivo exit.ii se da la orden de salir de la inspección
            return "exit"
