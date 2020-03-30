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

    # añadir la imagen sin filtrar a la lista de imágenes
    resulting_images.insert(0, ["rgb", inspection_image])

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
                # volver a intentarlo con la imagen rotada 180°
                board_image, _ = cv_func.rotate(board_image, 180)
                skip_with_180, _, _, _, _ = use_skip_function(board_image, settings["skip_function"])
                if skip_with_180:
                    # si no pasó con el tablero a 180°, saltar al siguiente tablero
                    board.set_status("skip")

                    # agregar resultados de la función skip con &%&
                    skip_function_results = results_management.create_algorithm_results_string(
                        settings["skip_function"]["name"], settings["skip_function"]["light"],
                        skip_status, skip_results, skip_fails
                    )
                    board.add_references_results(references_results="")
                    board.set_results()

                    results.val += board.get_results()
                    # exportar imágenes de la función de skip
                    if settings["check_mode"] == "check:yes":
                        operations.export_algorithm_images(
                            images=skip_images, board_number=board.get_number(),
                            reference_name="skip_function", inspection_point_name="skip_function",
                            algorithm_name=settings["skip_function"]["name"],
                            light=settings["skip_function"]["light"],
                            images_path=settings["images_path"]
                        )
                    continue

            # exportar imágenes de la función de skip si el modo de revisión es total
            if settings["check_mode"] == "check:total":
                operations.export_algorithm_images(
                    images=skip_images, board_number=board.get_number(),
                    reference_name="skip_function", inspection_point_name="skip_function",
                    algorithm_name=settings["skip_function"]["name"],
                    light=settings["skip_function"]["light"],
                    images_path=settings["images_path"]
                )


            # Registro
            # alinear imagen del tablero con las ventanas y guardar tiempo de registro
            fail, resulting_images, aligned_board_image, rotation, translation = reg_methods_func.align_board_image(
                board_image, registration_settings
            )

            if fail:
                # volver a intentar el registro con la imagen a 180°
                board_image, _ = cv_func.rotate(board_image, 180)
                fail, _, aligned_board_image, rotation, translation = reg_methods_func.align_board_image(
                    board_image, registration_settings
                )
                if fail:
                    registration_images = ["{0}-{1}".format(board.get_number(), "white"), resulting_images]
                    # si falló con el tablero a 180°, exportar imágenes del registro; se aborta la inspección del tablero y se continúa con el siguiente
                    operations.export_registration_images(registration_images, settings["images_path"])

                    board.set_status("registration_failed", code=fail)
                    board.add_references_results(references_results="")
                    board.set_results()
                    results.val += board.get_results()
                    continue

                if settings["uv_inspection"] == "uv_inspection:True":
                    # si pasó, rotar 180° la imagen con luz UV tambien
                    board_image_ultraviolet, _ = cv_func.rotate(board_image_ultraviolet, 180)


            # escribir imagen del tablero alineado con luz blanca
            imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/board-{0}.bmp".format(board.get_number()), aligned_board_image)

            if settings["uv_inspection"] == "uv_inspection:True":
                # alinear imagen del tablero con luz ultravioleta con las transformaciones de la imagen de luz blanca
                aligned_board_image_ultraviolet, _ = cv_func.rotate(board_image_ultraviolet, rotation)
                [x_translation, y_translation] = translation
                aligned_board_image_ultraviolet = cv_func.translate(aligned_board_image_ultraviolet, x_translation, y_translation)
                # escribir imagen del tablero alineado con luz ultravioleta
                imwrite("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/board-{0}-ultraviolet.bmp".format(board.get_number()), aligned_board_image_ultraviolet)


        # debugeo lee las imágenes de los tableros ya alineadas, no registra
        elif stage == "debug":
            photo = imread(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/{0}-white-board_aligned.bmp".format(board_number))
            aligned_board_image = photo
            if settings["uv_inspection"] == "uv_inspection:True":
                photo_ultraviolet = imread(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/{0}-ultraviolet-board_aligned.bmp".format(board_number))
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
            board.set_status("error", code="NO_RESULTS")

        # agregar resultados del tablero
        results.val += board.get_results()

def inspect(references, registration_settings, settings, stage, photo=None, photo_ultraviolet=None):
    results = operations.StaticVar("")
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
    total_time = end-start
    # agregar tiempo de inspección total al final de los resultados
    results.val += "$&${0}".format(total_time)
    return results

def start_inspection_loop(references, registration_settings, settings, stage):
    if stage == "inspection":
        while True:
            # esperar a que exista la imagen de los tableros o el archivo exit.ii para salir de la inspección
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
                    results.val = "NO_RESULTS"
                operations.write_results(results.val, stage)
                results.val = "" # vaciar los resultados

            elif instruction == "exit":
                delete_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/exit.ii")
                break # salir del bucle de inspección
    elif stage == "debug":
        results = inspect(references, registration_settings, settings, stage)
        if not results.val:
            results.val = "NO_RESULTS"
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
