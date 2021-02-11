from inspector_package import (ins_func, cv_func, results_management, images_operations)

def execute_inspection_function(inspection_image_filt, algorithm):
    """
    Ejecuta una función de inspección y retorna sus resultados.

    Args:
        inspection_image_filt (numpy.ndarray): Imagen del algoritmo con filtros
            secundarios.
        algorithm (dict): Contiene los datos del algoritmo.

    Returns:
        fails (list): Lista con códigos de fallos. Si no hay fallos, será una
            lista vacía.
        location (Union[dict, str]): Coordenadas en las que se encontró.
            Si no se encontró, tomará valor de "not_available".

            Si se encontró, será un diccionario.

            Si la localización es un par de coordenadas, 
            contendrá keys y values:
            'type': 'pair_of_coordinates',
            'coordinates': lista con dos coordenadas [x,y]

            Si la localización es una sola coordenada, contendrá keys y values:
            'type': 'one_coordinate',
            'axis': Eje de la coordenada. Puede ser 'x' ó 'y',
            'coordinate': coordenada de tipo int
        results (list): Contiene los resultados de la función de inspección.
        status (str): Estado de la función de inspección. 
        resulting_images (dict de numpy.ndarray): Imágenes del proceso de la 
            función de inspección.
    """
    if (algorithm["inspection_function"] == ins_func.BLOB):
        fails, location, results, status, resulting_images = \
            ins_func.inspection_function_blob(
                inspection_image_filt, algorithm
            )

    elif (algorithm["inspection_function"] == ins_func.TEMPLATE_MATCHING):
        fails, location, results, status, resulting_images = \
            ins_func.inspection_function_template_matching(
                inspection_image_filt, algorithm
            )

    elif (algorithm["inspection_function"] == ins_func.UNIQUE_TRANSITION):
        fails, location, results, status, resulting_images = \
            ins_func.inspection_function_unique_transition(
                inspection_image_filt, algorithm
            )

    elif (algorithm["inspection_function"] == ins_func.TRANSITIONS):
        fails, location, results, status, resulting_images = \
            ins_func.inspection_function_transitions(
                inspection_image_filt, algorithm
            )

    elif (algorithm["inspection_function"] == ins_func.HISTOGRAM):
        fails, location, results, status, resulting_images = \
            ins_func.inspection_function_histogram(
                inspection_image_filt, algorithm
            )

    else:
        return ["INVALID_INSPECTION_FUNCTION"], "not_available", [], "failed", []

    return fails, location, results, status, resulting_images

def calculate_location_inside_algorithm_in_photo(inspection_point, 
        algorithm, location):
    """Localización encontrada con un algoritmo (que toma el origen de la
    ventana del algoritmo) tomando como origen el (0,0) de la foto."""

    if location["type"] == "one_coordinate":
        # Convertir a par de coordenadas
        x1,y1,x2,y2 = algorithm["coordinates"]
        if location["axis"] == "x":
            window_height = y2-y1
            # tomar como «y» la mitad del alto de la ventana
            location["coordinates"] = [
                location["coordinate"], int(window_height/2)
            ]
        if location["axis"] == "y":
            window_width = x2-x1
            # tomar como «x» la mitad del ancho de la ventana
            location["coordinates"] = [
                int(window_width)/2, location["coordinate"]
            ]

    # localización = localización dentro de la ventana del algoritmo +
    # coordenadas del algoritmo + coordenadas del punto de inspección
    coordinates = math_functions.sum_lists(
        location["coordinates"],
        math_functions.sum_lists(
            inspection_point["coordinates"],
            algorithm["coordinates"][:2],
        ),
    )
    return location["coordinates"]

def get_algorithm_coordinates_origin(algorithm, inspection_point, 
                                    algorithms_results):
    if algorithm["take_as_origin"] == "$inspection_point":
        # tomar el punto de inspección
        origin = inspection_point["coordinates"]
    else:
        # tomar otro algoritmo
        origin = algorithms_results[algorithm["take_as_origin"]]["location"]
    return origin

def inspect_algorithm(algorithms_results, board, inspection_point, algorithm,
                      settings, image, image_ultraviolet):
    # si el algoritmo ya fue procesado, no volver a inspeccionarlo
    if algorithm["name"] in algorithms_results:
        if algorithms_results[algorithm["name"]]["locked"]:
            return algorithms_results[algorithm["name"]]

    algorithm_results = {
        "string":"", "status":"good", "inspection_function_results":[], 
        "location":"not_available", "codes":[], "locked": False
    }

    # no inspeccionar el algoritmo si se asignó para ignorarlo en él
    if board.get_position_in_panel() in algorithm["ignore_in_boards"]:
        # Puede volver a inspeccionarse si hay otro algoritmo con el mismo
        # nombre, y los resultados se actualizarán a los de este último.
        algorithm_results["status"] = "not_executed"
        return algorithm_results

    # no inspeccionar el algoritmo si la cadena no cumple con el status
    if (algorithm["chained_to"] != None and
            algorithms_results[algorithm["chained_to"]]["status"] \
            != algorithm["needed_status_of_chained_to_execute"]):
        # puede volver a inspeccionarse si hay otro algoritmo con el mismo
        # nombre
        algorithm_results["status"] = "not_executed"
        return algorithm_results


    # Recortar imagen del algoritmo
    origin = get_algorithm_coordinates_origin(
        algorithm, inspection_point, algorithms_results
    )

    if origin == "not_available":
        # !ALGORITHM_ERROR
        results_management.set_algorithm_as_error(
            algorithm, board, algorithm_results,
            "ALGO_TO_TAKE_AS_ORIGIN_DOESNT_HAVE_COORDINATES",
        )
        return algorithm_results

    if algorithm["light"] == "ultraviolet":
        inspection_image = cv_func.crop_image(
            image_ultraviolet, algorithm["coordinates"],
            take_as_origin=origin,
        )
    else:
        inspection_image = cv_func.crop_image(
            image, algorithm["coordinates"],
            take_as_origin=origin,
        )

    # Filtrar imagen
    inspection_image_filt = cv_func.apply_filters(
        inspection_image, algorithm["filters"]
    )


    # Ejecutar función de inspección
    [fails, location, results, inspection_function_status, resulting_images] = \
        execute_inspection_function(inspection_image_filt, algorithm)

    if location != "not_available":
        # guardar localización tomando como origen el (0,0) del tablero
        location = calculate_location_inside_algorithm_in_photo(
            inspection_point, algorithm, location,
        )

    algorithm_results["location"] = location
    algorithm_results["inspection_function_results"] = results
    algorithm_results["codes"] += fails
    algorithm_results["status"] = results_management.update_algorithm_status(
        algorithm_results["status"], inspection_function_status, algorithm,
    )
    algorithm_results["string"] = \
        results_management.create_algorithm_results_string(
            algorithm, board, algorithm_results["status"],
            algorithm_results["inspection_function_results"],
            algorithm_results["codes"],
        )
    algorithm_results["locked"] = True # no puede volver a inspeccionarse

    if (settings["check_mode"] == "check:total" or
            (settings["check_mode"] == "check:yes" and
             algorithm_results["status"] != "good"
            )
        ):
        images_operations.export_algorithm_images(
            resulting_images, algorithm, board, 
            settings["check_mode_images_path"]
        )

    return algorithm_results


def inspect_algorithms(results, board, inspection_point, settings, image, 
                       image_ultraviolet):
    # status global de todos los algoritmos
    algorithms_status = "good"
    # resultados de cada algoritmo
    algorithms_results = {}

    for algorithm in inspection_point["algorithms"]:
        algorithm_results = inspect_algorithm(
            algorithms_results, board, inspection_point, algorithm, settings, 
            image, image_ultraviolet,
        )

        # agregar dict de resultados
        algorithms_results[algorithm["name"]] = algorithm_results

        # agregar string de resultados
        if algorithm_results["status"] != "not_executed":
            results.val += algorithm_results["string"]

        # actualizar status global
        algorithms_status = results_management.update_algorithms_status(
            algorithms_status=algorithms_status,
            algorithm_status=algorithm_results["status"],
            algorithm=algorithm
        )

    return algorithms_status, algorithms_results

def inspect_inspection_point(results, board, inspection_point, settings,
                             image, image_ultraviolet):
    inspection_point_results = {
        "string":"", "status":"good", "algorithms_results":{}
    }

    # no inspeccionar si se asignó para ignorar en el tablero
    if board.get_position_in_panel() in inspection_point["ignore_in_boards"]:
        inspection_point_results["status"] = "not_executed"
        return inspection_point_results

    algorithms_status, inspection_point_results["algorithms_results"] \
        = inspect_algorithms(
            results, board, inspection_point, settings, image, 
            image_ultraviolet,
        )
    
    inspection_point_results["status"] = results_management.update_status(
        status=inspection_point_results["status"],
        new_status=algorithms_status,
    )

    inspection_point_results["string"] = \
        results_management.create_inspection_point_results_string(
            inspection_point, board, inspection_point_results["status"],
        )

    return inspection_point_results

def inspect_inspection_points(results, board, inspection_points, settings, 
                              image, image_ultraviolet):
    # status global de todos los puntos de inspección
    inspection_points_status = "good"
    # resultados de cada punto de inspección
    inspection_points_results = {}

    for inspection_point in inspection_points:
        inspection_point_results = inspect_inspection_point(
            results, board, inspection_point, settings, image,
            image_ultraviolet,
        )

        # agregar dict de resultados del punto de inspección al dict
        inspection_points_results[inspection_point["name"]] \
            = inspection_point_results

        # agregar string de resultados
        if inspection_point_results["status"] != "not_executed":
            results.val += inspection_point_results["string"]

        # actualizar status global
        inspection_points_status = results_management.update_status(
            status=inspection_points_status,
            new_status=inspection_point_results["status"],
        )

    return inspection_points_status, inspection_points_results


def reference_algorithm_classification(inspection_points_status):
    """Algoritmo de referencia: Clasificación.
    Adopta el status de los puntos de inspección y su resultado sólo es el
    status de los puntos de inspección."""
    fail = None
    results = [inspection_points_status]
    status = inspection_points_status
    return fail, results, status

def execute_reference_algorithm(reference_algorithm, inspection_points_results, 
                                inspection_points_status):
    """
    Ejecuta el algoritmo de referencia y retorna sus resultados.

    Args:
        reference_algorithm (dict): Contiene los datos del algoritmo de
            referencia.
        inspection_points_results (dict): Contiene los resultados de cada
            punto de inspección.
        inspection_points_status (str): Estado global de los puntos de 
            inspección.

    Returns:
        fail (str): Código de fallo. Si no hubo fallos, será None.
        results (list): Contiene los resultados de la función del algoritmo
            de referencia.
        status (str): Estado del algoritmo de referencia.
        results_string (str): String de resultados del algoritmo de referencia.
    """
    reference_algorithm_results = {
        "string":"", "status":"good", "results":{}, "code":None,
    }

    if reference_algorithm["function"] == ins_func.CLASSIFICATION:

        fail, results, status = reference_algorithm_classification(
            inspection_points_status,
        )

    else:
        fail, results, status = \
            "INVALID_REFERENCE_ALGORITHM_FUNCTION", [], "error"

    reference_algorithm_results["code"] = fail
    reference_algorithm_results["results"] = results
    reference_algorithm_results["status"] = status
    reference_algorithm_results["string"] = \
        results_management.create_reference_algorithm_results_string(
            reference_algorithm_results["status"], 
            reference_algorithm_results["results"],
        )

    return reference_algorithm_results


def inspect_reference(results, board, reference, settings, image, 
                      image_ultraviolet):
    """
    Inspecciona una referencia.

    Args:
        reference (dict): Contiene datos de una referencia.
            Creada con inspection_objects.create_reference()
        results, board, settings, image, image_ultraviolet -
            Ver ins_ref.inspect_references()
    Returns:
        Diccionario con los resultados de la referencia.
    """    
    reference_results = {
        "string":"", "status":"good", "inspection_points_results":{},
        "code": None,
    }

    # no inspeccionar si se asignó para ignorar en el tablero
    if board.get_position_in_panel() in reference["ignore_in_boards"]:
        reference_results["status"] = "not_executed"
        return reference_results

    inspection_points_status, reference_results["inspection_points_results"] \
        = inspect_inspection_points(
            results, board, reference["inspection_points"], 
            settings, image, image_ultraviolet,
        )

    results_management.update_status(
        status=reference_results["status"],
        new_status=inspection_points_status,
    )

    # ejecutar el algoritmo de la referencia
    reference_algorithm_results = \
        execute_reference_algorithm(
            reference["reference_algorithm"], 
            reference_results["inspection_points_results"],
            inspection_points_status,
        )

    reference_results["code"] = reference_algorithm_results["code"]
    reference_results["status"] = results_management.update_status(
        status=reference_results["status"],
        new_status=reference_algorithm_results["status"],
    )

    reference_results["string"] = \
        results_management.create_reference_results_string(
            reference, board, reference_results["status"], 
            reference_algorithm_results["string"],
        )

    return reference_results

def inspect_references(first_reference_number, last_reference_number, results, panel,
                       board, settings, references, stage, image,
                       image_ultraviolet):
    """
    Inspecciona referencias. Añade sus resultados al string de resultados.

    Args:
        first_reference_number (int): Número de la primera referencia que se
            inspeccionará.
        last_reference_number (int): Número de la última referencia que se
            inspeccionará.
        board (inspection_objects.Board): Tablero que contiene a estas 
            referencias.
        image (numpy.ndarray): Imagen del tablero.
        image_ultraviolet (numpy.ndarray or None): Imagen UV del tablero.
            Si no hay fotografía ultravioleta, será None.
        results, panel - Ver ins_loop_func.inspect_boards()
        settings, references, stage - 
            Ver ins_loop_func.start_inspection_loop()
    """
   # status global de todas las referencias
    references_status = "good"

    # obtener í­ndice de las referencias en la lista
    first_reference_index = first_reference_number-1
    last_reference_index = last_reference_number-1
    references = references[first_reference_index:last_reference_index+1]

    for reference in references:
        reference_results = inspect_reference(
            results, board, reference, settings, image, image_ultraviolet
        )

        if reference_results["status"] != "not_executed":
            results.val += reference_results["string"]

        references_status = results_management.update_status(
            status=references_status,
            new_status=reference_results["status"],
        )
    
    board.update_status(new_status=references_status)
    panel.update_status(new_status=board.get_status())
