from inspector_package import files_management, inspection_objects

def create_algorithm_results_string(algorithm, board, status, 
                                    inspection_function_results, 
                                    algorithm_coordinates_in_board, codes):
    """Retorna el string de resultados del algoritmo."""
    [inspection_point, reference, board, panel] = \
        inspection_objects.get_containers_of_inspection_object(
            inspection_object=algorithm, unlinked_container=board,
        )

    results = "{};{};{};{};{};{};{};{};{};{};{}#".format(
        "algorithm", panel.get_number(), board.get_position_in_panel(), 
        reference["name"], inspection_point["name"], algorithm["name"],
        status, algorithm["light"], inspection_function_results, 
        algorithm_coordinates_in_board, codes,
    )
    return results

def create_skip_function_results_string(skip_function, board, status, 
        inspection_function_results, codes):
    """Retorna el string de resultados de la función de skip."""
    panel = inspection_objects.get_container_of_inspection_object(board)

    results = "{};{};{};{};{};{};{};{}#".format(
        "skip_function", panel.get_number(), board.get_position_in_panel(),
        skip_function["name"], status, skip_function["light"], 
        inspection_function_results, codes,
    )
    return results

def create_inspection_point_results_string(inspection_point, board, status):
    """Retorna el string de resultados del punto de inspección."""
    [reference, board, panel] = \
        inspection_objects.get_containers_of_inspection_object(
            inspection_object=inspection_point, unlinked_container=board,
        )

    results = "{};{};{};{};{};{}#".format(
        "inspection_point", 
        panel.get_number(), board.get_position_in_panel(), reference["name"],
        inspection_point["name"], status,
    )
    return results

def create_reference_algorithm_results_string(status, results):
    """Retorna el string de resultados del algoritmo de referencia."""
    results_str = str(results)
    # eliminar las comillas: ['good'] --> [good]
    results_str = results_str.replace(r"'", "")
    reference_algorithm_results_string = "[{0}${1}]".format(status, results_str)
    return reference_algorithm_results_string

def create_reference_results_string(reference, board, status, 
                                    reference_algorithm_results_string):
    """Retorna el string de resultados de la referencia."""
    [board, panel] = \
        inspection_objects.get_containers_of_inspection_object(
            inspection_object=reference, unlinked_container=board,
        )

    results = "{};{};{};{};{};{}#".format(
        "reference",
        panel.get_number(), board.get_position_in_panel(), reference["name"],
        status, reference_algorithm_results_string,
    )
    return results


def status_has_to_change(status, new_status):
    """Se utiliza para saber si un estado actual debe cambiarse a uno nuevo.

    'error': Todos los status pueden cambiar a 'error'. Una vez se obtenga el 
    estado 'error', no puede cambiar a otro estado.

    'registration_failed': Todos los status, excepto 'error', pueden cambiar 
    a 'registration_failed'.
        Sólo puede cambiar a 'error'.
        Al obtener status 'registration_failed', no puede obtenerse el status 
        'skip', ya que no se omitió el tablero.

    'skip': Todos los status, excepto 'error', pueden cambiarse a 'skip'.
        Puede cambiar a 'error'.
        Al obtener status 'skip', no puede obtenerse el status 
        'registration_failed', ya que el proceso para registrar el tablero 
        no se ejecutaría.

    'bad': Los status 'error', 'skip' y 'registration_failed' no pueden 
        cambiar a 'bad'; los demás, sí pueden hacerlo.
        Puede cambiar a 'error', 'skip' y 'registration_failed'.

    'good': Ningún status puede cambiar a 'good'.
        Puede cambiar a cualquier estado.
        Los status siempre se inicializan en 'good'.

    Args:
        status (str): Estado actual.
        new_status (str): Estado que se quiere asignar.

    Returns:
        True si status debe cambiarse a new_status; False si no debe
        cambiarse el estado actual al nuevo.
    """
    if new_status == status:
        return False

    elif new_status == "error":
        return True
    elif new_status == "skip":
        if status in ["error"]:
            return False
        else:
            return True
    elif new_status == "registration_failed":
        if status in ["error"]:
            return False
        else:
            return True
    elif new_status == "bad":
        if status in ["error", "skip", "registration_failed"]:
            return False
        else:
            return True
    elif new_status == "good":
        return False
    elif new_status == "not_executed":
        return False

def update_status(status, new_status):
    """Evalúa si el estado actual debe cambiarse por el estado
    nuevo introducido; si debe cambiarse, retorna el estado nuevo, si no,
    retorna el estado actual.

    Advertencia: No funciona para puntos de inspección.

    Args:
        status (str): Estado actual.
        new_status (str): Estado que se quiere asignar.
    Returns:
        updated_status (str): Estado actualizado.
    """
    if status_has_to_change(status, new_status):
        updated_status = new_status
    else:
        updated_status = status
    return updated_status

def update_algorithm_status(algorithm_status, inspection_function_status, 
                            algorithm):
    """
    Retorna el estado del algoritmo actualizado según el estado de la 
    función de inspección.

    Args:
        algorithm_status (str): Estado del algoritmo.
        inspection_function_status (str): Estado obtenido en la función de
            inspección.
        algorithm (dict): Contiene los datos del algoritmo.
    """
    if inspection_function_status not in ["bad", "good"]:
        new_status = inspection_function_status
    elif inspection_function_status == algorithm["needed_status_to_be_good"]:
        new_status = "good"
    else:
        new_status = "bad"

    updated_status = update_status(algorithm_status, new_status)

    return updated_status

def update_algorithms_status(algorithms_status, algorithm_status, 
                             algorithm):
    """Retorna el estado global actualizado de los algoritmos según el
    estado de un algoritmo.
    
    Args:
        algorithms_status (str): Estado global de los algoritmos.
        algorithm_status (str): Estado del algoritmo.
        algorithm (dict): Contiene los datos del algoritmo.
    """
    if algorithm["ignore_bad_status"] and algorithm_status == "bad":
        # el estado no se cambia
        updated_status = algorithms_status
    else:
        updated_status = update_status(algorithms_status, algorithm_status)
    return updated_status

def set_algorithm_as_error(algorithm, board, algorithm_results, code):
    """
    Manipula directamente el diccionario de resultados del algoritmo.
    Asigna el estado de error y agrega el código del error a la lista
    de códigos de errores.
    Crea también el string de resultados.

    Args:
        algorithm (dict): Contiene los datos del algoritmo.
        board (inspection_objects.Board): Contiene datos del tablero.
        algorithm_results (dict): Contiene los resultados del algoritmo.
        code (str): Código de error que se agregará a la lista de errores.
    """
    algorithm_results["status"] = "error"
    algorithm_results["codes"].append(code)
    algorithm_results["string"] = create_algorithm_results_string(
        algorithm, board, algorithm_results["status"], 
        algorithm_results["inspection_function_results"],
        algorithm_results["coordinates_in_board"],
        algorithm_results["codes"],
    )

def save_algorithm_results(algorithm, algorithm_results, algorithms_results, algorithms_status, results):
    """
    Manipula directamente el diccionario de los resultados de los algoritmos.
    Agrega el diccionario de resultados del algoritmo al diccionario de 
    resultados de todos los algoritmos.
    Agrega el string de resultados del algoritmo al string de todos los 
    resultados (al que se escribirá en el archivo de results).
    Actualiza el status global de todos los algoritmos.
    
    Args:
        algorithm (dict): Contiene los datos del algoritmo.
        algorithm_results (dict): Contiene los resultados del algoritmo.
    """
    # agregar dict de resultados
    algorithms_results[algorithm["name"]] = algorithm_results

    # agregar string de resultados
    if algorithm_results["status"] != "not_executed":
        results.val += algorithm_results["string"]

    # actualizar status global de todos los algoritmos
    algorithms_status = update_algorithms_status(
        algorithms_status=algorithms_status,
        algorithm_status=algorithm_results["status"],
        algorithm=algorithm
    )

def write_results(results, stage):
    """
    Escribe el archivo de resultados.

    Args:
        results (str): contenido del archivo de resultados.
        stage (str): Etapa en la que se encuentra. 
            'debug', 'inspection', 'registration'.
    """
    if stage == 'inspection':
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/results.io"
    elif stage == 'debug':
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/dbg_results.do"
    elif stage == 'registration':
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_results.do"
    files_management.write_file(path, results)

def test_create_results():
    """
    (test) Crea objetos de panel, tablero, referencia, punto de inspección y
    algoritmo; luego, crea sus resultados y los imprime.
    """
    pan = inspection_objects.Panel(1)
    board = inspection_objects.Board(pan, 1)
    ref = {"name":"ref", "object_type": "reference"}
    ip = {"name": "ip", "object_type": "inspection_point", "reference": ref}
    alg = {"light":"white", "name": "algo", "object_type": "algorithm", "inspection_point": ip}

    print(create_algorithm_results_string(
        algorithm=alg, board=board, status="good", inspection_function_results=[], 
        algorithm_coordinates_in_board=[5,5,10,15], codes=[])
    )
    print(create_inspection_point_results_string(
        inspection_point=ip, board=board, status="good")
    )
    print(create_reference_results_string(
        reference=ref, board=board, status="good", 
        reference_algorithm_results_string=[])
    )
