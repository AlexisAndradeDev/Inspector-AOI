from inspector_package import math_functions, cv_func, \
    images_operations, threads_operations, results_management, excepts

from cv2 import imread, imwrite, rectangle, cvtColor, COLOR_BGR2GRAY
from numpy import array

TEMPLATE_MATCHING = "m"
BLOB = "b"
UNIQUE_TRANSITION = "ut"
TRANSITIONS = "t"
HISTOGRAM = "h"

CLASSIFICATION = "classification"

def get_inspection_function_parameters(algorithm, parameters_data):
    """
    Retorna un diccionario de parámetros para una función de inspección.

    Args:
        algorithm (dict): Diccionario con los datos del algoritmo creado 
            con la función inspection_objects.create_algorithm()
        parameters_data (list): Lista de parámetros de esta función de 
            inspección.

    Returns:
        parameters (dict): Diccionario con los parámetros de esta
            función de inspección.
    """
    # blob
    if (algorithm["inspection_function"] == BLOB):
        parameters = get_blob_parameters(algorithm, parameters_data)
    # template matching
    elif (algorithm["inspection_function"] == TEMPLATE_MATCHING):
        parameters = get_template_matching_parameters(algorithm, parameters_data)
    # unique transition
    elif (algorithm["inspection_function"] == UNIQUE_TRANSITION):
        parameters = get_unique_transition_parameters(algorithm, parameters_data)
    # transitions
    elif (algorithm["inspection_function"] == TRANSITIONS):
        parameters = get_transitions_parameters(algorithm, parameters_data)
    # histogram
    elif (algorithm["inspection_function"] == HISTOGRAM):
        parameters = get_histogram_parameters(algorithm, parameters_data)
    return parameters


def get_blob_parameters(algorithm, parameters_data):
    """
    Retorna un diccionario de parámetros para una función de inspección 'blob'.

    Args:
        algorithm, parameters_data - 
            Ver ins_func.get_inspection_function_parameters()

    Returns:
        parameters - Ver ins_func.get_inspection_function_parameters()
    """
    parameters = {
        "invert_binary": parameters_data[0],
        "color_scale": parameters_data[1],
        "lower_color": parameters_data[2],
        "upper_color": parameters_data[3],
        "min_blob_size": parameters_data[4],
        "max_blob_size": parameters_data[5],
        "min_area": parameters_data[6],
        "max_area": parameters_data[7],
        "biggest_blob_max_area": parameters_data[8],
    }

    if parameters["color_scale"] == "hsv":
        # convertir a array de numpy
        parameters["lower_color"] = array(parameters["lower_color"])
        parameters["upper_color"] = array(parameters["upper_color"])

    return parameters

def get_template_matching_parameters(algorithm, parameters_data):
    """
    Retorna un diccionario de parámetros para una función de inspección 'template matching'.

    Args:
        algorithm, parameters_data - 
            Ver ins_func.get_inspection_function_parameters()

    Returns:
        parameters - Ver ins_func.get_inspection_function_parameters()
    """
    color_scale = parameters_data[0]
    if type(color_scale) is str:
        name, invert_binary, color_scale_for_binary, color_range = \
            color_scale, None, None, None
    elif type(color_scale) is list:
        [name, invert_binary, color_scale_for_binary, color_range] = color_scale

    template_path = parameters_data[1]
    number_of_sub_templates = parameters_data[2]
    sub_templates = get_sub_templates(
        number_of_sub_templates, template_path, algorithm["filters"]
    )

    min_califications = parameters_data[3]
    required_matches = parameters_data[4]

    parameters = {
        "color_scale": name,
        "invert_binary": invert_binary,
        "color_scale_for_binary": color_scale_for_binary,
        "color_range": color_range,
        "sub_templates": sub_templates,
        "min_califications": min_califications,
        "required_matches": required_matches,
    }
    return parameters

def get_sub_templates(number_of_sub_templates, template_path, filters):
    sub_templates = []
    # Lista de subtemplates y la calificación mí­nima de cada una
    for template_index in range(number_of_sub_templates):
        template_number = template_index + 1
        sub_template_image = imread(template_path + "-" + str(template_number) + ".bmp")
        sub_template_image = cv_func.apply_filters(sub_template_image, filters)
        sub_templates.append([sub_template_image, template_number])
    return sub_templates


def get_unique_transition_parameters(algorithm, parameters_data):
    """
    Retorna un diccionario de parámetros para una función de inspección 'transición única'.

    Args:
        algorithm, parameters_data - 
            Ver ins_func.get_inspection_function_parameters()

    Returns:
        parameters - Ver ins_func.get_inspection_function_parameters()
    """
    parameters = {
        "searching_orientation":parameters_data[0],
        "min_difference":parameters_data[1],
        "brightness_difference_type":parameters_data[2],
        "group_size":parameters_data[3],
    }
    return parameters


def get_histogram_parameters(algorithm, parameters_data):
    """
    Retorna un diccionario de parámetros para una función de inspección 'histograma'.

    Args:
        algorithm, parameters_data - 
            Ver ins_func.get_inspection_function_parameters()

    Returns:
        parameters - Ver ins_func.get_inspection_function_parameters()
    """
    parameters = {
        "lower_gray": array(parameters_data[0][0]),
        "upper_gray": array(parameters_data[0][1]),
        "min_area_percentage": parameters_data[1][0],
        "max_area_percentage": parameters_data[1][1],
        "min_average_gray": parameters_data[2][0],
        "max_average_gray": parameters_data[2][1],
        "average_of_lowest_gray_levels_number": parameters_data[3],
        "min_average_of_lowest_gray_levels": parameters_data[4],
        "average_of_highest_gray_levels_number": parameters_data[5],
        "max_average_of_highest_gray_levels": parameters_data[6],
        "min_lowest_gray": parameters_data[7][0],
        "max_highest_gray": parameters_data[7][1],
    }
    return parameters

def determinate_required_transitions(transitions_data):
    """
    Determina qué transiciones se usarán en la función de inspección 'transiciones'.

    Args:
        transitions_data (dict): Diccionario de parámetros de cada una de 
            las transiciones.
            Ver ins_func.get_transitions_data(), sección Returns

    Raises:
        Exception('REQUIRED_TRANSITIONS_NOT_VALID'): Si una transición no tiene
            un nombre válido.

    Returns:
        required_transitions (list): lista con las transiciones que se
            utilizarán.
            Ejemplo: ['across1', 'across2', 'along']
    """
    transitions_number = len(transitions_data)

    required_transitions = transitions_data.keys()

    for transition in required_transitions:
        if transition not in ["across1", "across2", "along"]:
            raise excepts.FatalError("REQUIRED_TRANSITIONS_NOT_VALID") # !FATAL_ERROR

    return required_transitions

def get_transitions_data(transitions_data):
    """
    Retorna los datos de las transiciones en forma de diccionario.

    Args:
        transitions_data (list): Datos de las transiciones sin procesar.

    Returns:
        transitions_data_dict (dict): Datos de las transiciones procesados
            en forma de diccionario. 
            Las keys son los nombres de las transiciones 
                ('across1', 'across2', etc.).
            Los values son los datos de cada transición.
    """    
    # convertir transitions_data a diccionario
    transitions_data_dict = {}
    for transition_data in transitions_data:
        transition_data_dict = {
            "name": transition_data[0],
            "side": transition_data[1],
            "coordinates": transition_data[2],
            "min_difference": transition_data[3],
            "brightness_difference_type": transition_data[4],
            "searching_orientation": transition_data[5],
            "group_size": transition_data[6],
        }

        # agregar datos de la transición a la lista de transiciones
        transitions_data_dict[transition_data_dict["name"]] = transition_data_dict
    return transitions_data_dict

def get_transitions_parameters(algorithm, parameters_data):
    """
    Retorna un diccionario de parámetros para una función de inspección 'transiciones'.

    Args:
        algorithm, parameters_data - 
            Ver ins_func.get_inspection_function_parameters()

    Returns:
        parameters - Ver ins_func.get_inspection_function_parameters()
    """
    
    parameters = {
        "calculate_component_width": parameters_data[0],
        "min_component_width": parameters_data[1],
        "max_component_width": parameters_data[2],
    }

    transitions_data = parameters_data[3]
    parameters["transitions_data"] = get_transitions_data(transitions_data)

    parameters["required_transitions"] = determinate_required_transitions(
        parameters["transitions_data"]
    )

    parameters["required_transitions_number"] = len(parameters["required_transitions"])

    return parameters


def inspection_function_blob(inspection_image, algorithm):
    """
    Ejecuta y retorna los resultados de la función de inspección blob.

    Args:
        inspection_image, algorithm - Ver ins_ref.execute_inspection_function()

    Returns:
        resulting_images: Imágenes del proceso de la función de inspección.
            Contiene imágenes cuyas keys son: 'filtered', 'binary'.
            Más detalles en docstring de ins_ref.execute_inspection_function()
        fails, location, results, status -
            Ver ins_ref.execute_inspection_function()
    """
    fails = []
    location = "not_available"
    status = ""
    resulting_images = {}

    resulting_images["filtered"] = inspection_image

    blob_area, biggest_blob, binary_image = cv_func.calculate_blob_area(
        inspection_image,
        algorithm["parameters"]["lower_color"],
        algorithm["parameters"]["upper_color"],
        algorithm["parameters"]["color_scale"],
        algorithm["parameters"]["invert_binary"],
    )

    resulting_images["binary"] = binary_image


    blob_is_correct = evaluate_blob_results(
        blob_area, biggest_blob,
        algorithm["parameters"]["min_area"],
        algorithm["parameters"]["max_area"],
        algorithm["parameters"]["biggest_blob_max_area"]
    )

    if blob_is_correct:
        status = "good"
    else:
        status = "bad"

    results = [blob_area, biggest_blob]
    return fails, location, results, status, resulting_images

def evaluate_blob_results(blob_area, biggest_blob, min_area, max_area, 
                          biggest_blob_max_area):
    """
    Evalúa si los resultados de blob están dentro del rango correcto.

    Args:
        blob_area (Union[float, int]): Área total de blob calculada.
        biggest_blob (Union[float, int]): Área del blob más grande.
        min_area (Union[float, int]): Área total mínima que se debe tener.
        max_area (Union[float, int]): Área total máxima que se debe tener.
        biggest_blob_max_area (Union[float, int]): Área máxima del blob más 
            grande.

    Returns:
        True si los resultados de blob están dentro del rango correcto; False
        si no lo están.
    """
    max_blob_size_passed = evaluate_blob_results_by_biggest_blob(biggest_blob, biggest_blob_max_area)
    blob_area_passed = evaluate_blob_results_by_blob_area(blob_area, min_area, max_area)

    if max_blob_size_passed and blob_area_passed:
        return True
    else:
        return False

def evaluate_blob_results_by_biggest_blob(biggest_blob, biggest_blob_max_area):
    """
    Evalúa si el área del blob más grande se encuentra dentro del rango correcto.

    Args:
        biggest_blob, biggest_blob_max_area - Ver ins_func.evaluate_blob_results()
    Returns:
        True si el área del blob más grande está dentro del rango correcto;
        False si no lo está.
    """
    if not biggest_blob_max_area or biggest_blob <= biggest_blob_max_area:
        return True
    else:
        return False

def evaluate_blob_results_by_blob_area(blob_area, min_area, max_area):
    """
    Evalúa si el área total de blob se encuentra dentro del rango correcto.

    Args:
        blob_area, min_area, max_area - Ver ins_func.evaluate_blob_results()

    Returns:
        True si el área total de blob está dentro del rango correcto; False si
        no lo está.
    """    
    if min_area and max_area:
        if(blob_area >= min_area and blob_area <= max_area):
            return True
        else:
            return False

    elif min_area:
        if(blob_area >= min_area):
            return True
        else:
            return False

    elif max_area:
        if(blob_area <= max_area):
            return True
        else:
            return False

def inspection_function_template_matching(inspection_image, algorithm):
    """
    Ejecuta y retorna los resultados de la función de inspección template matching.

    Args:
        inspection_image, algorithm - Ver ins_ref.execute_inspection_function()

    Returns:
        resulting_images: Imágenes del proceso de la función de inspección.
            Contiene imágenes cuyas keys son: 'filtered', 'color_converted', 'matches'.
            Más detalles en docstring de ins_ref.execute_inspection_function()
        location: Contiene las coordenadas (x,y) donde se encontró la coincidencia.
            Más detalles en docstring de ins_ref.execute_inspection_function()
        fails, location, results, status -
            Ver ins_ref.execute_inspection_function()
    """
    fails = []
    location = "not_available"
    status = ""
    resulting_images = {}

    resulting_images["filtered"] = inspection_image

    # resultados de cada template
    best_match_per_template = []
    matches_number_per_template = []

    correct_matches = False # flag
    for sub_template, min_calification in zip(
            algorithm["parameters"]["sub_templates"],
            algorithm["parameters"]["min_califications"]):

        try:
            matches_locations, best_match, color_converted_img, = find_template(
                inspection_image, sub_template, min_calification, 
                algorithm["parameters"],
            )
            resulting_images["color_converted"] = color_converted_img
        except excepts.AlgorithmError as e:
            status, fail = "error", str(e)
            fail += "-{0}".format(sub_template_number)
            fails.append(fail)

            best_match_per_template.append(None)
            matches_number_per_template.append(None)
            continue


        best_match_per_template.append(best_match)
        matches_number = len(matches_locations)
        matches_number_per_template.append(matches_number)

        if matches_number == algorithm["parameters"]["required_matches"]:
            correct_matches = True
            break

    results = [matches_number_per_template, best_match_per_template]

    # asignar status
    if not status == "error":
        if correct_matches: status = "good"
        else: status = "bad"

    # Si se encontró al menos una coincidencia, exportar imagen con las coincidencias marcadas
    if matches_number:

        # guardar localización de la coincidencia, si sólo se encontró y buscaba una
        if matches_number == 1 and algorithm["parameters"]["required_matches"] == 1:
            location = {
                "type":"pair_of_coordinates", 
                "coordinates":matches_locations[0],
            }

        matches_image = inspection_image.copy()
        # Dibujar rectángulos en las coincidencias
        for match_location in matches_locations:
            x = match_location[0]
            y = match_location[1]
            # Dibujar un rectángulos en la coincidencia
            rectangle(matches_image, (x, y),
                         (x+template_width, y+template_height),
                         (0, 255, 0), 2)

        resulting_images["matches"] = matches_image

    return fails, location, results, status, resulting_images

def find_template(inspection_image, sub_template, min_calification, parameters):
    """
    Encuentra una template para la función de inspección template matching.

    Args:
        sub_template (list): Contiene la imagen del subtemplate (np.ndarray) 
            en el índice 0, y el número del subtemplate en el índice 1.
        min_calification (Union[float, int]): Calificación mínima para considerar
            una coincidencia del template.
        parameters (dict): Contiene los parámetros de la función de inspección.
        inspection_image - Ver ins_ref.execute_inspection_function()

    Raises:
        excepts.AlgorithmError("TEMPLATE_DOESNT_EXIST"): Si la imagen del
            subtemplate no existe (es None).

    Returns:
        matches_locations, best_match, color_converted_img - 
            Ver cv_func.find_matches()
    """    
    best_match = 0
    matches_number = 0

    sub_template_image, sub_template_number = sub_template
    template_height, template_width = sub_template_image.shape[:2]

    if sub_template_image is None:
        status = "failed"
        raise excepts.AlgorithmError("TEMPLATE_DOESNT_EXIST") # !ALGORITHM_ERROR

    # Encontrar coincidencias
    matches_locations, best_match, color_converted_img = cv_func.find_matches(
        inspection_image, sub_template_image, min_calification,
        algorithm["parameters"]["required_matches"],
        algorithm["parameters"]["color_scale"],
        algorithm["parameters"]["color_scale_for_binary"],
        algorithm["parameters"]["color_range"],
        algorithm["parameters"]["invert_binary"],
    )

    return matches_locations, best_match, color_converted_img, sub_template_size

def inspection_function_unique_transition(inspection_image, algorithm):
    """
    Encuentra una única transición en X o en Y.
    Las imágenes a exportar cuando se utiliza transition son:
        Imagen filtrada, imagen con la transición dibujada.
    Retorna como resultados de algoritmo:
        Coordenada de la transición, diferencia de brillo en la transición
    """
    location = "not_available"
    status = "good" # inicializar como good
    fails = []
    resulting_images = {}

    resulting_images["filtered"] = inspection_image


    # encontrar transición
    coordinate, brightness_difference = cv_func.find_transition(
        inspection_image, algorithm["parameters"]["searching_orientation"],
        algorithm["parameters"]["min_difference"],
        algorithm["parameters"]["brightness_difference_type"],
        algorithm["parameters"]["group_size"],
    )

    if not coordinate:
        status = "bad"
        results = [None, None]
        return fails, results, status, resulting_images

    axis = cv_func.get_transition_axis(algorithm["parameters"]["searching_orientation"])

    location = {
        "type":"one_coordinate",
        "axis":axis,
        "coordinate":coordinate
    }

    # dibujar transición
    transition_drawn = cv_func.draw_transition(inspection_image, coordinate, axis)
    resulting_images["transition_drawn"] = transition_drawn

    results = [coordinate, brightness_difference]
    return fails, location, results, status, resulting_images


def inspection_function_transitions(inspection_image, algorithm):
    """
    Las imágenes a exportar cuando se utiliza transitions son:
        Imagen filtrada, imagen rgb con las transiciones encontradas dibujadas y
        el punto tomado como location.
    Retorna como resultados de algoritmo:
        Número de transiciones encontradas, ancho del componente, coordenadas de la transición.
    """
    location = "not_available"
    status = "good" # inicializar como good
    fails = []
    resulting_images = {}

    resulting_images["filtered"] = inspection_image


    # encontrar transiciones
    transitions_number, transitions = cv_func.find_transitions(
        inspection_image, algorithm["parameters"]["transitions_data"],
    )

    transitions_drawn_image = cv_func.draw_transitions(inspection_image, transitions)

    if transitions_number != algorithm["parameters"]["required_transitions_number"]:
        resulting_images["transitions_drawn"] = transitions_drawn_image
        status = "bad"
        results = [transitions_number, None]
        return fails, location, results, status, resulting_images


    # encontrar la localización del algoritmo
    fail, location = calculate_location_for_transitions(
        algorithm["parameters"]["required_transitions"], transitions
    )
    if fail:
        status = "failed"
        fails.append(fail)
        results = [transitions_number, None]
        return fails, location, results, status, resulting_images

    # dibujar la localización
    if location["type"] == "pair_of_coordinates":
        transitions_drawn_image = cv_func.draw_point(transitions_drawn_image, location["coordinates"], color=[0, 255, 255])

    resulting_images["transitions_drawn"] = transitions_drawn_image


    # ancho del componente
    component_width = None
    if algorithm["parameters"]["calculate_component_width"]:
        component_width = cv_func.calculate_distance_between_across_transitions(transitions["across1"], transitions["across2"])
        if algorithm["parameters"]["min_component_width"] <= component_width <= algorithm["parameters"]["max_component_width"]:
            status = "bad"

    results = [transitions_number, component_width]
    return fails, location, results, status, resulting_images

def calculate_location_for_transitions(required_transitions, transitions):
    if required_transitions_is_unique(required_transitions):
        transition_name = transitions.keys()[0]
        transition = transitions[transition_name]
        location = {
            "type":"one_coordinate", 
            "axis":transition["axis"],
            "coordinate":transition["coordinate"]
        }

    elif required_transitions_is_two_across(required_transitions):
        # punto medio entre across1 y across2
        middle_coordinate = int(round((transitions["across1"]["coordinate"] + transitions["across2"]["coordinate"])/2))
        # tiene el mismo eje de coordenadas que cualquiera de los 2 across
        middle_coordinate_axis = transitions["across1"]["axis"]
        location = {
            "type":"one_coordinate", 
            "axis":middle_coordinate_axis,
            "coordinate":middle_coordinate
        }

    else:
        fail, intersections = calculate_transitions_intersections(
            required_transitions, transitions
        )
        if fail:
            return fail, "not_available"

        if required_transitions_is_one_across_and_along(required_transitions):
            intersection = intersections
            location = {
                "type":"pair_of_coordinates", 
                "coordinates":intersection
            }
        
        elif required_transitions_is_two_across_and_along(required_transitions):
            [intersection1, intersection2] = intersections
            middle_point = math_functions.average_coordinates(intersection1, intersection2)
            location = {
                "type":"pair_of_coordinates", 
                "coordinates":middle_point
            }

    return None, location

def required_transitions_is_unique(required_transitions):
    if len(required_transitions) == 1:
        return True
    else:
        return False

def required_transitions_is_two_across(required_transitions):
    if required_transitions == ["across1", "across2"]:
        return True
    else:
        return False

def required_transitions_is_one_across_and_along(required_transitions):
    if (required_transitions == ["across1", "along"] or
            required_transitions == ["across2", "along"]):
        return True
    else:
        return False

def required_transitions_is_two_across_and_along(required_transitions):
    if required_transitions == ["across1", "across2", "along"]:
        return True
    else:
        return False

def calculate_transitions_intersections(required_transitions, transitions):
    """Retorna la intersección o intersecciones formadas por las transiciones
    introducidas."""
    fail = None
    intersections = None

    if required_transitions == ["across1", "along"]:
        intersections = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across1"])
        if intersections == None:
            fail = "TRANSITIONS_INTERSECTIONS_COULD_NOT_BE_CALCULATED" # !ALGORITHM_ERROR

    elif required_transitions == ["across2", "along"]:
        intersections = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across2"])
        if intersections == None:
            fail = "TRANSITIONS_INTERSECTIONS_COULD_NOT_BE_CALCULATED" # !ALGORITHM_ERROR

    elif required_transitions == ["across1", "across2", "along"]:
        # calcular centro entre los across y el along
        intersection1 = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across1"])
        intersection2 = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across2"])

        if intersection1 == None or intersection2 == None:
            fail = "TRANSITIONS_INTERSECTIONS_COULD_NOT_BE_CALCULATED" # !ALGORITHM_ERROR

        intersections = [intersection1, intersection2]
    else:
        fail = "TRANSITIONS_NAMES_CAN_NOT_INTERSECT" # !ALGORITHM_ERROR
    return fail, intersections


def inspection_function_histogram(inspection_image, algorithm):
    """
    Las imágenes a exportar cuando se utiliza histogram son:
        Imagen filtrada, imagen en escala de grises.
    Retorna como resultados de algoritmo:
        Porcentaje de área (pixeles en el rango de color), nivel de gris promedio,
        promedio de los N niveles de gris más bajos,
        promedio de los N niveles de gris más altos,
        nivel de gris más bajo, nivel de gris más alto.
    """
    location = "not_available"
    status = "good" # inicializar como good
    fails = []
    resulting_images = {}
    area_percentage, average_gray, average_lowest_gray, average_highest_gray, lowest_gray, highest_gray = None, None, None, None, None, None

    resulting_images["filtered"] = inspection_image


    gray_image = cvtColor(inspection_image, COLOR_BGR2GRAY)
    resulting_images["gray"] = gray_image


    if algorithm["parameters"]["min_area_percentage"] or algorithm["parameters"]["max_area_percentage"]:
        # calcular porcentaje de área que está entre el rango de gris
        area_percentage, hist = cv_func.calculate_area_percentage_with_histogram(
            gray_image,
            algorithm["parameters"]["lower_gray"], algorithm["parameters"]["upper_gray"],
        )


    if algorithm["parameters"]["min_average_gray"] or algorithm["parameters"]["max_average_gray"]:
        # calcular nivel de gris promedio
        average_gray = math_functions.average_array(gray_image)


    if (algorithm["parameters"]["average_of_lowest_gray_levels_number"] or
            algorithm["parameters"]["average_of_highest_gray_levels_number"] or
            algorithm["parameters"]["min_lowest_gray"] or algorithm["parameters"]["max_highest_gray"]
        ):
        # calcular promedio de los N niveles de gris más bajos y N niveles más altos
        sorted_gray_levels = math_functions.sort_array_elements_low_to_high(gray_image)

        if algorithm["parameters"]["average_of_lowest_gray_levels_number"]:
            # N más bajos
            lowest_gray_levels = sorted_gray_levels[:algorithm["parameters"]["average_of_lowest_gray_levels_number"]+1]
            average_lowest_gray = math_functions.average_list(lowest_gray_levels)

        if algorithm["parameters"]["average_of_highest_gray_levels_number"]:
            # N más altos
            highest_gray_levels = sorted_gray_levels[-(algorithm["parameters"]["average_of_highest_gray_levels_number"]):]
            average_highest_gray = math_functions.average_list(highest_gray_levels)

        if algorithm["parameters"]["min_lowest_gray"]:
            # nivel de gris más bajo
            lowest_gray = sorted_gray_levels[0]

        if algorithm["parameters"]["max_highest_gray"]:
            # nivel de gris más alto
            highest_gray = sorted_gray_levels[-1]


    # Evaluar el punto de inspección
    histogram_is_correct = evaluate_histogram_results(
        area_percentage, average_gray, average_lowest_gray, average_highest_gray,
        lowest_gray, highest_gray, algorithm
    )

    if histogram_is_correct:
        status = "good"
    else:
        status = "bad"

    results = [area_percentage, average_gray, average_lowest_gray,
        average_highest_gray, lowest_gray, highest_gray]
    return fails, location, results, status, resulting_images

def evaluate_histogram_results(area_percentage, average_gray, average_lowest_gray,
        average_highest_gray, lowest_gray, highest_gray, algorithm
    ):
    # si los resultados del histograma pasan correctamente todos los parámetros,
    # retornar verdadero

    # evaluar porcentaje de área
    if algorithm["parameters"]["min_area_percentage"] and algorithm["parameters"]["max_area_percentage"]:
        if algorithm["parameters"]["min_area_percentage"] <= area_percentage <= algorithm["parameters"]["max_area_percentage"]:
            area_percentage_is_correct = True
        else:
            area_percentage_is_correct = False
    elif algorithm["parameters"]["min_area_percentage"]:
        if algorithm["parameters"]["min_area_percentage"] <= area_percentage:
            area_percentage_is_correct = True
        else:
            area_percentage_is_correct = False
    elif algorithm["parameters"]["max_area_percentage"]:
        if area_percentage <= algorithm["parameters"]["max_area_percentage"]:
            area_percentage_is_correct = True
        else:
            area_percentage_is_correct = False
    else:
        area_percentage_is_correct = True

    # evaluar nivel de gris promedio
    if algorithm["parameters"]["min_average_gray"] and algorithm["parameters"]["max_average_gray"]:
        if algorithm["parameters"]["min_average_gray"] <= average_gray <= algorithm["parameters"]["max_average_gray"]:
            average_gray_is_correct = True
        else:
            average_gray_is_correct = False
    elif algorithm["parameters"]["min_average_gray"]:
        if algorithm["parameters"]["min_average_gray"] <= average_gray:
            average_gray_is_correct = True
        else:
            average_gray_is_correct = False
    elif algorithm["parameters"]["max_average_gray"]:
        if average_gray <= algorithm["parameters"]["max_average_gray"]:
            average_gray_is_correct = True
        else:
            average_gray_is_correct = False
    else:
        average_gray_is_correct = True

    # evaluar promedio de los N niveles de gris más bajos
    if algorithm["parameters"]["min_average_of_lowest_gray_levels"]:
        if average_lowest_gray >= algorithm["parameters"]["min_average_of_lowest_gray_levels"]:
            average_lowest_gray_is_correct = True
        else:
            average_lowest_gray_is_correct = False
    else:
        average_lowest_gray_is_correct = True

    # evaluar promedio de los N niveles de gris más altos
    if algorithm["parameters"]["max_average_of_highest_gray_levels"]:
        if average_highest_gray <= algorithm["parameters"]["max_average_of_highest_gray_levels"]:
            average_highest_gray_is_correct = True
        else:
            average_highest_gray_is_correct = False
    else:
        average_highest_gray_is_correct = True

    # evaluar nivel de gris más bajo
    if algorithm["parameters"]["min_lowest_gray"]:
        if lowest_gray >= algorithm["parameters"]["min_lowest_gray"]:
            lowest_gray_is_correct = True
        else:
            lowest_gray_is_correct = False
    else:
        lowest_gray_is_correct = True

    # evaluar nivel de gris más alto
    if algorithm["parameters"]["max_highest_gray"]:
        if highest_gray <= algorithm["parameters"]["max_highest_gray"]:
            highest_gray_is_correct = True
        else:
            highest_gray_is_correct = False
    else:
        highest_gray_is_correct = True

    if (area_percentage_is_correct and average_gray_is_correct and
            average_lowest_gray_is_correct and average_highest_gray_is_correct and
            lowest_gray_is_correct and highest_gray_is_correct):
        return True
    else:
        return False

"""
Hacer que, cuando una cadena se rompa, no compruebe si se debe ejecutar uno por uno los algoritmos.
Es decir, si hay 4 algoritmos encadenados en secuencia, si falla el 2do, que el 3ro y el 4to
ni siquiera entren a un 'if' para comprobar si deben ejecutarse.
"""
