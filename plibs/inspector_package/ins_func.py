from inspector_package import math_functions, cv_func, operations, results_management, excepts

from cv2 import imread, imwrite, rectangle, cvtColor, COLOR_BGR2GRAY
from numpy import array

TEMPLATE_MATCHING = "m"
BLOB = "b"
UNIQUE_TRANSITION = "ut"
TRANSITIONS = "t"
HISTOGRAM = "h"

CLASSIFICATION = "classification"

def get_inspection_function_parameters(algorithm, parameters_data):
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
    parameters = {
        "invert_binary": parameters_data[0],
        "color_scale": parameters_data[1],
        "lower_color": parameters_data[2],
        "upper_color": parameters_data[3],
        "min_blob_size": parameters_data[4],
        "max_blob_size": parameters_data[5],
        "min_area": parameters_data[6],
        "max_area": parameters_data[7],
        "max_allowed_blob_size": parameters_data[8],
    }

    if parameters["color_scale"] == "hsv":
        # convertir a array de numpy
        parameters["lower_color"] = array(parameters["lower_color"])
        parameters["upper_color"] = array(parameters["upper_color"])

    return parameters


def get_template_matching_parameters(algorithm, parameters_data):
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
    for i in range(number_of_sub_templates):
        sub_template_image = imread(template_path + "-" + str(i+1) + ".bmp")
        sub_template_image = cv_func.apply_filters(sub_template_image, filters)
        sub_templates.append([sub_template_image, i])
    return sub_templates


def get_unique_transition_parameters(algorithm, parameters_data):
    parameters = {
        "searching_orientation":parameters_data[0],
        "min_difference":parameters_data[1],
        "brightness_difference_type":parameters_data[2],
        "group_size":parameters_data[3],
    }
    return parameters


def get_histogram_parameters(algorithm, parameters_data):
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


def get_transitions_parameters(algorithm, parameters_data):
    parameters = {
        "calculate_component_width": parameters_data[0],
        "min_component_width": parameters_data[1],
        "max_component_width": parameters_data[2],
    }

    transitions_data = parameters_data[3]
    parameters["transitions_data"] = get_transitions_data(transitions_data)

    fail, parameters["required_transitions"] = determinate_required_transitions(parameters["transitions_data"])
    if fail:
        print("FAIL IN TRANSITIONS PARAMETERS:", fail)
        sys.exit() # exit error

    parameters["required_transitions_number"] = len(parameters["required_transitions"])

    return parameters

def determinate_required_transitions(transitions_data):
    fail = None
    transitions_number = len(transitions_data)

    if transitions_number == 1 and "unique" in transitions_data.keys():
        required_transitions = "unique"
    elif (transitions_number == 2 and "across1" in transitions_data.keys() and
            "across2" in transitions_data.keys()):
        required_transitions = ["across1", "across2"]
    elif (transitions_number == 2 and "across1" in transitions_data.keys() and
            "along" in transitions_data.keys()):
        required_transitions = ["across1", "along"]
    elif (transitions_number == 2 and "across2" in transitions_data.keys() and
            "along" in transitions_data.keys()):
        required_transitions = ["across2", "along"]
    elif (transitions_number == 3 and "across1" in transitions_data.keys() and
            "across2" in transitions_data.keys() and "along" in transitions_data.keys()):
        required_transitions = ["across1", "across2", "along"]
    else:
        fail = "REQUIRED_TRANSITIONS_NOT_VALID"
    return fail, required_transitions

def get_transitions_data(transitions_data):
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


def create_algorithm(algorithm_data):
    algorithm = {
        "ignore_bad_status":algorithm_data[0],
        "needed_status_to_be_good":algorithm_data[1],
        "take_as_origin":algorithm_data[3],
        "light":algorithm_data[4], # white / ultraviolet
        "name":algorithm_data[5],
        "coordinates":algorithm_data[6],
        "boards_without_this_algorithm":algorithm_data[7],
        "inspection_function":algorithm_data[8],
        "filters":algorithm_data[10],
    }

    chain_data = algorithm_data[2]
    [algorithm["chained_to"], algorithm["needed_status_of_chained_to_execute"]] = chain_data

    # parámetros de la función de inspección del punto (áreas de blob, templates de
    # template matching, etc.)
    parameters_data = algorithm_data[9]
    algorithm["parameters"] = get_inspection_function_parameters(algorithm, parameters_data)

    return algorithm

def create_algorithms(data):
    algorithms = []
    for algorithm_data in data:
        algorithm = create_algorithm(algorithm_data)
        algorithms.append(algorithm)
    return algorithms


def create_inspection_point(inspection_point_data):
    inspection_point = {
        "name":inspection_point_data[0],
        "coordinates":inspection_point_data[1],
    }

    algorithms_data = inspection_point_data[2]
    inspection_point["algorithms"] = create_algorithms(algorithms_data)

    return inspection_point

def create_inspection_points(data):
    inspection_points = []
    for inspection_point_data in data:
        inspection_point = create_inspection_point(inspection_point_data)
        inspection_points.append(inspection_point)
    return inspection_points


def get_reference_algorithm(reference_algorithm_data):
    reference_algorithm = {}
    reference_algorithm["function"] = reference_algorithm_data[0]

    if reference_algorithm["function"] == "classification":
        pass
    if reference_algorithm["function"] == "displacement":
        reference_algorithm["algorithm1_to_measure"] = reference_algorithm_data[1]
        reference_algorithm["algorithm2_to_measure"] = reference_algorithm_data[2]
        reference_algorithm["x_translation_tolerance"] = reference_algorithm_data[3]
        reference_algorithm["y_translation_tolerance"] = reference_algorithm_data[4]
        reference_algorithm["rotation_tolerance"] = reference_algorithm_data[5]
        reference_algorithm["min_width"] = reference_algorithm_data[6]
        reference_algorithm["max_width"] = reference_algorithm_data[7]

    return reference_algorithm

def create_reference(reference_data):
    reference = {
        "name":reference_data[0],
        "boards_without_this_reference":reference_data[2],
        "inspection_points":create_inspection_points(reference_data[3]),
    }

    reference_algorithm_data = reference_data[1]
    reference["reference_algorithm"] = get_reference_algorithm(reference_algorithm_data)

    return reference

def create_references(data):
    references = []
    for reference_data in data:
        reference = create_reference(reference_data)
        references.append(reference)
    return references


def execute_algorithm(inspection_image_filt, algorithm):
    if (algorithm["inspection_function"] == BLOB):
        fails, location, results, status, resulting_images = \
            inspection_function_blob(inspection_image_filt, algorithm)

    elif (algorithm["inspection_function"] == TEMPLATE_MATCHING):
        fails, location, results, status, resulting_images = \
            inspection_function_template_matching(inspection_image_filt, algorithm)

    elif (algorithm["inspection_function"] == UNIQUE_TRANSITION):
        fails, location, results, status, resulting_images = \
            inspection_function_unique_transition(inspection_image_filt, algorithm)

    elif (algorithm["inspection_function"] == TRANSITIONS):
        fails, location, results, status, resulting_images = \
            inspection_function_transitions(inspection_image_filt, algorithm)

    elif (algorithm["inspection_function"] == HISTOGRAM):
        fails, location, results, status, resulting_images = \
            inspection_function_histogram(inspection_image_filt, algorithm)

    else:
        return ["INVALID_INSPECTION_FUNCTION"], "not_available", [], "failed", []

    return fails, location, results, status, resulting_images

def calculate_location_inside_algorithm_in_photo(inspection_point, algorithm, location):
    """Localización encontrada con un algoritmo (que toma el origen de la
    ventana del algoritmo) tomando como origen el (0,0) de la foto."""

    if location["type"] == "one_coordinate":
        x1,y1,x2,y2 = algorithm["coordinates"]
        if location["axis"] == "x":
            window_height = y2-y1
            # tomar como «y» la mitad del alto de la ventana
            location["coordinates"] = [location["coordinate"], int(window_height/2)]
        if location["axis"] == "y":
            window_width = x2-x1
            # tomar como «x» la mitad del ancho de la ventana
            location["coordinates"] = [int(window_width)/2, location["coordinate"]]

    # localización = localización dentro de la ventana del algoritmo +
    # coordenadas del algoritmo + coordenadas del punto de inspección
    location["coordinates"] = math_functions.sum_lists(location["coordinates"],
        math_functions.sum_lists(inspection_point["coordinates"],
        algorithm["coordinates"][:2])
    )
    return location["coordinates"]

def inspect_inspection_points(image, image_ultraviolet, board, inspection_points, check_mode="check:no"):
    inspection_points_results = {
        "results":[], "string":"", "status":"good", "images":[],
    }

    for inspection_point in inspection_points:
        inspection_point_results = {
            "status":"good", "results":[],
        }
        algorithms_results = {
            "results":{}, "algorithms_status":{}, "locations":{}, "string":"", "images":[],
        }

        for algorithm in inspection_point["algorithms"]:
            algorithm_results = {
                "results":[], "string":"", "status":"", "location":{}, "images":[], "fails":[]
            }

            if board.get_position_in_photo() in algorithm["boards_without_this_algorithm"]:
                continue

            # si el algoritmo ya fue inspeccionado (está en los resultados y su status no es "not_executed"),
            # no volver a inspeccionarlo
            if algorithm["name"] in algorithms_results["algorithms_status"] and algorithms_results.get(algorithm["name"]) != "not_executed":
                continue


            # verificar si se ejecutará o no el algoritmo, dependiendo de la cadena
            if (algorithm["chained_to"] != None and
                    not algorithms_results["algorithms_status"][algorithm["chained_to"]] == algorithm["needed_status_of_chained_to_execute"]):

                # abortar inspección del algoritmo y marcarlo "not_executed"
                # no agregarlo al string de resultados

                algorithm_results = results_management.get_algorithm_results(
                    algorithm_results=algorithm_results, algorithm=algorithm,
                    results=[], status="not_executed", fails=[],
                    location="not_available", images=[]
                )

                algorithms_results = results_management.add_algorithm_results_to_algorithms_results(
                    algorithm, algorithm_results, algorithms_results, add_string=False
                )

                continue


            # determinar el origen de las coordenadas del algoritmo
            if algorithm["take_as_origin"] == "$inspection_point":
                # tomar el punto de inspección
                origin = inspection_point["coordinates"]
            else:
                # tomar otro algoritmo
                origin = algorithms_results["locations"][algorithm["take_as_origin"]]
                if origin == "not_available":
                    # abortar inspección del algoritmo y marcarlo "failed"

                    algorithm_results = results_management.get_algorithm_results(
                        algorithm_results=algorithm_results, algorithm=algorithm,
                        results=[], status="failed",
                        fails=["ALGO_TO_TAKE_AS_ORIGIN_DOESNT_HAVE_COORDINATES"], # !FAIL
                        location="not_available", images=[],
                    )
                    algorithms_results = results_management.add_algorithm_results_to_algorithms_results(algorithm, algorithm_results, algorithms_results)

                    # cambiar el status del punto de inspección si es necesario
                    inspection_point_results["status"] = results_management.evaluate_inspection_point_status(
                    algorithm_results["status"], inspection_point_results["status"], algorithm,
                    )

                    break


            if algorithm["light"] == "ultraviolet":
                inspection_image = cv_func.crop_image(image_ultraviolet,algorithm["coordinates"],
                    take_as_origin=origin)
            else:
                inspection_image = cv_func.crop_image(image,algorithm["coordinates"],
                    take_as_origin=origin)

            # filtrar imágenes
            inspection_image_filt = cv_func.apply_filters(
                inspection_image, algorithm["filters"]
            )

            # ejecutar algoritmo
            [fails, location, results, inspection_function_status, resulting_images] = \
                execute_algorithm(inspection_image_filt, algorithm)

            if location != "not_available":
                # guardar localización tomando como origen el (0,0) del tablero
                location = calculate_location_inside_algorithm_in_photo(
                    inspection_point, algorithm, location,
                )

            status = results_management.evaluate_algorithm_status(inspection_function_status, algorithm)

            algorithm_results = results_management.get_algorithm_results(
                algorithm_results=algorithm_results, algorithm=algorithm,
                results=results, status=status, fails=fails, location=location,
                images=resulting_images
            )

            algorithms_results = results_management.add_algorithm_results_to_algorithms_results(algorithm, algorithm_results, algorithms_results)


            # cambiar el status del punto de inspección si es necesario
            inspection_point_results["status"] = results_management.evaluate_inspection_point_status(
            algorithm_results["status"], inspection_point_results["status"], algorithm,
            )


        # cambiar el status de los puntos de inspección si es necesario
        inspection_points_results["status"] = results_management.evaluate_status(
            inspection_point_results["status"], inspection_points_results["status"]
        )

        # agregar resultados del punto de inspección a los resultados de todos los puntos
        inspection_points_results["results"] += inspection_point_results["results"]

        inspection_point_results["string"] = results_management.create_inspection_point_results_string(
            algorithms_results["string"], inspection_point["name"], inspection_point_results["status"]
        )
        inspection_points_results["string"] += inspection_point_results["string"]

        # exportar imágenes si: a) check:yes y el status del punto es malo, o
        # b) check:total
        if ((check_mode == "check:yes" and inspection_point_results["status"] == "bad") or
                check_mode == "check:total"):
            # agregar imágenes del punto de inspección imágenes de todos los puntos
            inspection_points_results["images"].append([inspection_point["name"], algorithms_results["images"]])

    return inspection_points_results


def reference_algorithm_classification(inspection_points_results):
    # sólo retorna el status de los puntos de inspección
    fails = []
    results = [inspection_points_results["status"]]
    status = inspection_points_results["status"]
    return fails, results, status

def execute_reference_algorithm(reference_algorithm, inspection_points_results):
    if (reference_algorithm["function"] == CLASSIFICATION):
        fails, results, status = reference_algorithm_classification(inspection_points_results)

    else:
        fails, results, status = ["INVALID_INSPECTION_FUNCTION"], [], "failed"

    return fails, results, status


def inspect_reference(image, board, reference, check_mode, images_path , image_ultraviolet=None):
    reference_results = {"string":"", "status":"good"}

    inspection_points_results = inspect_inspection_points(image, image_ultraviolet, board, reference["inspection_points"], check_mode)

    # cambiar el status de la referencia si es necesario
    reference_results["status"] = results_management.evaluate_status(
        inspection_points_results["status"], reference_results["status"]
    )

    operations.export_reference_images(inspection_points_results["images"], board.get_photo_number(), board.get_position_in_photo(), reference["name"], images_path)

    reference_algorithm_results = { "status":"", "results":[], "fails":[] }

    # ejecutar el algoritmo de la referencia
    reference_algorithm_results["fails"], \
    reference_algorithm_results["results"], \
    reference_algorithm_results["status"], = \
        execute_reference_algorithm(
            reference["reference_algorithm"], inspection_points_results
        )

    # cambiar el status de la referencia si es necesario
    reference_results["status"] = results_management.evaluate_status(
        reference_algorithm_results["status"], reference_results["status"]
    )

    # cambiar el status del tablero si es necesario
    board.evaluate_status(reference_results["status"])


    # resultados del algoritmo de la referencia
    reference_algorithm_results["string"] = results_management.create_reference_algorithm_results_string(
        reference_algorithm_results["status"], reference_algorithm_results["results"]
    )

    # resultados de la referencia
    reference_results["string"] = results_management.create_reference_results_string(
        inspection_points_results["string"], reference["name"], reference_results["status"], reference_algorithm_results["string"]
    )
    return reference_results["string"]

def inspect_references(first_reference, last_reference,
        image, board, references, stage, check_mode, image_ultraviolet=None,
    ):

    if stage == "debug":
        images_path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/"
    elif stage == "inspection":
        images_path = "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/bad_windows_results/"

    # se le resta 1 a la posición de las referencias para obtener su í­ndice en la lista
    first_reference -= 1
    last_reference -= 1
    # la función range toma desde first hasta last-1, así­ que hay que sumarle 1
    references = references[first_reference:last_reference+1]

    references_results_string = ""
    for reference in references:
        if board.get_position_in_photo() in reference["boards_without_this_reference"]:
            continue

        reference_results_string = inspect_reference(image, board, reference,
            check_mode, images_path, image_ultraviolet
        )
        references_results_string += reference_results_string

    board.add_references_results(references_results_string)


def inspection_function_blob(inspection_image, algorithm):
    """
    Las imágenes a exportar cuando se utiliza blob son:
    imagen filtrada, imagen binarizada.
    """
    location = "not_available"
    status = ""
    fails = []
    images_to_return = {}

    images_to_return["filtered"] = inspection_image

    blob_area, biggest_blob, binary_image = cv_func.calculate_blob_area(
        inspection_image,
        algorithm["parameters"]["lower_color"],
        algorithm["parameters"]["upper_color"],
        algorithm["parameters"]["color_scale"],
        algorithm["parameters"]["invert_binary"],
    )

    images_to_return["binary"] = binary_image

    # Evaluar el punto de inspección
    blob_is_correct = evaluate_blob_results(
        blob_area, biggest_blob,
        algorithm["parameters"]["min_area"],
        algorithm["parameters"]["max_area"],
        algorithm["parameters"]["max_allowed_blob_size"]
    )

    if blob_is_correct:
        status = "good"
    else:
        status = "bad"

    window_results = [blob_area, biggest_blob]
    return fails, location, window_results, status, images_to_return

def evaluate_blob_results(blob_area, biggest_blob, min_area, max_area, max_allowed_blob_size):
    # evaluar con máximo tamaño de blob permitido
    max_blob_size_passed = evaluate_blob_results_by_blob_size(biggest_blob, max_allowed_blob_size)
    # evaluar con área total de blobs
    blob_area_passed = evaluate_blob_results_by_blob_area(blob_area, min_area, max_area)

    if max_blob_size_passed and blob_area_passed:
        return True
    else:
        return False

def evaluate_blob_results_by_blob_size(biggest_blob, max_allowed_blob_size):
    if not max_allowed_blob_size:
        return True
    if(biggest_blob <= max_allowed_blob_size):
        return True
    else:
        return False

def evaluate_blob_results_by_blob_area(blob_area, min_area, max_area):
    # Evaluar con 3 opciones el área total de blobs:
    # Si hay área mí­nima y máxima
    # Si hay área mí­nima
    # Si hay área máxima

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
    Las imágenes a exportar cuando se utiliza template matching son:
    imagen filtrada,
    imagen rgb con las coincidencias encontradas marcadas.
    """
    # Inspeccionar con template matching usando sub-templates
    location = "not_available"
    status = ""
    fails = []
    images_to_return = {}

    images_to_return["filtered"] = inspection_image

    # listas con los resultados de cada template
    best_match_per_template = [] # mejor coincidencia de cada una
    matches_number_per_template = [] # núm de coincidencias de cada una
    correct_matches_number = False
    for sub_template,min_calification in zip(
            algorithm["parameters"]["sub_templates"],
            algorithm["parameters"]["min_califications"]
        ):
        best_match = 0
        matches_number = 0

        sub_template_image, sub_template_index = sub_template

        if sub_template_image is None:
            status = "failed"
            fail = "TEMPLATE_DOESNT_EXIST-{0}".format(sub_template_index+1) # !FAIL
            fails.append(fail)
            continue

        # Dimensiones del template
        template_height = sub_template_image.shape[0]
        template_width = sub_template_image.shape[1]

        # Encontrar coincidencias
        try:
            matches_locations, best_match, color_converted_img = cv_func.find_matches(
                inspection_image, sub_template_image, min_calification,
                algorithm["parameters"]["required_matches"],
                algorithm["parameters"]["color_scale"],
                algorithm["parameters"]["color_scale_for_binary"],
                algorithm["parameters"]["color_range"],
                algorithm["parameters"]["invert_binary"],
            )
            images_to_return["color_converted"] = color_converted_img
        except excepts.MT_ERROR:
            best_match_per_template.append(None)
            matches_number_per_template.append(None)
            status = "failed"
            fail = "MT_ERROR-{}".format(sub_template_index+1) # !FAIL
            fails.append(fail)
        except excepts.UNKNOWN_CF_ERROR:
            best_match_per_template.append(None)
            matches_number_per_template.append(None)
            status = "failed"
            fail = "UNKNOWN_CF_ERROR-{}".format(sub_template_index+1) # !FAIL
            fails.append(fail)

        except Exception as fail:
            best_match_per_template.append(None)
            matches_number_per_template.append(None)
            status = "failed"
            fails.append(str(fail))
            continue

        # agregar a las listas de resultados de cada template
        best_match_per_template.append(best_match) # mejor match de cada template
        matches_number = len(matches_locations)
        matches_number_per_template.append(matches_number) # número de matches de cada template

        # Evaluar el punto de inspección
        if(matches_number == algorithm["parameters"]["required_matches"]):
            correct_matches_number = True
            status = "good"
            break

    window_results = [matches_number_per_template, best_match_per_template]

    # Si no se encontraron las coincidencia necesarias con ninguna subtemplate
    if not status == "good" and status != "failed":
        status = "bad"

    # Si se encontró al menos una coincidencia, exportar imagen con las coincidencias marcadas
    if matches_number:

        # guardar localización de la coincidencia, si sólo se encontró y buscaba una
        if matches_number == 1 and algorithm["parameters"]["required_matches"] == 1:
            location = {"type":"pair_of_coordinates", "coordinates":matches_locations[0]}

        matches_image = inspection_image.copy()
        # Dibujar rectángulos en las coincidencias
        for match_location in matches_locations:
            x = match_location[0]
            y = match_location[1]
            # Dibujar un rectángulos en la coincidencia
            rectangle(matches_image, (x, y),
                         (x+template_width, y+template_height),
                         (0, 255, 0), 2)

        images_to_return["matches"] = matches_image

    return fails, location, window_results, status, images_to_return


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
    images_to_return = {}

    images_to_return["filtered"] = inspection_image


    # encontrar transición
    coordinate, brightness_difference = cv_func.find_transition(
        inspection_image, algorithm["parameters"]["searching_orientation"],
        algorithm["parameters"]["min_difference"],
        algorithm["parameters"]["brightness_difference_type"],
        algorithm["parameters"]["group_size"],
    )

    if not coordinate:
        status = "bad"
        window_results = [None, None]
        return fails, window_results, status, images_to_return

    axis = cv_func.get_transition_axis(algorithm["parameters"]["searching_orientation"])

    location = {
        "type":"one_coordinate",
        "axis":axis,
        "coordinate":coordinate
    }

    # dibujar transición
    transition_drawn = cv_func.draw_transition(inspection_image, coordinate, axis)
    images_to_return["transition_drawn"] = transition_drawn

    window_results = [coordinate, brightness_difference]
    return fails, location, window_results, status, images_to_return


def inspection_function_transitions(inspection_image, algorithm):
    """
    Las imágenes a exportar cuando se utiliza transitions son:
        Imagen filtrada, imagen rgb con las transiciones encontradas dibujadas y
        el punto tomado como location.
    Retorna como resultados de algoritmo:
        Número de transiciones encontradas, ancho del componente.
    """
    location = "not_available"
    status = "good" # inicializar como good
    fails = []
    images_to_return = {}

    images_to_return["filtered"] = inspection_image


    # encontrar transiciones
    transitions_number, transitions = cv_func.find_transitions(
        inspection_image, algorithm["parameters"]["transitions_data"],
    )

    transitions_drawn_image = cv_func.draw_transitions(inspection_image, transitions)

    if transitions_number != algorithm["parameters"]["required_transitions_number"]:
        images_to_return["transitions_drawn"] = transitions_drawn_image
        status = "bad"
        window_results = [transitions_number, None]
        return fails, location, window_results, status, images_to_return


    # encontrar la localización del algoritmo
    fail, location = calculate_location_for_transitions(
        algorithm["parameters"]["required_transitions"], transitions
    )
    if fail:
        status = "failed"
        fails.append(fail)
        window_results = [transitions_number, None]
        return fails, location, window_results, status, images_to_return

    # dibujar la localización
    if location["type"] == "pair_of_coordinates":
        transitions_drawn_image = cv_func.draw_point(transitions_drawn_image, location["coordinates"], color=[0, 255, 255])

    images_to_return["transitions_drawn"] = transitions_drawn_image


    # ancho del componente
    component_width = None
    if algorithm["parameters"]["calculate_component_width"]:
        component_width = cv_func.calculate_distance_between_across_transitions(transitions["across1"], transitions["across2"])
        if algorithm["parameters"]["min_component_width"] <= component_width <= algorithm["parameters"]["max_component_width"]:
            status = "bad"

    window_results = [transitions_number, component_width]
    return fails, location, window_results, status, images_to_return

def calculate_location_for_transitions(required_transitions, transitions):
    if required_transitions_is_unique(required_transitions):
        transition = transitions["unique"]
        location = {"type":"one_coordinate", "axis":transition["axis"],
            "coordinate":transition["coordinate"]
        }

    elif required_transitions_is_two_across(required_transitions):
        # punto medio entre across1 y across2
        middle_coordinate = int(round((transitions["across1"]["coordinate"] + transitions["across2"]["coordinate"])/2))
        # tiene el mismo eje de coordenadas que cualquiera de los 2 across
        middle_coordinate_axis = transitions["across1"]["axis"]
        location = {"type":"one_coordinate", "axis":middle_coordinate_axis,
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
            location = {"type":"pair_of_coordinates", "coordinates":intersection}

        elif required_transitions == ["across1", "across2", "along"]:
            [intersection1, intersection2] = intersections
            middle_point = math_functions.average_coordinates(intersection1, intersection2)
            location = {"type":"pair_of_coordinates", "coordinates":middle_point}

    return None, location

def required_transitions_is_unique(required_transitions):
    if required_transitions == "unique":
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

def calculate_transitions_intersections(required_transitions, transitions):
    """Retorna la intersección o intersecciones formadas por las transiciones
    introducidas."""
    fail = None
    intersections = None

    if required_transitions == ["across1", "along"]:
        intersections = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across1"])
        if intersections == None:
            fail = "TRANSITIONS_INTERSECTIONS_COULD_NOT_BE_CALCULATED" # !FAIL

    elif required_transitions == ["across2", "along"]:
        intersections = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across2"])
        if intersections == None:
            fail = "TRANSITIONS_INTERSECTIONS_COULD_NOT_BE_CALCULATED" # !FAIL

    elif required_transitions == ["across1", "across2", "along"]:
        # calcular centro entre los across y el along
        intersection1 = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across1"])
        intersection2 = cv_func.calculate_transitions_intersection(transitions["along"], transitions["across2"])

        if intersection1 == None or intersection2 == None:
            fail = "TRANSITIONS_INTERSECTIONS_COULD_NOT_BE_CALCULATED" # !FAIL

        intersections = [intersection1, intersection2]
    else:
        fail = "TRANSITIONS_NAMES_CAN_NOT_INTERSECT" # !FAIL
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
    images_to_return = {}
    area_percentage, average_gray, average_lowest_gray, average_highest_gray, lowest_gray, highest_gray = None, None, None, None, None, None

    images_to_return["filtered"] = inspection_image


    gray_image = cvtColor(inspection_image, COLOR_BGR2GRAY)
    images_to_return["gray"] = gray_image


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
        lowest_gray, highest_gray, algorithm["parameters"]
    )

    if histogram_is_correct:
        status = "good"
    else:
        status = "bad"

    window_results = [area_percentage, average_gray, average_lowest_gray,
        average_highest_gray, lowest_gray, highest_gray]
    return fails, location, window_results, status, images_to_return

def evaluate_histogram_results(area_percentage, average_gray, average_lowest_gray,
        average_highest_gray, lowest_gray, highest_gray, parameters
    ):
    # si los resultados del histograma pasan correctamente todos los parámetros,
    # retornar verdadero

    # evaluar porcentaje de área
    if parameters["min_area_percentage"] and parameters["max_area_percentage"]:
        if parameters["min_area_percentage"] <= area_percentage <= parameters["max_area_percentage"]:
            area_percentage_is_correct = True
        else:
            area_percentage_is_correct = False
    elif parameters["min_area_percentage"]:
        if parameters["min_area_percentage"] <= area_percentage:
            area_percentage_is_correct = True
        else:
            area_percentage_is_correct = False
    elif parameters["max_area_percentage"]:
        if area_percentage <= parameters["max_area_percentage"]:
            area_percentage_is_correct = True
        else:
            area_percentage_is_correct = False
    else:
        area_percentage_is_correct = True

    # evaluar nivel de gris promedio
    if parameters["min_average_gray"] and parameters["max_average_gray"]:
        if parameters["min_average_gray"] <= average_gray <= parameters["max_average_gray"]:
            average_gray_is_correct = True
        else:
            average_gray_is_correct = False
    elif parameters["min_average_gray"]:
        if parameters["min_average_gray"] <= average_gray:
            average_gray_is_correct = True
        else:
            average_gray_is_correct = False
    elif parameters["max_average_gray"]:
        if average_gray <= parameters["max_average_gray"]:
            average_gray_is_correct = True
        else:
            average_gray_is_correct = False
    else:
        average_gray_is_correct = True

    # evaluar promedio de los N niveles de gris más bajos
    if parameters["min_average_of_lowest_gray_levels"]:
        if average_lowest_gray >= parameters["min_average_of_lowest_gray_levels"]:
            average_lowest_gray_is_correct = True
        else:
            average_lowest_gray_is_correct = False
    else:
        average_lowest_gray_is_correct = True

    # evaluar promedio de los N niveles de gris más altos
    if parameters["max_average_of_highest_gray_levels"]:
        if average_highest_gray <= parameters["max_average_of_highest_gray_levels"]:
            average_highest_gray_is_correct = True
        else:
            average_highest_gray_is_correct = False
    else:
        average_highest_gray_is_correct = True

    # evaluar nivel de gris más bajo
    if parameters["min_lowest_gray"]:
        if lowest_gray >= parameters["min_lowest_gray"]:
            lowest_gray_is_correct = True
        else:
            lowest_gray_is_correct = False
    else:
        lowest_gray_is_correct = True

    # evaluar nivel de gris más alto
    if parameters["max_highest_gray"]:
        if highest_gray <= parameters["max_highest_gray"]:
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
