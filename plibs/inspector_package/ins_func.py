from inspector_package import math_functions, cv_func, operations, results_management

from cv2 import imread, imwrite, rectangle
from numpy import array

TEMPLATE_MATCHING = "m"
BLOB = "b"
TRANSITION = "t"


def get_inspection_function_parameters(inspection_point, parameters_data):
    # blob
    if (inspection_point["inspection_function"] == "b"):
        parameters = get_blob_parameters(inspection_point, parameters_data)
    # template matching
    elif (inspection_point["inspection_function"] == "m"):
        parameters = get_template_matching_parameters(inspection_point, parameters_data)
    return parameters

def get_blob_parameters(inspection_point, parameters_data):
    parameters = {
        "invert_binary": parameters_data[0],
        "color_scale": parameters_data[1],
        "lower_color": array(parameters_data[2][0]),
        "upper_color": array(parameters_data[2][1]),
        "min_blob_size": parameters_data[3],
        "max_blob_size": parameters_data[4],
        "min_area": parameters_data[5],
        "max_area": parameters_data[6],
        "max_allowed_blob_size": parameters_data[7],
    }
    return parameters

def get_template_matching_parameters(inspection_point, parameters_data):
    color_scale = parameters_data[0]
    if type(color_scale) is str:
        name, invert_binary, color_scale_for_binary, color_range = color_scale, None, None, None
    elif type(color_scale) is list:
        [name, invert_binary, color_scale_for_binary, color_range] = color_scale

    template_path = parameters_data[1]
    number_of_sub_templates = parameters_data[2]
    sub_templates = get_sub_templates(
        number_of_sub_templates, template_path, inspection_point["filters"]
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
        sub_template_img = imread(template_path + "-" + str(i+1) + ".bmp")
        sub_template_img = cv_func.apply_filters(sub_template_img, filters)
        sub_templates.append([sub_template_img, i])
    return sub_templates

def get_transitions_parameters(inspection_point, parameters_data):
    parameters = {

    }
    return parameters


def create_algorithm(algorithm_data):
    algorithm = {
        "necessary_status":algorithm_data[0],
        "algorithm_to_take_as_origin":algorithm_data[1],
        "light":algorithm_data[2], # white / ultraviolet
        "name":algorithm_data[3],
        "coordinates":algorithm_data[4],
        "inspection_function":algorithm_data[5],
        "filters":algorithm_data[7],
    }

    # parámetros de la función de inspección del punto (áreas de blob, templates de template matching, etc.)
    parameters_data = algorithm_data[6]
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
        "inspection_points":create_inspection_points(reference_data[2]),
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

    else:
        return ["INVALID_INSPECTION_FUNCTION"], None, None, None

    return fails, location, results, status, resulting_images


def inspect_inspection_points(image, image_ultraviolet, inspection_points, check_mode="check:no"):
    inspection_points_status = "good"
    inspection_points_results = []
    inspection_points_results_string = ""
    images_to_export = []

    for inspection_point in inspection_points:
        inspection_point_status = "good"
        inspection_point_results = []
        algorithms_results = {}
        algorithms_results_string = ""
        algorithms_locations = {}
        inspection_point_images = []

        for algorithm in inspection_point["algorithms"]:
            # coordenadas del origen de las coordenadas del algoritmo
            if algorithm["algorithm_to_take_as_origin"] == "$inspection_point":
                origin = inspection_point["coordinates"]
            else:
                algorithm_to_take_as_origin_coordinates = algorithms_locations[algorithm["algorithm_to_take_as_origin"]]

                # si el punto de inspección que se desea tomar como origen no
                # almacenó una localización, salir de la inspección del punto
                if algorithm_to_take_as_origin_coordinates == "not_available":
                    # agregar resultados del algoritmo como si fuera "failed"
                    # y un código de fallo a los resultados de los algoritmos
                    algorithm_results_string = results_management.create_algorithm_results(
                        name=algorithm["name"], light=algorithm["light"], status="failed",
                        results=[], fails=["FAILCODE1_ALGORITHM_TO_TAKE_AS_ORIGIN_DOESNT_HAVE_COORDINATES"]
                    )
                    algorithms_results[algorithm["name"]] = [] # al diccionario de datos de resultados
                    algorithms_results_string += algorithm_results_string # al string de resultados
                    break

                origin = math_functions.sum_lists(inspection_point["coordinates"], algorithm_to_take_as_origin_coordinates)

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
            fails, location, algorithm_results, algorithm_status, resulting_images = execute_algorithm(inspection_image_filt, algorithm)

            inspection_point_images.append([algorithm["name"], algorithm["light"], resulting_images])

            # cambiar el status del punto de inspección si es necesario
            inspection_point_status = results_management.evaluate_status(algorithm_status, inspection_point_status)

            # agregar resultados del algoritmo a los resultados de los algoritmos
            algorithm_results_string = results_management.create_algorithm_results(
                algorithm["name"], algorithm["light"], algorithm_status, algorithm_results, fails
            )
            algorithms_results[algorithm["name"]] = algorithm_results # al diccionario de datos de resultados
            algorithms_results_string += algorithm_results_string # al string de resultados
            algorithms_locations[algorithm["name"]] = location # localización de los algoritmos

            # dejar de inspeccionar el punto de inspección si su estatus no es el necesario para continuar
            if algorithm_status != algorithm["necessary_status"]:
                break


        # cambiar el status de los puntos de inspección si es necesario
        inspection_points_status = results_management.evaluate_status(inspection_point_status, inspection_points_status)

        # agregar resultados del punto de inspección a los resultados de todos los puntos
        inspection_points_results += inspection_point_results

        inspection_points_results = results_management.create_inspection_point_results(
            algorithms_results_string, inspection_point["name"], inspection_point_status
        )
        inspection_points_results_string += inspection_points_results

        # exportar imágenes si: a) check:yes y el status del punto es malo, o
        # b) check:total
        if ((check_mode == "check:yes" and inspection_point_status == "bad") or
                check_mode == "check:total"):
            # agregar imágenes del punto de inspección imágenes de todos los puntos
            images_to_export.append([inspection_point["name"], inspection_point_images])

    return inspection_points_status, inspection_points_results, inspection_points_results_string, images_to_export

def execute_reference_algorithm(reference_algorithm, inspection_points_results, inspection_point_status):
    if reference_algorithm["function"] == "classification":
        # algoritmo de clasificación sólo retorna status de las referencia
        reference_algorithm_status = inspection_point_status
        reference_algorithm_results = [reference_algorithm_status]
    else:
        reference_algorithm_status, reference_algorithm_results = "failed", []
    return reference_algorithm_status, reference_algorithm_results

def inspect_reference(image, board, reference, check_mode, images_path , image_ultraviolet=None):
    inspection_points_status, inspection_points_results, inspection_points_results_string, images_to_export = \
        inspect_inspection_points(image, image_ultraviolet, reference["inspection_points"], check_mode)

    operations.export_reference_images(images_to_export, board.get_number(), reference["name"], images_path)

    # si no falló ningún punto de inspección, ejecutar el algoritmo de la referencia
    if inspection_points_status == "good":
        reference_algorithm_status, reference_algorithm_results = \
            execute_reference_algorithm(reference["reference_algorithm"], inspection_points_results, inspection_points_status)
        reference_status = reference_algorithm_status
    else:
        reference_algorithm_status, reference_algorithm_results = "", []
        reference_status = inspection_points_status

    # cambiar el status del tablero si es necesario
    board.evaluate_status(reference_status)

    reference_algorithm_results_string = "[{0}${1}]".format(reference_algorithm_status, reference_algorithm_results)

    reference_results_string = results_management.create_reference_results(
        inspection_points_results_string, reference["name"], reference_status, reference_algorithm_results_string
    )
    return reference_results_string

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
        reference_results_string = inspect_reference(image, board, reference, check_mode, images_path, image_ultraviolet)
        references_results_string += reference_results_string

    board.add_references_results(references_results_string)


def inspection_function_blob(inspection_point_image, inspection_point):
    """
    Las imágenes a exportar cuando se utiliza blob son:
    imagen filtrada, imagen binarizada.
    """
    location = "not_available"
    status = ""
    fails = []
    images_to_export = []

    images_to_export += [["filtered", inspection_point_image]]

    blob_area, biggest_blob, binary_image = cv_func.calculate_blob_area(
        inspection_point_image,
        inspection_point["parameters"]["lower_color"],
        inspection_point["parameters"]["upper_color"],
        inspection_point["parameters"]["color_scale"],
        inspection_point["parameters"]["min_blob_size"],
        inspection_point["parameters"]["max_blob_size"],
        inspection_point["parameters"]["invert_binary"],
    )

    images_to_export.append(["binary", binary_image])

    # Evaluar el punto de inspección
    blob_is_correct = evaluate_blob_results(
        blob_area, biggest_blob,
        inspection_point["parameters"]["min_area"],
        inspection_point["parameters"]["max_area"],
        inspection_point["parameters"]["max_allowed_blob_size"]
    )

    if blob_is_correct:
        status = "good"
    else:
        status = "bad"

    window_results = [blob_area, biggest_blob]
    return fails, location, window_results, status, images_to_export

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

def inspection_function_template_matching(inspection_point_image, inspection_point):
    """
    Las imágenes a exportar cuando se utiliza template matching son:
    imagen filtrada,
    imagen rgb con las coincidencias encontradas marcadas.
    """
    # Inspeccionar con template matching usando sub-templates
    location = "not_available"
    status = ""
    fails = []
    images_to_export = []

    images_to_export += [["filtered", inspection_point_image]]

    best_match = 0
    matches_number = 0
    correct_matches_number = False
    for sub_template,min_calification in zip(
            inspection_point["parameters"]["sub_templates"],
            inspection_point["parameters"]["min_califications"]
        ):

        sub_template_image, sub_template_index = sub_template

        if sub_template_image is None:
            fail = "TEMPLATE_DOESNT_EXIST-{0}".format(sub_template_index+1)
            fails.append(fail)
            continue

        # Dimensiones del template
        template_height = sub_template_image.shape[0]
        template_width = sub_template_image.shape[1]

        # Encontrar coincidencias
        try:
            matches_locations, best_match, color_converted_img = cv_func.find_matches(
                inspection_point_image, sub_template_image, min_calification,
                inspection_point["parameters"]["required_matches"],
                inspection_point["parameters"]["color_scale"],
                inspection_point["parameters"]["color_scale_for_binary"],
                inspection_point["parameters"]["color_range"],
                inspection_point["parameters"]["invert_binary"],
            )
            images_to_export += [["color_converted", color_converted_img]]
        except Exception as fail:
            status = "failed"
            fails.append(str(fail))
            continue


        # Evaluar el punto de inspección
        matches_number = len(matches_locations)
        if(matches_number == inspection_point["parameters"]["required_matches"]):
            correct_matches_number = True
            status = "good"
            break

    window_results = [matches_number, best_match]

    # Si no se encontraron las coincidencia necesarias con ninguna subtemplate
    if not status == "good" and status != "failed":
        status = "bad"

    # Si se encontró al menos una coincidencia, exportar imagen con las coincidencias marcadas
    if matches_number:

        # guardar localización de la coincidencia, si sólo se encontró y buscaba una
        if matches_number == 1 and inspection_point["parameters"]["required_matches"] == 1:
            location = matches_locations[0]

        matches_image = inspection_point_image.copy()
        # Dibujar rectángulos en las coincidencias
        for match_location in matches_locations:
            x = match_location[0]
            y = match_location[1]
            # Dibujar un rectángulos en la coincidencia
            rectangle(matches_image, (x, y),
                         (x+template_width, y+template_height),
                         (0, 255, 0), 2)

        images_to_export += [["matches", matches_image]]

    return fails, location, window_results, status, images_to_export
