import sys
# Hacer esto para importar módulos y paquetes externos
sys.path.append('C:/Dexill/Inspector/Alpha-Premium/x64/plibs/inspector_package/')
import math_functions, cv_func, operations

from cv2 import imread, imwrite, rectangle
from numpy import array

TEMPLATE_MATCHING = "m"
BLOB = "b"
TRANSITION = "t"


def create_inspection_point(inspection_point_data):
    inspection_point = {
        "light":inspection_point_data[0], # white / ultraviolet
        "name":inspection_point_data[1],
        "type":"single", # no utiliza cadenas esta versión
        "coordinates":inspection_point_data[2],
        "inspection_function":inspection_point_data[3],
        "filters":inspection_point_data[5],
    }

    # parámetros de la función de inspección del punto (áreas de blob, templates de template matching, etc.)
    parameters_data = inspection_point_data[4]
    inspection_point["parameters"] = get_inspection_function_parameters(inspection_point, parameters_data)

    return inspection_point

def create_inspection_points(data):
    inspection_points = []
    for inspection_point_data in data:
        inspection_point = create_inspection_point(inspection_point_data)
        inspection_points.append(inspection_point)
    return inspection_points

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
    # Lista de subtemplates y la calificación mínima de cada una
    for i in range(number_of_sub_templates):
        sub_template_img = imread(template_path + "-" + str(i+1) + ".bmp")
        sub_template_img = cv_func.apply_filters(sub_template_img, filters)
        sub_templates.append([sub_template_img, i])
    return sub_templates


def inspect_point(inspection_image_filt, inspection_point):
    if (inspection_point["inspection_function"] == BLOB):
        fails, window_results, window_status, resulting_images = \
            inspection_function_blob(inspection_image_filt, inspection_point)

    elif (inspection_point["inspection_function"] == TEMPLATE_MATCHING):
        fails, window_results, window_status, resulting_images = \
            inspection_function_template_matching(inspection_image_filt, inspection_point)

    else:
        return ["INAVLID_INSPECTION_FUNCTION"], None, None, None

    return fails, window_results, window_status, resulting_images

def inspect_inspection_points(first_inspection_point, last_inspection_point,
        aligned_board_image, board, inspection_points, stage, check_mode,
        aligned_board_image_ultraviolet=None,
    ):

    if stage == "debug":
        images_path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/"
    elif stage == "inspection":
        images_path = "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/bad_windows_results/"

    # se le resta 1 a la posición de los puntos de inspección para obtener su índice en la lista
    first_inspection_point -= 1
    last_inspection_point -= 1
    # la función range toma desde first hasta last-1, así que hay que sumarle 1
    inspection_points = inspection_points[first_inspection_point:last_inspection_point+1]

    for inspection_point in inspection_points:
        if inspection_point["light"] == "ultraviolet":
            inspection_image = cv_func.crop_image(aligned_board_image_ultraviolet,inspection_point["coordinates"])
        else:
            inspection_image = cv_func.crop_image(aligned_board_image,inspection_point["coordinates"])

        inspection_image_filt = cv_func.apply_filters(
            inspection_image,
            inspection_point["filters"]
            )

        fails, window_results, window_status, resulting_images = inspect_point(inspection_image_filt, inspection_point)

        # Escribir imágenes sin filtrar de puntos de inspección malos si se activó el modo de revisión bajo
        if(window_status == "bad" and check_mode == "check:low"):
            imwrite(
                "{0}{1}-{2}-{3}-rgb.bmp".format(images_path, board.get_number(), inspection_point["name"], inspection_point["light"]),
                inspection_image
            )

        # Escribir imágenes filtradas de puntos de inspección malos si se activó el modo de revisión avanzado
        elif(window_status == "bad" and check_mode == "check:advanced" and resulting_images is not None):
            imwrite(
                "{0}{1}-{2}-{3}-rgb.bmp".format(images_path, board.get_number(), inspection_point["name"], inspection_point["light"]),
                inspection_image
            )
            # Exportar imágenes
            operations.export_images(resulting_images, board.get_number(), inspection_point["name"], inspection_point["light"], images_path)

        # Escribir todas las imágenes de todos los puntos de inspección buenos y malos con el modo de revisión total (solo usado en debugeo)
        elif (check_mode == "check:total" and resulting_images is not None):
            imwrite(
                "{0}{1}-{2}-{3}-rgb.bmp".format(images_path, board.get_number(), inspection_point["name"], inspection_point["light"]),
                inspection_image
            )
            # Exportar imágenes
            operations.export_images(resulting_images, board.get_number(), inspection_point["name"], inspection_point["light"], images_path)

        # El estado del tablero es malo si hubo un defecto y no hubo fallos.
        # El estado del tablero es fallido si hubo un fallo al inspeccionar
        # y no se puede cambiar del estado fallido a otro.
        if (window_status == "bad" and board.get_status() != "failed"):
            board.set_status("bad")
        if (window_status == "failed"):
            board.set_status("failed")

        # Agregar resultados al string que se utilizará para escribirlos en el archivo ins_results.io
        board.add_inspection_point_results(inspection_point["name"], inspection_point["light"], window_status, window_results, fails)


def inspection_function_blob(inspection_point_image, inspection_point):
    """
    Las imágenes a exportar cuando se utiliza blob son:
    imagen filtrada, imagen binarizada.
    """
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
    return fails, window_results, status, images_to_export

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
    # Si hay área mínima y máxima
    # Si hay área mínima
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

    # Si no se encontraron las coincidencia necesarias con ninguna subtemplate
    if not status == "good" and status != "failed":
        status = "bad"

    # Si se encontró al menos una coincidencia, exportar imagen con las coincidencias marcadas
    if matches_number:
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
        # fails, window_results, window_status, resulting_images
        return fails, [matches_number, best_match], status, images_to_export
    # Si no se encontraron coincidencias, exportar sólo imagen filtrada
    else:
        # fails, window_results, window_status, resulting_images
        return fails, [matches_number, best_match], status, images_to_export
