import cv2

from inspector_package import math_functions, cv_func, operations


def register_with_rotation_points_and_translation_point(board_image, rotation_iterations,
        rotation_point1, rotation_point2, translation_point, objective_angle,
        objective_x, objective_y,
    ):
    """
    Rota y traslada la imagen del tablero para alinearla con las ventanas de inspección.
    Retorna las imágenes de la última iteración de rotación y la imagen del tablero alineado.
    Lo logra con el siguiente algoritmo:
        1. for (iteraciones deseadas):
            1. Localizar el punto de rotación 1 (se localiza encontrando el centroide de un contorno).
            2. Localizar el punto de rotación 2.
            3. Calcular el ángulo entre los 2 puntos de rotación.
            4. Rotar al ángulo objetivo con: ángulo entre puntos de rotación - ángulo objetivo
        2. Encontrar punto de traslación (como la esquina del tablero o el centroide de un triángulo).
        3. Trasladar el tablero con la diferencia entre las coordenadas objetivas y el punto de traslación.
    """

    # número de grados que se ha rotado el tablero sumando todas las iteraciones
    total_rotation = 0
    for _ in range(rotation_iterations):
        # limpiar lista de imágenes que se retornará para ser exportada
        images_to_export = []

        # Encontrar punto de rotación 1
        rotation_point1_coordinates, resulting_images = cv_func.find_reference_point_in_board(board_image, rotation_point1)
        # agregar nombre del punto de rotación a las imágenes
        resulting_images = operations.add_to_images_name(resulting_images, "-rp1")
        images_to_export += resulting_images

        if not rotation_point1_coordinates:
            fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point1["name"]) # !REGISTRATION_FAIL
            return fail, images_to_export, None, None, None

        # Encontrar punto de rotación 2
        rotation_point2_coordinates, resulting_images = cv_func.find_reference_point_in_board(board_image, rotation_point2)
        # agregar nombre del punto de rotación a las imágenes
        resulting_images = operations.add_to_images_name(resulting_images, "-rp2")
        images_to_export += resulting_images

        if not rotation_point2_coordinates:
            fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point2["name"]) # !REGISTRATION_FAIL
            return fail, images_to_export, None, None, None


        # ángulo entre los 2 puntos de rotación
        angle_between_rotation_points = math_functions.calculate_angle(rotation_point1_coordinates, rotation_point2_coordinates)

        # Rotar la imagen
        rotation = angle_between_rotation_points - objective_angle
        board_image, trMat = cv_func.rotate(board_image, rotation)

        total_rotation += rotation


    # Encontrar punto de traslación
    translation_point_coordinates, resulting_images = cv_func.find_reference_point_in_board(board_image, translation_point)
    # agregar nombre del punto de traslación a las imágenes
    resulting_images = operations.add_to_images_name(resulting_images, "-tp")
    images_to_export += resulting_images

    if not translation_point_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(translation_point["name"]) # !REGISTRATION_FAIL
        return fail, images_to_export, None, None, None

    # Se traslada la diferencia entre las coordenadas donde deberá ubicarse el fiducial 1 menos las coordenadas encontradas
    x_diference = objective_x - translation_point_coordinates[0]
    y_diference = objective_y - translation_point_coordinates[1]
    board_image = cv_func.translate(board_image, x_diference, y_diference)

    return None, images_to_export, board_image, total_rotation, [x_diference, y_diference]

def align_board_image(board_image, registration_settings):
    if registration_settings["method"] == "rotation_points_and_translation_point":
        fail, images_to_export, aligned_board_image, rotation, translation = register_with_rotation_points_and_translation_point(
            board_image, registration_settings["rotation_iterations"],
            registration_settings["rotation_point1"], registration_settings["rotation_point2"],
            registration_settings["translation_point"], registration_settings["objective_angle"],
            registration_settings["objective_x"], registration_settings["objective_y"]
        )

    return fail, images_to_export, aligned_board_image, rotation, translation


def calculate_missing_registration_data_rotation_points_and_translation_point(board_image, rotation_point1, rotation_point2, translation_point):
    fail = None
    images_to_export = []


    # Encontrar punto de rotación 1
    rotation_point1_coordinates, resulting_images = cv_func.find_reference_point_in_board(board_image, rotation_point1)
    # agregar nombre del punto de rotación a las imágenes
    resulting_images = operations.add_to_images_name(resulting_images, "-rp1")
    images_to_export += resulting_images

    if not rotation_point1_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point1["name"]) # !REGISTRATION_FAIL
        return fail, images_to_export, None

    # Encontrar punto de rotación 2
    rotation_point2_coordinates, resulting_images = cv_func.find_reference_point_in_board(board_image, rotation_point2)
    # agregar nombre del punto de rotación a las imágenes
    resulting_images = operations.add_to_images_name(resulting_images, "-rp2")
    images_to_export += resulting_images

    if not rotation_point2_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point2["name"]) # !REGISTRATION_FAIL
        return fail, images_to_export, None


    # ángulo entre los 2 puntos de rotación
    angle_between_rotation_points = math_functions.calculate_angle(rotation_point1_coordinates, rotation_point2_coordinates)


    # Encontrar punto de traslación
    translation_point_coordinates, resulting_images = cv_func.find_reference_point_in_board(board_image, translation_point)
    # agregar nombre del punto de traslación a las imágenes
    resulting_images = operations.add_to_images_name(resulting_images, "-tp")
    images_to_export += resulting_images

    if not translation_point_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(translation_point["name"]) # !REGISTRATION_FAIL
        return fail, images_to_export, None

    return None, images_to_export, [angle_between_rotation_points, rotation_point1_coordinates,
        rotation_point2_coordinates, translation_point_coordinates]

def calculate_missing_registration_data(board_image, method_settings):
    """Obtener los datos necesarios para crear todos los datos del registro
    (ángulo entre puntos de rotación, centros de puntos de referencia),
    apartir de algunos parámetros del método de registro que se use.

    Por ejemplo, se puede utilizar para encontrar datos del método de registro
    «puntos de rotación y punto de traslación»:
        - Ángulo entre los puntos de rotación.
        - Coordenadas de los centros de los puntos de rotación. (que sirven para
          traslación de la imagen del tablero en el registro).

    a partir de los parámetros utilizados para encontrar los puntos de rotación
    y traslación:
        - Tipo de punto de referencia (corner/centroid).
        - Región de búsqueda de cada punto de referencia.
        - Datos para binarizado del contorno.
        - Filtros secundarios para cada punto de referencia.
        - Filtros de contornos para cada punto de referencia.
        ...
    """

    if method_settings["method"] == "rotation_points_and_translation_point":
        fail, images_to_export, missing_data = calculate_missing_registration_data_rotation_points_and_translation_point(
            board_image, method_settings["rotation_point1"],
            method_settings["rotation_point2"],
            method_settings["translation_point"],
        )

    return fail, images_to_export, missing_data
