import cv2

from inspector_packageOptimizandoNuevo import (math_functions, cv_func, 
    threads_operations, images_operations)


def register_with_rotation_points_and_translation_point(image, rotation_iterations,
        rotation_point1, rotation_point2, translation_point, target_angle,
        target_x, target_y,
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
        images_to_return = {}

        # Encontrar punto de rotación 1
        rotation_point1_coordinates, resulting_images = cv_func.find_reference_point_in_board(image, rotation_point1)
        # agregar nombre del punto de rotación a las imágenes
        resulting_images = images_operations.add_to_images_name(resulting_images, "-rp1")
        images_to_return.update(resulting_images)

        if not rotation_point1_coordinates:
            fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point1["name"]) # !REGISTRATION_FAIL
            return fail, images_to_return, None, None, None

        # Encontrar punto de rotación 2
        rotation_point2_coordinates, resulting_images = cv_func.find_reference_point_in_board(image, rotation_point2)
        # agregar nombre del punto de rotación a las imágenes
        resulting_images = images_operations.add_to_images_name(resulting_images, "-rp2")
        images_to_return.update(resulting_images)

        if not rotation_point2_coordinates:
            fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point2["name"]) # !REGISTRATION_FAIL
            return fail, images_to_return, None, None, None


        # ángulo entre los 2 puntos de rotación
        angle_between_rotation_points = math_functions.calculate_angle(rotation_point1_coordinates, rotation_point2_coordinates)

        # Rotar la imagen
        rotation = angle_between_rotation_points - target_angle
        image, trMat = cv_func.rotate(image, rotation)

        total_rotation += rotation


    # Encontrar punto de traslación
    translation_point_coordinates, resulting_images = cv_func.find_reference_point_in_board(image, translation_point)
    # agregar nombre del punto de traslación a las imágenes
    resulting_images = images_operations.add_to_images_name(resulting_images, "-tp")
    images_to_return.update(resulting_images)

    if not translation_point_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(translation_point["name"]) # !REGISTRATION_FAIL
        return fail, images_to_return, None, None, None

    # Se traslada la diferencia entre las coordenadas donde deberá ubicarse el punto de traslación menos las coordenadas encontradas
    x_diference = target_x - translation_point_coordinates[0]
    y_diference = target_y - translation_point_coordinates[1]
    image = cv_func.translate(image, x_diference, y_diference)

    return None, images_to_return, image, total_rotation, [x_diference, y_diference]

def register_image(image, image_ultraviolet, registration_settings):
    image = image.copy() # no corromper la original

    if registration_settings["method"] == "rotation_points_and_translation_point":
        fail, images_to_return, image, rotation, translation = register_with_rotation_points_and_translation_point(
            image, registration_settings["rotation_iterations"],
            registration_settings["rotation_point1"], registration_settings["rotation_point2"],
            registration_settings["translation_point"], registration_settings["target_angle"],
            registration_settings["target_x"], registration_settings["target_y"]
        )

    if image_ultraviolet is not None and not fail:
        image_ultraviolet = image_ultraviolet.copy()
        image_ultraviolet = cv_func.apply_transformations(image_ultraviolet, rotation, translation)

    return fail, images_to_return, image, image_ultraviolet, rotation, translation


def calculate_missing_registration_data_rotation_points_and_translation_point(image, rotation_point1, rotation_point2, translation_point):
    fail = None
    images_to_return = {}


    # Encontrar punto de rotación 1
    rotation_point1_coordinates, resulting_images = cv_func.find_reference_point_in_board(image, rotation_point1)
    # agregar nombre del punto de rotación a las imágenes
    resulting_images = images_operations.add_to_images_name(resulting_images, "-rp1")
    images_to_return.update(resulting_images)

    if not rotation_point1_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point1["name"]) # !REGISTRATION_FAIL
        return fail, images_to_return, None

    # Encontrar punto de rotación 2
    rotation_point2_coordinates, resulting_images = cv_func.find_reference_point_in_board(image, rotation_point2)
    # agregar nombre del punto de rotación a las imágenes
    resulting_images = images_operations.add_to_images_name(resulting_images, "-rp2")
    images_to_return.update(resulting_images)

    if not rotation_point2_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point2["name"]) # !REGISTRATION_FAIL
        return fail, images_to_return, None


    # ángulo entre los 2 puntos de rotación
    angle_between_rotation_points = math_functions.calculate_angle(rotation_point1_coordinates, rotation_point2_coordinates)


    # Encontrar punto de traslación
    translation_point_coordinates, resulting_images = cv_func.find_reference_point_in_board(image, translation_point)
    # agregar nombre del punto de traslación a las imágenes
    resulting_images = images_operations.add_to_images_name(resulting_images, "-tp")
    images_to_return.update(resulting_images)

    if not translation_point_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(translation_point["name"]) # !REGISTRATION_FAIL
        return fail, images_to_return, None

    return None, images_to_return, [angle_between_rotation_points, rotation_point1_coordinates,
        rotation_point2_coordinates, translation_point_coordinates]

def calculate_missing_registration_data(image, method_settings):
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
        fail, images_to_return, missing_data = calculate_missing_registration_data_rotation_points_and_translation_point(
            image, method_settings["rotation_point1"],
            method_settings["rotation_point2"],
            method_settings["translation_point"],
        )

    return fail, images_to_return, missing_data
