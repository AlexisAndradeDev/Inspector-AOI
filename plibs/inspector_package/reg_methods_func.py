import cv2

from inspector_package import math_functions, cv_func, operations


def register_with_rotation_points_and_translation_point(photo, rotation_iterations, rotation_point1, rotation_point2, translation_point, objective_angle, objective_x, objective_y):
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
        rotation_point1_coordinates, resulting_images = cv_func.find_reference_point_in_photo(photo, rotation_point1)
        # agregar nombre del punto de rotación a las imágenes
        resulting_images = operations.add_to_images_name(resulting_images, "-rp1")
        images_to_export += resulting_images

        if not rotation_point1_coordinates:
            fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point1["name"]) # !REGISTRATION_FAIL
            return fail, images_to_export, None, None, None

        # Encontrar punto de rotación 2
        rotation_point2_coordinates, resulting_images = cv_func.find_reference_point_in_photo(photo, rotation_point2)
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
        photo, trMat = cv_func.rotate(photo, rotation)

        total_rotation += rotation


    # Encontrar punto de traslación
    translation_point_coordinates, resulting_images = cv_func.find_reference_point_in_photo(photo, translation_point)
    # agregar nombre del punto de traslación a las imágenes
    resulting_images = operations.add_to_images_name(resulting_images, "-tp")
    images_to_export += resulting_images

    if not translation_point_coordinates:
        fail = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(translation_point["name"]) # !REGISTRATION_FAIL
        return fail, images_to_export, None, None, None

    # Se traslada la diferencia entre las coordenadas donde deberá ubicarse el fiducial 1 menos las coordenadas encontradas
    x_diference = objective_x - translation_point_coordinates[0]
    y_diference = objective_y - translation_point_coordinates[1]
    photo = cv_func.translate(photo, x_diference, y_diference)

    images_to_export += [["board_aligned", photo]]

    return None, images_to_export, photo, total_rotation, [x_diference, y_diference]

def align_board_image(board_image, registration_settings):
    if registration_settings["method"] == "rotation_points_and_translation_point":
        fail, images_to_export, aligned_board_image, rotation, translation = register_with_rotation_points_and_translation_point(
            board_image, registration_settings["rotation_iterations"],
            registration_settings["rotation_point1"], registration_settings["rotation_point2"],
            registration_settings["translation_point"], registration_settings["objective_angle"],
            registration_settings["objective_x"], registration_settings["objective_y"]
        )

    return fail, images_to_export, aligned_board_image, rotation, translation
