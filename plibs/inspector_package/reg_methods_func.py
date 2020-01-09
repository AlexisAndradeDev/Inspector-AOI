import cv2

import sys
# Hacer esto para importar mÃ³dulos y paquetes externos
sys.path.append('C:/Dexill/Inspector/Alpha-Premium/x64/plibs/inspector_package/')
import math_functions, cv_func, operations


class Fiducial:
    """
    Objeto con los datos de un fiducial.
    # ADVERTENCIA IMPORTANTE: No asignar atributo de coordenadas del centro
    # del fiducial al objeto Fiducial que se pasarÃ¡ a todos los hilos,
    # ya que ocasiona que los hilos utilicen el mismo objeto Fiducial y asignen
    # valores incorrectos a las coordenadas.
    """
    def __init__(self, number, window, min_diameter, max_diameter, min_circle_perfection, max_circle_perfection, filters):
        self.number = number
        self.window = window
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.min_circle_perfection = min_circle_perfection
        self.max_circle_perfection = max_circle_perfection
        self.filters = filters


def find_circular_fiducial(photo, fiducial):
    # lista de imÃ¡genes que se retornarÃ¡
    images_to_return = []
    # Recortar el Ã¡rea en que se buscarÃ¡ el fiducial
    x1,y1,x2,y2 = fiducial.window
    searching_area_img = photo[y1:y2, x1:x2]

    # Aplicarle filtros secundarios (como blurs)
    searching_area_img_filtered = cv_func.apply_filters(searching_area_img, fiducial.filters)

    # Agregar imagen sin filtrar y filtrada del Ã¡rea de bÃºsqueda a images_to_return
    images_to_return.append(["rgb{0}".format(fiducial.number), searching_area_img])
    images_to_return.append(["filtered{0}".format(fiducial.number), searching_area_img_filtered])

    # Encontrar contornos
    try:
        _,contours,_ = cv2.findContours(searching_area_img_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        return ("CONTOURS_NOT_FOUND_FID_{0}".format(fiducial.number)), None, None, images_to_return

    # Encontrar contorno que cumpla con los requisitos de circularidad y diÃ¡metro
    circle = None
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv_func.get_contour_area(cnt)
        _,_,diameter,_ = cv2.boundingRect(cnt)

        if(diameter >= fiducial.min_diameter and diameter <= fiducial.max_diameter):
            if not perimeter:
                continue
            circularity = math_functions.calculate_circularity(area, perimeter)
            if fiducial.min_circle_perfection <= circularity <= fiducial.max_circle_perfection:
                circle = cnt
                break

    if circle is None:
        return ("APPROPRIATE_CIRCLE_FID_{0}".format(fiducial.number)), None, None, images_to_return

    (x, y), circle_radius = cv2.minEnclosingCircle(circle)
    circle_center = (round(x), round(y))
    circle_radius = round(circle_radius)
    images_to_return.append(["found{0}".format(fiducial.number), cv2.circle(
            cv2.circle(searching_area_img.copy(), circle_center, circle_radius, (0, 255, 0), 1),
            circle_center, 1, (0, 255, 0), 1)])

    return None, circle_center, circle_radius, images_to_return

def register_with_two_circular_fiducials(photo, fiducial_1, fiducial_2, objective_angle, objective_x, objective_y):
    """Rotates and translates the image to align the windows in the correct place."""
    # lista de imÃ¡genes que se retornarÃ¡ para ser exportada
    images_to_export = []
    # dimensiones originales de la foto
    w_original = photo.shape[1]
    h_original = photo.shape[0]
    # Detectar centro del fiducial 1
    fail_code, circle_center, circle_radius, resulting_images = find_circular_fiducial(photo, fiducial_1)
    images_to_export += resulting_images
    if fail_code:
        return fail_code, images_to_export, None, None, None

    fiducial_1_center = (circle_center[0] + fiducial_1.window[0], circle_center[1] + fiducial_1.window[1])
    fiducial_1_radius = circle_radius

    # Detectar centro del fiducial 2
    fail_code, circle_center, circle_radius, resulting_images = find_circular_fiducial(photo, fiducial_2)
    images_to_export += resulting_images
    if fail_code:
        return fail_code, images_to_export, None, None, None

    fiducial_2_center = (circle_center[0] + fiducial_2.window[0], circle_center[1] + fiducial_2.window[1])
    fiducial_2_radius = circle_radius

    # Ã¡ngulo entre los 2 fiduciales
    angle = math_functions.calculate_angle(fiducial_1_center, fiducial_2_center)
    # Rotar la imagen para alinearla con las ventanas
    rotation = angle - objective_angle
    photo, trMat = cv_func.rotate(photo, rotation)

    # Multiplicar el centro del fiducial desaliando por la matriz de rotaciÃ³n usada para rotar la imagen, para encontrar el centro rotado
    fiducial_1_rotated_center = math_functions.multiply_matrices(fiducial_1_center, trMat)
    fiducial_1_rotated_center = (int(fiducial_1_rotated_center[0]), int(fiducial_1_rotated_center[1]))

    # Se traslada la diferencia entre coordenadas del fiducial 1 trazado en la creaciÃ³n de programas menos el fiducial 1 encontrado
    x_diference = objective_x - fiducial_1_rotated_center[0]
    y_diference = objective_y - fiducial_1_rotated_center[1]

    photo = cv_func.translate(photo, x_diference, y_diference)

    images_to_export += [["board_aligned", photo]]

    return fail_code, images_to_export, photo, rotation, [x_diference, y_diference]

def register_with_rotation_points_and_translation_point(photo, rotation_iterations, rotation_point1, rotation_point2, translation_point, objective_angle, objective_x, objective_y):
    """
    Rota y traslada la imagen del tablero para alinearla con las ventanas de inspecciÃ³n.
    Retorna las imÃ¡genes de la Ãºltima iteraciÃ³n de rotaciÃ³n y la imagen del tablero alineado.
    Lo logra con el siguiente algoritmo:
        1. for (iteraciones deseadas):
            1. Localizar el punto de rotaciÃ³n 1 (se localiza encontrando el centroide de un contorno).
            2. Localizar el punto de rotaciÃ³n 2.
            3. Calcular el Ã¡ngulo entre los 2 puntos de rotaciÃ³n.
            4. Rotar al Ã¡ngulo objetivo con: Ã¡ngulo entre puntos de rotaciÃ³n - Ã¡ngulo objetivo
        2. Encontrar punto de traslaciÃ³n (como la esquina del tablero o el centroide de un triÃ¡ngulo).
        3. Trasladar el tablero con la diferencia entre las coordenadas objetivas y el punto de traslaciÃ³n.
    """

    # nÃºmero de grados que se ha rotado el tablero sumando todas las iteraciones
    total_rotation = 0
    for _ in range(rotation_iterations):
        # limpiar lista de imÃ¡genes que se retornarÃ¡ para ser exportada
        images_to_export = []

        # Encontrar punto de rotaciÃ³n 1
        rotation_point1_coordinates, resulting_images = cv_func.find_reference_point_in_photo(photo, rotation_point1)
        # agregar nombre del punto de rotaciÃ³n a las imÃ¡genes
        resulting_images = operations.add_to_images_name(resulting_images, "rp1")
        images_to_export += resulting_images

        if not rotation_point1_coordinates:
            fail_code = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point1["name"])
            return fail_code, images_to_export, None, None, None

        # Encontrar punto de rotaciÃ³n 2
        rotation_point2_coordinates, resulting_images = cv_func.find_reference_point_in_photo(photo, rotation_point2)
        # agregar nombre del punto de rotaciÃ³n a las imÃ¡genes
        resulting_images = operations.add_to_images_name(resulting_images, "rp2")
        images_to_export += resulting_images

        if not rotation_point2_coordinates:
            fail_code = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(rotation_point2["name"])
            return fail_code, images_to_export, None, None, None


        # Ã¡ngulo entre los 2 puntos de rotaciÃ³n
        angle_between_rotation_points = math_functions.calculate_angle(rotation_point1_coordinates, rotation_point2_coordinates)

        # Rotar la imagen
        rotation = angle_between_rotation_points - objective_angle
        photo, trMat = cv_func.rotate(photo, rotation)

        total_rotation += rotation


    # Encontrar punto de traslaciÃ³n
    translation_point_coordinates, resulting_images = cv_func.find_reference_point_in_photo(photo, translation_point)
    # agregar nombre del punto de traslaciÃ³n a las imÃ¡genes
    resulting_images = operations.add_to_images_name(resulting_images, "tp")
    images_to_export += resulting_images

    if not translation_point_coordinates:
        fail_code = "APPROPRIATE_CONTOUR_NOT_FOUND_{0}".format(translation_point["name"])
        return fail_code, images_to_export, None, None, None

    # Se traslada la diferencia entre las coordenadas donde deberÃ¡ ubicarse el fiducial 1 menos las coordenadas encontradas
    x_diference = objective_x - translation_point_coordinates[0]
    y_diference = objective_y - translation_point_coordinates[1]
    photo = cv_func.translate(photo, x_diference, y_diference)

    images_to_export += [["board_aligned", photo]]

    return None, images_to_export, photo, total_rotation, [x_diference, y_diference]

def align_board_image(board_image, registration_settings):
    if registration_settings["method"] == "circular_fiducials":
        fail, images_to_export, aligned_board_image, rotation, translation = register_with_two_circular_fiducials(
            board_image, registration_settings["fiducial_1"], registration_settings["fiducial_2"],
            registration_settings["objective_angle"], registration_settings["objective_x"],
            registration_settings["objective_y"]
        )
    elif registration_settings["method"] == "rotation_points_and_translation_point":
        fail, images_to_export, aligned_board_image, rotation, translation = register_with_rotation_points_and_translation_point(
            board_image, registration_settings["rotation_iterations"],
            registration_settings["rotation_point1"], registration_settings["rotation_point2"],
            registration_settings["translation_point"], registration_settings["objective_angle"],
            registration_settings["objective_x"], registration_settings["objective_y"]
        )

    return fail, images_to_export, aligned_board_image, rotation, translation
