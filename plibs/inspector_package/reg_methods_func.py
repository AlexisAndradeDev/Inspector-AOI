import cv2

import sys
# Hacer esto para importar módulos y paquetes externos
sys.path.append('C:/Dexill/Inspector/Alpha-Premium/x64/plibs/inspector_package/')
import math_functions, cv_func


class Fiducial:
    """
    Objeto con los datos de un fiducial.
    # ADVERTENCIA IMPORTANTE: No asignar atributo de coordenadas del centro
    # del fiducial al objeto Fiducial que se pasará a todos los hilos,
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

def find_fiducial(photo, fiducial):
    # lista de imágenes que se retornará para ser exportada
    images_to_export = []
    # Recortar el área en que se buscará el fiducial
    x1,y1,x2,y2 = fiducial.window
    searching_area_img = photo[y1:y2, x1:x2]

    # Aplicarle filtros secundarios (como blurs)
    searching_area_img_filtered = cv_func.apply_filters(searching_area_img, fiducial.filters)

    # Agregar imagen sin filtrar y filtrada del área de búsqueda a images_to_export
    images_to_export.append(["rgb{0}".format(fiducial.number), searching_area_img])
    images_to_export.append(["filtered{0}".format(fiducial.number), searching_area_img_filtered])

    # Encontrar contornos
    try:
        _,contours,_ = cv2.findContours(searching_area_img_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        return ("CONTOURS_NOT_FOUND_FID_{0}".format(fiducial.number)), None, None, images_to_export

    # Encontrar contorno que cumpla con los requisitos de circularidad y diámetro
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
        return ("APPROPRIATE_CIRCLE_FID_{0}".format(fiducial.number)), None, None, images_to_export

    (x, y), circle_radius = cv2.minEnclosingCircle(circle)
    circle_center = (round(x), round(y))
    circle_radius = round(circle_radius)
    images_to_export.append(["found{0}".format(fiducial.number), cv2.circle(
            cv2.circle(searching_area_img.copy(), circle_center, circle_radius, (0, 255, 0), 1),
            circle_center, 1, (0, 255, 0), 1)])

    return None, circle_center, circle_radius, images_to_export

def align_board_image(photo, fiducial_1, fiducial_2, objective_angle, objective_x, objective_y, return_rotation_and_translation=False):
    """Rotates and translates the image to align the windows in the correct place."""
    # lista de imágenes que se retornará para ser exportada
    images_to_export = []
    # dimensiones originales de la foto
    w_original = photo.shape[1]
    h_original = photo.shape[0]
    # Detectar centro del fiducial 1
    fail_code, circle_center, circle_radius, resulting_images = find_fiducial(photo, fiducial_1)
    images_to_export += resulting_images
    if fail_code:
        if return_rotation_and_translation:
            return fail_code, images_to_export, None, None, None
        else: return fail_code, images_to_export, None

    fiducial_1_center = (circle_center[0] + fiducial_1.window[0], circle_center[1] + fiducial_1.window[1])
    fiducial_1_radius = circle_radius

    # Detectar centro del fiducial 2
    fail_code, circle_center, circle_radius, resulting_images = find_fiducial(photo, fiducial_2)
    images_to_export += resulting_images
    if fail_code:
        if return_rotation_and_translation:
            return fail_code, images_to_export, None, None, None
        else: return fail_code, images_to_export, None

    fiducial_2_center = (circle_center[0] + fiducial_2.window[0], circle_center[1] + fiducial_2.window[1])
    fiducial_2_radius = circle_radius

    # Ángulo entre los 2 fiduciales
    angle = math_functions.calculate_angle(fiducial_1_center, fiducial_2_center)
    # Rotar la imagen para alinearla con las ventanas
    rotation = angle - objective_angle
    photo, trMat = cv_func.rotate(photo, rotation)

    # Multiplicar el centro del fiducial desaliando por la matriz de rotación usada para rotar la imagen, para encontrar el centro rotado
    fiducial_1_rotated_center = math_functions.multiply_matrices(fiducial_1_center, trMat)
    fiducial_1_rotated_center = (int(fiducial_1_rotated_center[0]), int(fiducial_1_rotated_center[1]))

    # Se traslada la diferencia entre coordenadas del fiducial 1 trazado en la creación de programas menos el fiducial 1 encontrado
    x_diference = objective_x - fiducial_1_rotated_center[0]
    y_diference = objective_y - fiducial_1_rotated_center[1]

    photo = cv_func.translate(photo, x_diference, y_diference)

    images_to_export += [["board_aligned", photo]]

    if return_rotation_and_translation:
        return fail_code, images_to_export, photo, rotation, [x_diference, y_diference]
    else: return fail_code, images_to_export, photo
