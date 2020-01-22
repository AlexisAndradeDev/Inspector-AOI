import cv2
import math
import numpy as np

from inspector_package import math_functions, excepts
from timeit import default_timer as timer

def draw_found_circle(img, x, y, c1_size=1, c1_color=(0,255,255), c1_thickness=1, c2_size=3, c2_color=(0,0,255), c2_thickness=1):
    found = img.copy()
    cv2.circle(found, (x, y), c1_size, c1_color, c1_thickness)
    cv2.circle(found, (x, y), c2_size, c2_color, c2_thickness)
    return found

def crop_image(image, coordinates):
    [x1,y1,x2,y2] = coordinates
    return image[y1:y2,x1:x2].copy()

def open_camera(camera_number, camera_dim_width, camera_dim_height, captures_to_adapt):
    camera = cv2.VideoCapture(camera_number)
    if not camera.isOpened():
        return "CAM_1_NOT_OPENED", None

    camera.set(3, camera_dim_width)
    camera.set(4, camera_dim_height)
    # Adaptar cámara a la luz
    for _ in range(captures_to_adapt):
        _, frame = camera.read()

    return None, camera

def showImage(name, img):
    """Shows and image with a name."""
    cv2.imshow(name, img)
    cv2.moveWindow(name, 50, 100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(img, scale):
    """ Resizes the # # input image with the specified scale. """
    scaled = cv2.resize(img, (0, 0), None, scale, scale)

    return scaled

def rotate(image, angleInDegrees):
    # Función para rotar una imagen cualquiera
    h, w = image.shape[:2]
    # image center
    img_c = (w/2, h/2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    # new x and y
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_NEAREST)
    return outImg, rot

def translate(img, x, y):
    rows = img.shape[0]
    cols = img.shape[1]

    M = np.float32([[1, 0, x],[0, 1, y]])
    dst = cv2.warpAffine(img,M,(cols,rows), flags=cv2.INTER_NEAREST)

    return dst

def apply_filters(img, filters):
    if not filters:
        return img
    for filter in filters:
        if filter[0] == "GaussianBlur":
            filter_area_size = filter[1]
            img = cv2.GaussianBlur(img, (filter_area_size, filter_area_size), 0)
        elif filter[0] == "blur":
            filter_area_size = filter[1]
            img = cv2.blur(img, (filter_area_size, filter_area_size))
        elif filter[0] == "medianBlur":
            filter_area_size = filter[1]
            img = cv2.medianBlur(img, filter_area_size)
        elif filter[0] == "bilateral":
            filter_area_size = filter[1]
            bilateralFilter_param2 = filter[2]
            img = cv2.bilateralFilter(img, filter_area_size, bilateralFilter_param2, bilateralFilter_param2)
        elif filter[0] == "binary":
            # Convertir imagen de fiducial a HSV
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            # Aplicar threshold para encontrar los colores para encontrar el fiducial
            threshold_range = filter[1]
            [lower, upper] = threshold_range
            img = cv2.inRange(hsv, np.array(lower), np.array(upper))
        elif filter[0] == "reverseBinary":
            # Convertir imagen de fiducial a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Aplicar threshold para encontrar los colores para encontrar el fiducial
            threshold_range = filter[1]
            [lower, upper] = threshold_range
            img = cv2.inRange(hsv, np.array(lower), np.array(upper))
            img = cv2.bitwise_not(img)
        elif filter[0] == "bitwise":
            # Convertir imagen de fiducial a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Aplicar threshold para encontrar los colores para encontrar el fiducial
            threshold_range = filter[1]
            [lower, upper] = threshold_range
            thresholded = cv2.inRange(hsv, np.array(lower), np.array(upper))
            img = cv2.bitwise_and(img, img, mask = thresholded)
        elif filter[0] == "reverseBitwise":
            # Convertir imagen de fiducial a HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Aplicar threshold para encontrar los colores para encontrar el fiducial
            threshold_range = filter[1]
            [lower, upper] = threshold_range
            thresholded = cv2.inRange(hsv, np.array(lower), np.array(upper))
            img = cv2.bitwise_and(img, img, mask = 255 - thresholded)
        elif filter[0] == "canny":
            img = cv2.Canny(img, filter[1], filter[2])
        elif filter[0] == "gray":
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return img

# analysis functions
def get_vertices_from_cnt(cnt):
    peri = cv2.arcLength(cnt, True)
    approxPoly = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    vertices = len(approxPoly)
    return vertices, approxPoly

def detect_shape(cnt):
    """Retorna el nombre del polígono según su número de vértices."""
    # Inicializar el nombre de la figura y aproximar el contorno (número de vértices)
    shape = "unidentified"

    vertices, approxPoly = get_vertices_from_cnt(cnt)

    if vertices == 3:
	       shape = "triangle"
    elif vertices == 4:
        # Computar la relación de aspecto (relación entre ancho y alto)
        (x, y, w, h) = cv2.boundingRect(approxPoly)
        ar = w / float(h)

        # un cuadrado tendrá una relación de aspecto aproximada a uno,
        # de otra forma será un cuadrado
        if ar >= 0.8 and ar <= 1.2:
            shape = "square"
        else:
            shape = "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"

    return shape

def sort_contours(cnts):
    """ Sorts contours max to min. """
    cnts = sorted(cnts, key = get_contour_area, reverse = True)

    return cnts

def get_contour_area(contour):
    x, y, w, h = cv2.boundingRect(contour)
    width, height = x+w, y+h

    # Crear imagen negra
    blank_image = np.zeros((height, width, 1), np.uint8)

    # Dibujar contorno con color blanco
    cv2.drawContours(blank_image, [contour], 0, (255), cv2.FILLED)
    # Contar los pixeles que no tengan un valor de 0 (contar pixeles blancos)
    none_zero_pixels = cv2.countNonZero(blank_image)

    return none_zero_pixels


CONTOURS_FILTERS = ["min_area", "max_area", "polygon", "vertices", "circularity", "min_diameter", "max_diameter"]
def get_valid_contours_filters(filters):
    valid_filters = {} # dict
    invalid_filters = [] # filtros no existentes

    for filter, parameters in filters.items():
        if filter in CONTOURS_FILTERS:
            valid_filters[filter] = parameters
        else:
            invalid_filters.append(filter)
    return valid_filters, invalid_filters

def contour_fulfills_filters(cnt, contours_filters):
    """
    Retorna verdadero si el contorno cumple con todos los filtros; retorna
    falso si no cumple con uno o más.

    Puede filtrarse según los siguientes parámetros (contours_filters):
    1. Área mínima del contorno.
    2. Área máxima del contorno
    3. Polígono: selecciona los contornos cuya forma sea parecida al polígono señalado o cuyo número de vértices coincida con el requerido.
    4. Círculo: circularidad mín del contorno.
    5. Círculo: mín diámetro.
    6. Círculo: máx diámetro.
    """
    # si no pasa alguno de los filtros, tomará valor False y continuará con el siguiente contorno
    cnt_is_valid = True

    for filter, parameters in contours_filters.items():
        if filter not in CONTOURS_FILTERS:
            continue

        if filter == "min_area":
            min_area = parameters["min_area"]
            if not get_contour_area(cnt) >= min_area:
                cnt_is_valid = False
                break
        elif filter == "max_area":
            max_area = parameters["max_area"]
            if not get_contour_area(cnt) <= max_area:
                cnt_is_valid = False
                break
        elif filter == "polygon":
            required_polygon = parameters["required_polygon"]
            polygon = detect_shape(cnt)
            if polygon != required_polygon:
                cnt_is_valid = False
                break
        elif filter == "vertices":
            required_vertices = parameters["required_vertices"]
            vertices, approxPoly = get_vertices_from_cnt(cnt)
            if vertices != required_vertices:
                cnt_is_valid = False
                break
        elif filter == "circularity":
            min_circle_perfection = parameters["min_circularity"]
            max_circle_perfection = 1.2

            perimeter = cv2.arcLength(cnt, True)
            area = get_contour_area(cnt)

            if not perimeter:
                cnt_is_valid = False
                break

            circularity = math_functions.calculate_circularity(area, perimeter)

            if not min_circle_perfection <= circularity <= max_circle_perfection:
                cnt_is_valid = False
                break
        elif filter == "min_diameter":
            min_diameter = parameters["min_diameter"]
            _,_,diameter,_ = cv2.boundingRect(cnt)
            if not diameter >= min_diameter:
                cnt_is_valid = False
                break
        elif filter == "max_diameter":
            max_diameter = parameters["max_diameter"]
            _,_,diameter,_ = cv2.boundingRect(cnt)
            if not diameter <= max_diameter:
                cnt_is_valid = False
                break

    return cnt_is_valid

def find_filtered_contour(img, color_scale, lower_color, upper_color, invert_binary, contours_filters):
    """
    Retorna el primer contorno que cumpla con los filtros de contornos introducidos
    por el usuario.
    """

    contours, binary = find_contours(img,
        lower_color, upper_color, color_scale, invert_binary)

    if contours is None:
        return None, binary

    # eliminar de la lista los filtros que no existan
    contours_filters, _ = get_valid_contours_filters(contours_filters)

    for cnt in contours:
        cnt_is_valid = contour_fulfills_filters(cnt, contours_filters)
        if cnt_is_valid:
            return cnt, binary

    return None, binary

def find_filtered_contours(img, color_scale, lower_color, upper_color, invert_binary, contours_filters):
    """
    Retorna todos los contornos que cumplan con los filtros de contornos introducidos
    por el usuario.
    """

    contours, binary = find_contours(img,
        lower_color, upper_color, color_scale, invert_binary)

    if contours is None:
        return [], binary

    # eliminar de la lista los filtros que no existan
    contours_filters, _ = get_valid_contours_filters(contours_filters)

    # filtrar los contornos y agregarlos a la lista si los cumplen todos
    valid_contours = []
    for cnt in contours:
        cnt_is_valid = contour_fulfills_filters(cnt, contours_filters)
        if cnt_is_valid:
            valid_contours.append(cnt)

    return valid_contours, binary

def find_contours(img, lower, upper, color_scale, invert_binary=False):
    """Retorna todos los contornos, sin filtros."""
    # retorna los contornos encontrados y la imagen binarizada
    if color_scale == "hsv":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        binary_image = cv2.inRange(hsv, lower, upper)
    elif color_scale == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.inRange(gray, lower, upper)

    if invert_binary:
        # invertir binarizado
        binary_image = cv2.bitwise_not(binary_image)


    # Encontrar contornos
    try:
        _, contours, _ = cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    except:
         return None, None

    return contours, binary_image


def create_corner_parameters(name, coordinates, lower_color, upper_color,
        color_scale, invert_binary=False, filters=[], contours_filters={}
    ):
    corner_parameters = {
        "name":name,
        "coordinates":coordinates,
        "lower_color":lower_color,
        "upper_color":upper_color,
        "color_scale":color_scale,
        "invert_binary":invert_binary,
        "filters":filters,
        "contours_filters":contours_filters,
    }
    return corner_parameters

def find_corner(img, color_scale, lower_color, upper_color, invert_binary, contours_filters):
    images_to_return = [] # imágenes que se retornarán

    corner_contour, binary = find_filtered_contour(img, color_scale, lower_color, upper_color, invert_binary, contours_filters)
    images_to_return.append(["binary",binary])

    if corner_contour is None:
        return None, images_to_return

    # calcular x,y de la esquina superior izquierda del contorno
    x,y,_,_ = cv2.boundingRect(corner_contour)

    # imagen de la esquina encontrada
    found = draw_found_circle(img, x, y)
    images_to_return.append(["found", found])

    return [x,y], images_to_return

def create_centroid_parameters(name, coordinates, lower_color, upper_color,
        color_scale, invert_binary=False, filters=[], contours_filters={}
    ):
    centroid_parameters = {
        "name":name,
        "coordinates":coordinates,
        "lower_color":lower_color,
        "upper_color":upper_color,
        "color_scale":color_scale,
        "invert_binary":invert_binary,
        "filters":filters,
        "contours_filters":contours_filters,
    }
    return centroid_parameters

def find_centroid(img, color_scale, lower_color, upper_color, invert_binary, contours_filters):
    images_to_return = [] # imágenes que se retornarán

    centroid_contour, binary = find_filtered_contour(img, color_scale, lower_color, upper_color, invert_binary, contours_filters)
    images_to_return.append(["binary", binary])

    if centroid_contour is None:
        return None, images_to_return

    # Obtener datos sobre el blob
    M = cv2.moments(centroid_contour)
    # Obtener x,y del centroide del contorno
    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    else:
        return None, images_to_return

    # imagen del centroide encontrado
    found = draw_found_circle(img, x, y)
    images_to_return.append(["found", found])

    return [x,y], images_to_return

def create_reference_point(rp_type, name, coordinates, color_scale, lower_color,
        upper_color, invert_binary=False, filters=[], contours_filters={}):
    """
    Crea un diccionario con los parámetros para encontrar un punto de referencia.
    Un punto de referencia es aquel cuyas coordenadas son utilizadas para orientarse
    en el tablero, usándolas para rotarlo o trasladarlo para el registro de la imagen.
    Los puntos de referencia son centroides o esquinas de contornos.
    """
    contours_filters, invalid_filters = get_valid_contours_filters(contours_filters)

    if color_scale == "hsv":
        # convertir a array de numpy
        if type(lower_color) is not np.ndarray:
            lower_color = np.array(lower_color)
        if type(upper_color) is not np.ndarray:
            upper_color = np.array(upper_color)

    if rp_type == "corner":
        # la coordenada será la esquina de un contorno
        reference_point = create_corner_parameters(
            name, coordinates, lower_color, upper_color, color_scale,
            invert_binary, filters, contours_filters
        )

    elif rp_type == "centroid":
        # la coordenada será el centroide de un contorno
        reference_point = create_centroid_parameters(
            name, coordinates, lower_color, upper_color, color_scale,
            invert_binary, filters, contours_filters
        )

    reference_point["type"] = rp_type
    return reference_point

def find_reference_point(img_, reference_point):
    img = img_.copy() # no corromper la imagen original
    images_to_return = []

    images_to_return.append(["rgb", img])

    # Aplicar filtros secundarios a la imagen    
    img = apply_filters(img, reference_point["filters"])

    if reference_point["type"] == "centroid":
        coordinates, resulting_images = find_centroid(img, reference_point["color_scale"],
            reference_point["lower_color"], reference_point["upper_color"],
            reference_point["invert_binary"], reference_point["contours_filters"])

    elif reference_point["type"] == "corner":
        coordinates, resulting_images = find_corner(img, reference_point["color_scale"],
            reference_point["lower_color"], reference_point["upper_color"],
            reference_point["invert_binary"], reference_point["contours_filters"])

    images_to_return += resulting_images

    return coordinates, images_to_return

def find_reference_point_in_photo(img, reference_point):
    """Suma las coordenadas del punto de referencia dentro de la ventana,
    más las coordenadas de la esquina superior izquierda de la ventana."""
    [x1,y1,x2,y2] = reference_point["coordinates"]
    rp_img = img[y1:y2, x1:x2].copy()

    coordinates, images_to_export = find_reference_point(rp_img, reference_point)
    if not coordinates:
        return None, images_to_export

    # Coordenadas reales en el tablero
    [x,y] = coordinates
    coordinates = (x+x1,y+y1)

    return coordinates, images_to_export


def choose_blobs_by_min_max_area(img, contours, min_blob_size, max_blob_size):
    total_area = 0
    for cnt in contours:
        # Calcular el área del contorno encontrado
        area = get_contour_area(cnt)

        if(area >= min_blob_size and area <= max_blob_size):
            # Agregar al área total de blob
            total_area += area

    return total_area

def choose_blobs_by_max_area(img, contours, max_blob_size):
    total_area = 0
    for cnt in contours:
        # Calcular el área del contorno encontrado
        area = get_contour_area(cnt)

        if(area <= max_blob_size):
            # Agregar al área total de blob
            total_area += area

    return total_area

def choose_blobs_by_min_area(img, contours, min_blob_size):
    total_area = 0
    for cnt in contours:
        # Calcular el área del contorno encontrado
        area = get_contour_area(cnt)

        if(area >= min_blob_size):
            # Agregar al área total de blob
            total_area += area

    return total_area

def choose_all_blobs(img, contours):
    total_area = 0
    for cnt in contours:
        # Calcular el área del contorno encontrado
        area = get_contour_area(cnt)
        # Agregar al área total de blob
        total_area += area

    return total_area

def calculate_blob_area(img, lower_color, upper_color, color_scale, min_blob_size, max_blob_size, invert_binary=False):
    contours, binary_image = find_contours(img, lower_color, upper_color, color_scale, invert_binary)
    if not contours:
        return 0, 0, binary_image

    # sortear contornos de mayor a menor
    contours = sort_contours(contours)
    # área del blob más grande
    biggest_blob = get_contour_area(contours[0])

    # Calcular área de blob contando solo los blobs que estén en el
    # rango de tamaño indicado por el usuario.
    #
    # Evaluar con 4 opciones:
    # Si hay área mínima y máxima de blob.
    # Si hay área mínima de blob.
    # Si hay área máxima de blob.
    # Si no hay rango de tamaños.

    if(min_blob_size and max_blob_size):
        blob_area = choose_blobs_by_min_max_area(
                        img, contours, min_blob_size, max_blob_size)
    elif(min_blob_size):
        blob_area = choose_blobs_by_min_area(
                        img, contours, min_blob_size)
    elif(max_blob_size):
        blob_area = choose_blobs_by_max_area(
                        img, contours, max_blob_size)
    # Si no hay rango, solo retornar el área total de todos los blobs
    else:
        blob_area = choose_all_blobs(
                        img, contours)

    return blob_area, biggest_blob, binary_image

def find_matches(img, template, min_calification, required_matches, color_scale, color_scale_for_binary=None, color_range=None, invert_binary=False):
    # Dimensiones del template
    width = template.shape[1]
    height = template.shape[0]

    if color_scale == "gray":
        # Convertir imagen y template a grises
        color_converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        color_converted_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    elif color_scale == "binary":
        # convertir imagen y template a color binario (blanco y negro)
        if color_scale_for_binary == "gray":
            # Si se eligió usar gray para binarizar la imagen
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # imagen a gray
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) # template a gray

            [lower, upper] = color_range

            color_converted_img = cv2.inRange(gray_img, lower, upper) # imagen a binary
            color_converted_template = cv2.inRange(gray_template, lower, upper) # template a binary

        elif color_scale_for_binary == "hsv":
            # Si se eligió usar hsv para binarizar la imagen
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # imagen a hsv
            hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV) # template a hsv

            [lower, upper] = color_range # rango para binarizar imagen
            lower, upper = np.array(lower), np.array(upper) # convertir a arreglo de numpy

            color_converted_img = cv2.inRange(hsv_img, lower, upper) # imagen a binary
            color_converted_template = cv2.inRange(hsv_template, lower, upper) # template a binary

        if invert_binary:
            # invertir binarizado (lo negro se pone blanco, y lo blanco, negro)
            color_converted_img = cv2.bitwise_not(color_converted_img)
            color_converted_template = cv2.bitwise_not(color_converted_template)


    # Template matching
    try:
        res = cv2.matchTemplate(color_converted_img, color_converted_template, cv2.TM_CCOEFF_NORMED)
    except:
        raise excepts.MT_ERROR()

    # Mejor coincidencia encontrada
    _, best_match, _, max_loc = cv2.minMaxLoc(res)
    # Lista donde se guardarán las coincidencias
    matches_locations = []

    if required_matches == 1 and best_match >= min_calification:
        matches_locations.append(max_loc)

    else:
        # Bucle para filtrar múltiples coincidencias
        while True:
            try:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc

                if max_val >= min_calification:
                    # floodfill the already found area
                    sx, sy = top_left
                    for x in range(int(sx-width/2), int(sx+width/2)):
                        for y in range(int(sy-height/2), int(sy+height/2)):
                            try:
                                res[y][x] = np.float32(-10000) # -MAX
                            except IndexError: # ignore out of bounds
                                pass
                    matches_locations.append(top_left)
                else:
                    break
            except:
                raise excepts.UNKNOWN_CF_ERROR()

    return matches_locations, best_match, color_converted_img

"""
--> Optimizar velocidad de filtrado de múltiples coincidencias en Template Matching, o
    decirle al usuario que puede crear múltiples puntos de inspección para cada coincidencia
    con 1 sola coincidencia requerida cada uno, o utilizar Template Matching múltiple
    que es más lento pero menos tardado para creación de programas.
"""
