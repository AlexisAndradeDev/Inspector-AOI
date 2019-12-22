import sys
# Hacer esto para importar módulos y paquetes externos
sys.path.append('C:/Dexill/Inspector/Alpha-Premium/x64/plibs/pyv_functions/')
import math_functions

import cv2
import math
import numpy as np

TEMPLATE_MATCHING = "m"
BLOB = "b"
TRANSITION = "t"

class ObjectInspected:
    def __init__(self,board_number):
        self.number = board_number
        # el índice es igual al número del tablero menos uno, ya que el índice
        # es usado para posiciones en lista, cuya primera posición es 0.
        self.index = board_number-1
        self.status = "good" # iniciar como "bueno" por defecto
        self.inspection_points_results = ""
        self.board_results = ""
        self.results = ""

    def set_number(self, number):
        self.number = number
    def set_index(self, number):
        self.number = number
    def set_status(self, status, code=None):
        if not code:
            self.status = str(status)
        else:
            self.status = "{0};{1}".format(str(status), str(code))
    def add_inspection_point_results(self, name, light, status, results, fails):
        inspection_point_results = "{0};{1};{2};{3};{4};{5}$".format(
            self.number, name, light, status, results, fails
        )
        self.inspection_points_results += inspection_point_results

        # agregar resultados del punto a los resultados del tablero
        self.results += inspection_point_results
    def set_board_results(self, registration_time, inspection_time, stage):
        """
        El parámetro << stage >> se refiere a la etapa en la que se está: debugeo o inspección.
        Si se está en debugeo, el tiempo de registro no se escribe ya que no se registra el tablero en debugeo.
        Si se está en inspección, se escribe el tiempo de registro e inspección.
        """
        if stage == "inspection":
            self.board_results = "&{0};{1};{2};{3}#".format(
                self.number, self.status, registration_time, inspection_time
            )
        if stage == "debug":
            self.board_results = "&{0};{1};{2}#".format(
                self.number, self.status, inspection_time
            )

        # agregar resultados a los resultados del tablero
        self.results += self.board_results

    def get_index(self):
        return self.index
    def get_number(self):
        return self.number
    def get_status(self):
        return self.status
    def get_inspection_points_results(self):
        """
        Resultados de cada punto de inspección:
            * Número de tablero
            * Nombre del punto
            * Status del punto (good, bad, failed)
            * Resultados de la función de inspección (área de blob, calificación de tm)
            * Fallos del punto (si no hubo, es una lista vacía [] )
        """
        return self.inspection_points_results
    def get_board_results(self):
        """
        Resultados del tablero:
            * Número de tablero
            * Status del tablero (good, bad, failed, skip, registration_failed, error)
            * (SI SE ESTÁ EN LA ETAPA DE INSPECCIÓN): Tiempo de registro
            * Tiempo de inspección
        """
        return self.board_results
    def get_results(self):
        """
        Resultados de los puntos de inspección y del tablero combinados.
        """
        return self.results

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


def create_inspection_point(inspection_point_data):
    # diccionario
    inspection_point = {
        "light":inspection_point_data[0], # white / ultraviolet
        "name":inspection_point_data[1],
        "type":"single", # no utiliza cadenas esta versión
        "coordinates":inspection_point_data[2],
        "inspection_function":inspection_point_data[3],
        "parameters_data":inspection_point_data[4],
        "filters":inspection_point_data[5],
        "chained_inspection_points": [], # no utiliza cadenas esta versión
    }
    # parámetros de la función de inspección del punto (áreas de blob, templates de template matching, etc.)
    inspection_point["parameters"] = get_inspection_function_parameters(inspection_point)

    del inspection_point["parameters_data"] # parameters_data ya no es necesario

    return inspection_point

def create_inspection_points(data):
    inspection_points = []
    for inspection_point_data in data:
        inspection_point = create_inspection_point(inspection_point_data)
        inspection_points.append(inspection_point)
    return inspection_points

def get_inspection_function_parameters(inspection_point):
    # blob
    if (inspection_point["inspection_function"] == "b"):
        parameters = get_blob_params(inspection_point)
    # template matching
    elif (inspection_point["inspection_function"] == "m"):
        parameters = get_template_matching_params(inspection_point)
    return parameters

def get_blob_params(inspection_point):
    params_data = inspection_point["parameters_data"]
    params = {
        "invert_binary": params_data[0],
        "color_scale": params_data[1],
        "lower_color": np.array(params_data[2][0]),
        "upper_color": np.array(params_data[2][1]),
        "min_blob_size": params_data[3],
        "max_blob_size": params_data[4],
        "min_area": params_data[5],
        "max_area": params_data[6],
        "max_allowed_blob_size": params_data[7],
    }
    return params

def get_template_matching_params(inspection_point):
    params_data = inspection_point["parameters_data"]

    color_scale = params_data[0]
    if is_string(color_scale):
        name, invert_binary, color_scale_for_binary, color_range = color_scale, None, None, None
    elif is_list(color_scale):
        [name, invert_binary, color_scale_for_binary, color_range] = color_scale

    template_path = params_data[1]
    number_of_sub_templates = params_data[2]
    sub_templates = get_sub_templates(
        number_of_sub_templates, template_path, inspection_point["filters"]
    )

    min_califications = params_data[3]
    required_matches = params_data[4]

    params = {
        "color_scale": name,
        "invert_binary": invert_binary,
        "color_scale_for_binary": color_scale_for_binary,
        "color_range": color_range,
        "sub_templates": sub_templates,
        "min_califications": min_califications,
        "required_matches": required_matches,
    }
    return params

def get_sub_templates(number_of_sub_templates, template_path, filters):
    sub_templates = []
    # Lista de subtemplates y la calificación mínima de cada una
    for i in range(number_of_sub_templates):
        sub_template_img = cv2.imread(template_path + "-" + str(i+1) + ".bmp")
        sub_template_img = apply_filters(sub_template_img, filters)
        sub_templates.append([sub_template_img, i])
    return sub_templates

def is_string(var):
    if type(var) is str: return True
    else: return False

def is_list(var):
    if type(var) is list: return True
    else: return False


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
        aligned_board_image, board, inspection_points, stage, check_mode, aligned_board_image_ultraviolet=None,
    ):
    global results
    """
    Recibe como parámetros:
        * Primer punto de inspección.
        * Último punto de inspección.
        * Imagen del tablero alineado con luz blanca.
        * Objeto ObjectInspected el tablero.
        * Etapa (debugeo o inspección).
        * Modo de revisión (no, low, advanced).
        * (Opcional) Imagen del tablero alineado con luz ultravioleta. Por defecto, es None.

    Al finalizar la función, habrá escrito en el atributo "results" del
    objeto del tablero (board.results) los resultados del tablero.

    También cambiará el status del tablero (board.get_status()):
        * El tablero es recibido como parámetro de la función con un estado bueno.
        * El estado del tablero es malo si hubo defectos y no hubo fallos.
        * El estado del tablero es fallido si hubo un fallo al inspeccionar.
        * No se puede cambiar del estado fallido a otro.
    """
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
            inspection_image = crop_image(aligned_board_image_ultraviolet,inspection_point["coordinates"])
        else:
            inspection_image = crop_image(aligned_board_image,inspection_point["coordinates"])

        inspection_image_filt = apply_filters(
            inspection_image,
            inspection_point["filters"]
            )

        fails, window_results, window_status, resulting_images = inspect_point(inspection_image_filt, inspection_point)

        # Escribir imágenes sin filtrar de puntos de inspección malos si se activó el modo de revisión bajo
        if(window_status == "bad" and check_mode == "check:low"):
            cv2.imwrite(
                "{0}{1}-{2}-{3}-rgb.bmp".format(images_path, board.get_number(), inspection_point["name"], inspection_point["light"]),
                inspection_image
            )

        # Escribir imágenes filtradas de puntos de inspección malos si se activó el modo de revisión avanzado
        elif(window_status == "bad" and check_mode == "check:advanced" and resulting_images is not None):
            cv2.imwrite(
                "{0}{1}-{2}-{3}-rgb.bmp".format(images_path, board.get_number(), inspection_point["name"], inspection_point["light"]),
                inspection_image
            )
            # Exportar imágenes
            export_images(resulting_images, board.get_number(), inspection_point["name"], inspection_point["light"], images_path)

        # Escribir todas las imágenes de todos los puntos de inspección buenos y malos con el modo de revisión total (solo usado en debugeo)
        elif (check_mode == "check:total" and resulting_images is not None):
            cv2.imwrite(
                "{0}{1}-{2}-{3}-rgb.bmp".format(images_path, board.get_number(), inspection_point["name"], inspection_point["light"]),
                inspection_image
            )
            # Exportar imágenes
            export_images(resulting_images, board.get_number(), inspection_point["name"], inspection_point["light"], images_path)

        # El estado del tablero es malo si hubo un defecto y no hubo fallos.
        # El estado del tablero es fallido si hubo un fallo al inspeccionar
        # y no se puede cambiar del estado fallido a otro.
        if (window_status == "bad" and board.get_status() != "failed"):
            board.set_status("bad")
        if (window_status == "failed"):
            board.set_status("failed")

        # Agregar resultados al string que se utilizará para escribirlos en el archivo ins_results.io
        board.add_inspection_point_results(inspection_point["name"], inspection_point["light"], window_status, window_results, fails)

def export_images(images, board_number, ins_point_name, light, images_path):
    # Exportar imágenes del proceso de la función skip
    for image_name, image in images:
        # num_de_tablero-nombre_de_punto_de_inspección-luz_usada(ultraviolet/white)-nombre_de_imagen
        cv2.imwrite("{0}{1}-{2}-{3}-{4}.bmp".format(images_path, board_number, ins_point_name, light, image_name), image)


# Image Tools
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

def read_image(path):
    img = cv2.imread(path)
    return img

def write_image(path,img):
    img = cv2.imwrite(path,img)

def bgr2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray

def bgr2hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return hsv

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
            hsv = bgr2hsv(img)
            # Aplicar threshold para encontrar los colores para encontrar el fiducial
            threshold_range = filter[1]
            [lower, upper] = threshold_range
            img = cv2.inRange(hsv, np.array(lower), np.array(upper))
            img = cv2.bitwise_not(img)
        elif filter[0] == "bitwise":
            # Convertir imagen de fiducial a HSV
            hsv = bgr2hsv(img)
            # Aplicar threshold para encontrar los colores para encontrar el fiducial
            threshold_range = filter[1]
            [lower, upper] = threshold_range
            thresholded = cv2.inRange(hsv, np.array(lower), np.array(upper))
            img = cv2.bitwise_and(img, img, mask = thresholded)
        elif filter[0] == "reverseBitwise":
            # Convertir imagen de fiducial a HSV
            hsv = bgr2hsv(img)
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


# Registration tools
def find_contours(img, lower, upper, color_scale, invert_binary=False):
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

def find_fiducial(photo, fiducial):
    # lista de imágenes que se retornará para ser exportada
    images_to_export = []
    # Recortar el área en que se buscará el fiducial
    x1,y1,x2,y2 = fiducial.window
    searching_area_img = photo[y1:y2, x1:x2]

    # Aplicarle filtros secundarios (como blurs)
    searching_area_img_filtered = apply_filters(searching_area_img, fiducial.filters)

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
        area = get_contour_area(cnt)
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
    photo, trMat = rotate(photo, rotation)

    # Multiplicar el centro del fiducial desaliando por la matriz de rotación usada para rotar la imagen, para encontrar el centro rotado
    fiducial_1_rotated_center = math_functions.multiply_matrices(fiducial_1_center, trMat)
    fiducial_1_rotated_center = (int(fiducial_1_rotated_center[0]), int(fiducial_1_rotated_center[1]))

    # Se traslada la diferencia entre coordenadas del fiducial 1 trazado en la creación de programas menos el fiducial 1 encontrado
    x_diference = objective_x - fiducial_1_rotated_center[0]
    y_diference = objective_y - fiducial_1_rotated_center[1]

    photo = translate(photo, x_diference, y_diference)

    images_to_export += [["board_aligned", photo]]

    if return_rotation_and_translation:
        return fail_code, images_to_export, photo, rotation, [x_diference, y_diference]
    else: return fail_code, images_to_export, photo


# Inspection tools
def inspection_function_blob(inspection_point_image, inspection_point):
    """
    Las imágenes a exportar cuando se utiliza blob son:
    imagen filtrada, imagen binarizada.
    """
    images_to_export = [["filtered", inspection_point_image]]
    blob_area, biggest_blob, binary_image = calculate_blob_area(
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

    # fails, window_results, window_status, resulting_images
    return [], [blob_area, biggest_blob], status, images_to_export

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
    fails = []
    status = None
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
            fails.append("TEMPLATE_DOESNT_EXIST-{}".format(sub_template_index+1))
            continue

        # Dimensiones del template
        template_height = sub_template_image.shape[0]
        template_width = sub_template_image.shape[1]

        # Encontrar coincidencias
        fail_code, matches_locations, best_match, color_converted_img = find_matches(
            inspection_point_image, sub_template_image, min_calification,
            inspection_point["parameters"]["required_matches"],
            inspection_point["parameters"]["color_scale"],
            inspection_point["parameters"]["invert_binary"],
            inspection_point["parameters"]["color_scale_for_binary"],
            inspection_point["parameters"]["color_range"],
        )
        images_to_export += [["color_converted", color_converted_img]]

        if fail_code:
            status = "failed"
            fails.append(fail_code)
            continue

        # Evaluar el punto de inspección
        matches_number = len(matches_locations)
        if(matches_number == inspection_point["parameters"]["required_matches"]):
            correct_matches_number = True
            status = "good"
            break

    # Si no se encontraron las coincidencia necesarias con ninguna subtemplate
    if not correct_matches_number and status != "failed":
        status = "bad"

    # Si se encontró al menos una coincidencia, exportar imagen con las coincidencias marcadas
    if matches_number:
        matches_image = inspection_point_image.copy()
        # Dibujar rectángulos en las coincidencias
        for match_location in matches_locations:
            x = match_location[0]
            y = match_location[1]
            # Dibujar un rectángulos en la coincidencia
            cv2.rectangle(matches_image, (x, y),
                         (x+template_width, y+template_height),
                         (0, 255, 0), 2)

        images_to_export += [["matches", matches_image]]
        # fails, window_results, window_status, resulting_images
        return fails, [matches_number, best_match], status, images_to_export
    # Si no se encontraron coincidencias, exportar sólo imagen filtrada
    else:
        # fails, window_results, window_status, resulting_images
        return fails, [matches_number, best_match], status, images_to_export


def find_matches(img, template, min_calification, required_matches, color_scale, invert_binary=False, color_scale_for_binary=None, color_range=None):
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
        return "MT_FAIL", None, None, None

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
                return "Unknown_CF_error", None, None, None

    return None, matches_locations, best_match, color_converted_img

"""
--> Optimizar velocidad de filtrado de múltiples coincidencias en Template Matching, o
    decirle al usuario que puede crear múltiples puntos de inspección para cada coincidencia
    con 1 sola coincidencia requerida cada uno, o utilizar Template Matching múltiple
    que es más lento pero menos tardado para creación de programas.
"""
