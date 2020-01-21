from cv2 import imwrite

class ObjectInspected:
    def __init__(self,photo_number,board_number):
        self.photo_number = photo_number
        self.board_number = board_number
        # el índice es igual al número del tablero menos uno, ya que el índice
        # es usado para posiciones en lista, cuya primera posición es 0.
        self.board_index = board_number-1
        self.status = "good" # iniciar como "bueno" por defecto
        self.inspection_points_results = ""
        self.board_results = ""
        self.results = ""

    def set_photo_number(self, photo_number):
        self.photo_number = photo_number
    def set_board_number(self, board_number):
        self.board_number = board_number
    def set_board_index(self, board_number):
        self.board_number = board_number
    def set_status(self, status, code=None):
        if not code:
            self.status = str(status)
        else:
            self.status = "{0};{1}".format(str(status), str(code))
    def add_inspection_point_results(self, name, light, status, results, fails):
        inspection_point_results = "{0};{1};{2};{3};{4};{5}$".format(
            self.board_number, name, light, status, results, fails
        )
        self.inspection_points_results += inspection_point_results

        # agregar resultados del punto a los resultados del tablero
        self.results += inspection_point_results
    def set_board_results(self):
        self.board_results = "&{0};{1}#".format(
            self.board_number, self.status
        )

        # agregar resultados a los resultados del tablero
        self.results += self.board_results

    def get_board_index(self):
        return self.board_index
    def get_board_number(self):
        return self.board_number
    def get_status(self):
        return self.status
    def get_inspection_points_results(self):
        """
        Resultados de cada punto de inspección:
            * Número de tablero
            * Nombre del punto
            * Status del punto (good, bad, failed)
            * Resultados de la función de inspección (área de blob, calificación de tm)
            * Fallos del punto (si no hubo, es una lista vací­a [] )
        """
        return self.inspection_points_results
    def get_board_results(self):
        """
        Resultados del tablero:
            * Número de tablero
            * Status del tablero (good, bad, failed, skip, registration_failed, error)
            * (SI SE ESTÁ EN LA ETAPA DE INSPECCIóN): Tiempo de registro
            * Tiempo de inspección
        """
        return self.board_results
    def get_results(self):
        """
        Resultados de los puntos de inspección y del tablero combinados.
        """
        return self.results


def export_images(images, photo_number, board_number, ins_point_name, light, images_path):
    # Exportar imágenes de un punto de inspección
    for image_name, image in images:
        imwrite("{0}{1}-{2}-{3}-{4}-{5}.bmp".format(images_path, photo_number, board_number, ins_point_name, light, image_name), image)

def add_to_images_name(images, str_):
    """
    Es utilizado para agregar una cadena de texto al nombre de todas las
    imágenes que son retornadas por funciones de inspeccin y métodos de registro
    para ser exportadas.
    """
    for image_index in range(len(images)):
        image_name, image = images[image_index]
        new_name = image_name + str_
        # actualizar nombre
        images[image_index][0] = new_name

    return images
