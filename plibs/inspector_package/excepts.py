class MT_ERROR(Exception):
    """Se lanza al tener un error al utilizar cv2.matchTemplate en find_matches
    del mÃ³dulo cv_func."""
    def __init__(self):
        # al convertir a str la excepciÃ³n, su valor serÃ¡ args.
        self.args = ["MT_ERROR"]

class UNKNOWN_CF_ERROR(Exception):
    """Se lanza al tener un error en el bucle para filtrar mÃºltiples coincidencias
    en find_matches del mÃ³dulo cv_func."""
    def __init__(self):
        # al convertir a str la excepciÃ³n, su valor serÃ¡ args.
        self.args = ["UNKNOWN_CF_ERROR"]
