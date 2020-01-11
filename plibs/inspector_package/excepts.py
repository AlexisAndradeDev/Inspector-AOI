class MT_ERROR(Exception):
    """Se lanza al tener un error al utilizar cv2.matchTemplate en find_matches
    del módulo cv_func."""
    def __init__(self):
        # al convertir a str la excepción, su valor será args.
        self.args = ["MT_ERROR"]

class UNKNOWN_CF_ERROR(Exception):
    """Se lanza al tener un error en el bucle para filtrar múltiples coincidencias
    en find_matches del módulo cv_func."""
    def __init__(self):
        # al convertir a str la excepción, su valor será args.
        self.args = ["UNKNOWN_CF_ERROR"]
