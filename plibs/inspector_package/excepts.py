class MT_ERROR(Exception):
    """Se lanza al tener un error al utilizar cv2.matchTemplate en find_matches
    del módulo cv_func."""
    def __init__(self):
        pass

class UNKNOWN_CF_ERROR(Exception):
    """Se lanza al tener un error en el bucle para filtrar múltiples coincidencias
    en find_matches del módulo cv_func."""
    def __init__(self):
        pass
