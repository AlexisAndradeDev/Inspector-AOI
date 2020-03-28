class ObjectInspected:
    def __init__(self,board_number):
        self.number = board_number
        # el í­ndice es igual al número del tablero menos uno, ya que el índice
        # es usado para posiciones en lista, cuya primera posición es 0.
        self.index = board_number-1
        self.status = "good" # iniciar como "bueno" por defecto
        self.references_results = ""

    def set_number(self, number):
        self.number = number
    def set_index(self, index):
        self.index = index
    def get_number(self):
        return self.number
    def get_index(self):
        return self.index

    def set_status(self, status, code=None):
        if not code:
            self.status = str(status)
        else:
            self.status = "{0};{1}".format(str(status), str(code))
    def get_status(self):
        return self.status
    def evaluate_status(self, reference_status):
        """El estado del tablero es malo si hubo un defecto y no hubo fallos.
        El estado del tablero es fallido si hubo un fallo al inspeccionar.
        No se puede cambiar del estado fallido a otro."""
        if reference_status == "bad" and self.get_status() != "failed":
            self.set_status("bad")
        if reference_status == "failed":
            self.set_status("failed")

    def add_references_results(self, references_results):
        self.references_results += references_results

    def set_results(self, status="not_entered"):
        if status != "not_entered":
            self.set_status(status) # por si no se utilizó el método set_status antes

        self.results = "{0}&{1};{2}%%".format(
            self.references_results, self.number, self.status
        )
    def get_results(self):
        """
        Resultados de las referencias y generales del tablero.
        """
        return self.results


def create_algorithm_results(name, light, status, results, fails):
    algorithm_results = "{0};{1};{2};{3};{4}$".format(
        name, light, status, results, fails
    )
    return algorithm_results

def create_inspection_point_results(algorithms_results, name, status):
    inspection_point_results = "{0}&&&{1};{2}%".format(
        algorithms_results, name, status
    )
    return inspection_point_results

def create_reference_results(inspection_points_results, name, status, reference_algorithm_results):
    reference_results = "{0}&&{1};{2};{3}#".format(
        inspection_points_results, name, status, reference_algorithm_results
    )
    return reference_results

def evaluate_status(status, object_status):
    """El estado del objeto es malo si hubo un defecto y no hubo fallos.
    El estado del objeto es fallido si hubo un fallo al inspeccionar.
    No se puede cambiar del estado fallido a otro."""
    if status == "bad" and object_status != "failed":
        object_status = "bad"
    if status == "failed":
        object_status = "failed"
    return object_status
