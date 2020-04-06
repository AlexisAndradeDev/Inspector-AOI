class ObjectInspected:
    def __init__(self,board_number):
        self.number = board_number
        # el í­ndice es igual al número del tablero menos uno, ya que el índice
        # es usado para posiciones en lista, cuya primera posición es 0.
        self.index = board_number-1
        self.status = "good" # iniciar como "bueno" por defecto
        self.code = None # inicializar código de error o de fallo
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
        self.status = str(status)
        if code:
            self.code = str(code)
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

        self.results = "{0}&{1};{2};{3}%%".format(
            self.references_results, self.number, self.status, self.code,
        )
    def get_results(self):
        """
        Resultados de las referencias y generales del tablero.
        """
        return self.results


def create_algorithm_results_string(name, light, status, results, fails):
    algorithm_results = "{0};{1};{2};{3};{4}$".format(
        name, light, status, results, fails
    )
    return algorithm_results

def create_inspection_point_results_string(algorithms_results, name, status):
    inspection_point_results = "{0}&&&{1};{2}%".format(
        algorithms_results, name, status
    )
    return inspection_point_results

def create_reference_results_string(inspection_points_results, name, status, reference_algorithm_results):
    reference_results = "{0}&&{1};{2};{3}#".format(
        inspection_points_results, name, status, reference_algorithm_results
    )
    return reference_results

def evaluate_status(status, object_status):
    """El estado del objeto es malo si hubo un defecto y no hubo fallos.
    El estado del objeto es fallido si hubo un fallo al inspeccionar.
    No se puede cambiar del estado fallido a otro.

    * IMPORTANTE ---> NO FUNCIONA PARA PUNTOS DE INSPECCIÓN <--- IMPORTANTE"""

    if status == "bad" and object_status != "failed":
        object_status = "bad"
    if status == "failed":
        object_status = "failed"
    return object_status

def evaluate_inspection_point_status(algorithm_status, inspection_point_status, algorithm):
    """Determina el status de un punto de inspección."""
    # transformar status del algoritmo solo para simplificar la evaluación del IP,
    # no modificarlo fuera de esta función
    if algorithm_status == algorithm["needed_status_to_be_good"]:
        algorithm_status = "good"
    algorithm_status = evaluate_status(algorithm_status, algorithm_status)

    if algorithm["ignore_bad_status"] and algorithm_status == "bad":
        # si se ignorará el status de algoritmo "bad" y el status de éste es "bad", el algoritmo será "good"
        algorithm_status = "good"

    inspection_point_status = evaluate_status(algorithm_status, inspection_point_status)

    return inspection_point_status


def add_algorithm_results_string_to_algorithms_results(algorithm, algorithm_results, algorithms_results):
    algorithms_results["string"] += algorithm_results["string"]
    return algorithms_results

def add_algorithm_results_to_algorithms_results(algorithm, algorithm_results, algorithms_results,
        add_string=True):

    # añadir string de resultados
    if add_string:
        algorithms_results = add_algorithm_results_string_to_algorithms_results(algorithm, algorithm_results, algorithms_results)

    # añadir lista de resultados
    algorithms_results["results"][algorithm["name"]] = algorithm_results["results"]

    # añadir status del algoritmo
    algorithms_results["algorithms_status"][algorithm["name"]] = algorithm_results["status"]

    # añadir localización del algoritmo
    algorithms_results["locations"][algorithm["name"]] = algorithm_results["location"]

    # añadir imágenes del algoritmo
    if algorithm_results["images"]:
        algorithms_results["images"].append(
            [algorithm["name"], algorithm["light"], algorithm_results["images"]],
        )

    return algorithms_results

def get_algorithm_results(algorithm_results, algorithm, results, status, fails, location, images):
    # crear string de resultados
    algorithm_results["string"] = create_algorithm_results_string(
        algorithm["name"], algorithm["light"], status, results, fails,
    )

    algorithm_results["status"] = status
    algorithm_results["fails"] = fails
    algorithm_results["location"] = location
    algorithm_results["images"] = images

    return algorithm_results
