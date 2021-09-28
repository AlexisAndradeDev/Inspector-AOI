"""Clases y diccionarios de los objectos que se inspeccionarán, como 
paneles, tableros o algoritmos."""

from inspector_package import results_management, ins_func

class Panel:
    """Clase para almacenar información sobre un panel."""
    def __init__(self,number):
        """
        Args:
            number (int): número del panel.
        """
        self.number = number
        self.status = "good"
        self.code = None
        self.results = ""
        self.boards = []

    def get_number(self):
         return self.number
    def set_number(self, number):
        self.number = number

    def get_code(self):
        """Retorna el código de error/fallo del panel."""
        return self.code
    def set_code(self, code):
        """Asigna un código de error/fallo al panel."""
        self.code = code

    def add_board(self, board):
        """Añade el objeto de un tablero a una lista para que el panel pueda
        acceder a él."""
        self.boards.append(board)
    def get_boards(self):
        """Retorna la lista de tableros que contiene."""
        return self.boards

    def get_status(self):
        return self.status
    def set_status(self, new_status):
        """Asigna un status al panel."""
        self.status = new_status
    def update_status(self, new_status):
        """Evalúa si el estado actual debe cambiarse por el estado
        nuevo introducido; si debe cambiarse, lo cambia.

        Args:
            new_status (str): Estado que se quiere asignar.
        """
        if results_management.status_has_to_change(
                status=self.get_status(), new_status=new_status):
            self.set_status(new_status)

    def get_results_string(self):
        return self.results
    def create_results_string(self):
        """Crea el string de resultados."""
        self.results = "{0};{1};{2};{3}#".format(
            "panel", self.get_number(), self.get_status(), self.get_code(),
        )

    def set_as_skip(self):
        """Asigna el estado de skip. Crea también el string de resultados."""
        self.set_status("skip")
        self.create_results_string()
    def set_as_registration_failed(self, code):
        """Asigna el estado de fallo de registro y le asigna el 
        código del fallo. Crea también el string de resultados."""
        self.set_status("registration_failed")
        self.set_code(code)
        self.create_results_string()
    def set_as_error(self, code):
        """Asigna el estado de error y le asigna el código del error. 
        Crea también el string de resultados."""
        self.set_status("error")
        self.set_code(code)
        self.create_results_string()


class Board:
    """Clase para almacenar información sobre un tablero."""
    def __init__(self,container_panel,position_in_panel):
        """
        Args:
            panel (inspection_objects.Panel): Panel que contiene a 
                este tablero.
            position_in_panel (int): Número de posición del tablero dentro
                del panel.
        """
        self.container_panel = container_panel
        self.position_in_panel = position_in_panel
        self.status = "good"
        self.code = None
        self.results = ""

        self.container_panel.add_board(self)

    def get_container_panel(self):
        """Retorna el objeto de panel que contiene a este tablero."""
        return self.container_panel

    def get_position_in_panel(self):
        """Retorna el número de posición del tablero dentro del panel."""
        return self.position_in_panel

    def get_code(self):
        """Retorna el código de error/fallo del tablero."""
        return self.code
    def set_code(self, code):
        """Asigna un código de error/fallo al tablero."""
        self.code = code

    def get_status(self):
        """Retorna el status del tablero."""
        return self.status
    def set_status(self, new_status):
        """Asigna un status al tablero."""
        self.status = new_status
    def update_status(self, new_status):
        """Evalúa si el estado actual debe cambiarse por el estado
        nuevo introducido; si debe cambiarse, lo cambia.

        Args:
            new_status (str): Estado que se quiere asignar.
        """
        if results_management.status_has_to_change(
                status=self.get_status(), new_status=new_status):
            self.set_status(new_status)

    def get_results_string(self):
        return self.results
    def create_results_string(self):
        """Crea el string de resultados."""
        self.results = "{0};{1};{2};{3};{4}#".format(
            "board", self.container_panel.get_number(), 
            self.get_position_in_panel(), self.get_status(), 
            self.get_code(),
        )

    def set_as_skip(self):
        """Asigna el estado de skip. Crea también el string de resultados."""
        self.set_status("skip")
        self.create_results_string()
    def set_as_registration_failed(self, code):
        """Asigna el estado de fallo de registro y le asigna el 
        código del fallo. Crea también el string de resultados."""
        self.set_status("registration_failed")
        self.set_code(code)
        self.create_results_string()
    def set_as_error(self, code):
        """Asigna el estado de error y le asigna el código del error. 
        Crea también el string de resultados."""
        self.set_status("error")
        self.set_code(code)
        self.create_results_string()


def create_algorithm(container_inspection_point, algorithm_data):
    """Retorna los datos procesados de un algoritmo.
    
    Args:
        algorithm_data (list): Datos sin procesar de un algoritmo leídos
            directamente del archivo de datos de entrada.
        container_inspection_point (dict) -
            Ver inspection_objects.create_algorithms()

    Returns:
        algorithm (dict): Contiene los datos del algoritmo.
    """
    algorithm = {
        "object_type": "algorithm",
        "inspection_point":container_inspection_point,
        "ignore_bad_status":algorithm_data[0],
        "needed_status_to_be_good":algorithm_data[1],
        "take_as_origin":algorithm_data[3],
        "light":algorithm_data[4],
        "name":algorithm_data[5],
        "coordinates":algorithm_data[6],
        "ignore_in_boards":algorithm_data[7],
        "inspection_function":algorithm_data[8],
        "filters":algorithm_data[10],
    }

    chain_data = algorithm_data[2]
    [algorithm["chained_to"], algorithm["needed_status_of_chained_to_execute"]] = chain_data

    # parámetros de la función de inspección
    parameters_data = algorithm_data[9]
    algorithm["parameters"] = ins_func.get_inspection_function_parameters(
        algorithm, parameters_data
    )

    return algorithm

def create_algorithms(container_inspection_point, data):
    """Retorna una lista de algoritmos.
    
    Args:
        container_inspection_point (dict): Punto de inspección que contiene a 
            estos algoritmos.
        data (list): Datos sin procesar de los algoritmos leídos 
            directamente del archivo de datos de entrada.

    Returns:
        algorithms (list): Lista con los algoritmos procesados.
    """
    algorithms = {}
    for algorithm_data in data:
        algorithm = create_algorithm(container_inspection_point, algorithm_data)
        algorithms[algorithm["name"]] = algorithm
    return algorithms


def create_inspection_point(container_reference, inspection_point_data):
    """Retorna los datos procesados de un punto de inspección.
    
    Args:
        container_reference (dict) - 
            Ver inspection_objects.create_inspection_points()
        inspection_point_data (list): Datos sin procesar de un punto de 
            inspección leídos directamente del archivo de datos de entrada.

    Returns:
        inspection_point (dict): Datos de un punto de inspección.
    """
    inspection_point = {
        "object_type": "inspection_point",
        "reference": container_reference,
        "name":inspection_point_data[0],
        "coordinates":inspection_point_data[1],
        "ignore_in_boards":inspection_point_data[2],
    }

    algorithms_data = inspection_point_data[3]
    inspection_point["algorithms"] = create_algorithms(
        inspection_point, algorithms_data,
    )

    return inspection_point

def create_inspection_points(container_reference, data):
    """Retorna una lista de puntos de inspección.
    
    Args:
        container_reference (dict): Referencia que contiene a estos punto de 
            inspección.
        data (list): Datos sin procesar de los puntos de inspección leídos 
            directamente del archivo de datos de entrada.

    Returns:
        inspection_points (list): Lista con los puntos de inspección procesados.
    """
    inspection_points = []
    for inspection_point_data in data:
        inspection_point = create_inspection_point(
            container_reference, inspection_point_data
        )
        inspection_points.append(inspection_point)
    return inspection_points


def get_reference_algorithm(reference_algorithm_data):
    """Retorna los datos del algoritmo de referencia.
    
    Args:
        reference_algorithm_data (list): Datos sin procesar del algoritmo
            de referencia leídos directamente del archivo de datos de entrada.

    Returns:
        reference_algorithm (dict): Datos del algoritmo de referencia.
    """
    reference_algorithm = {}
    reference_algorithm["function"] = reference_algorithm_data[0]

    if reference_algorithm["function"] == "classification":
        pass
    if reference_algorithm["function"] == "displacement":
        reference_algorithm["algorithm1_to_measure"] = reference_algorithm_data[1]
        reference_algorithm["algorithm2_to_measure"] = reference_algorithm_data[2]
        reference_algorithm["x_translation_tolerance"] = reference_algorithm_data[3]
        reference_algorithm["y_translation_tolerance"] = reference_algorithm_data[4]
        reference_algorithm["rotation_tolerance"] = reference_algorithm_data[5]
        reference_algorithm["min_width"] = reference_algorithm_data[6]
        reference_algorithm["max_width"] = reference_algorithm_data[7]

    return reference_algorithm

def create_reference(reference_data):
    """Retorna los datos procesados de una referencia.
    
    Args:
        reference_data (list): Datos sin procesar de una referencia leídos
            directamente del archivo de datos de entrada.

    Returns:
        reference (dict): Datos de una referencia.
    """
    reference = {
        "object_type": "reference",
        "name":reference_data[0],
        "ignore_in_boards":reference_data[2],
    }

    reference_algorithm_data = reference_data[1]
    reference["reference_algorithm"] = get_reference_algorithm(
        reference_algorithm_data,
    )

    inspection_points_data = reference_data[3]
    reference["inspection_points"] = create_inspection_points(
        reference, inspection_points_data,
    )

    return reference

def create_references(data):
    """Retorna una lista de referencias.
    
    Args:
        data (list): Datos sin procesar de las referencias leídos directamente 
            del archivo de datos de entrada.

    Returns:
        references (list): Lista de referencias procesadas.
    """
    references = []
    for reference_data in data:
        reference = create_reference(reference_data)
        references.append(reference)
    return references


def get_container_of_inspection_object(inspection_object):
    """Retorna el contenedor de un objeto de inspección."""
    if type(inspection_object) == dict:
        if inspection_object["object_type"] == "algorithm":
            container = inspection_object["inspection_point"]
        if inspection_object["object_type"] == "inspection_point":
            container = inspection_object["reference"]
        if inspection_object["object_type"] == "reference":
            container = None

    elif type(inspection_object) == Board:
        container = inspection_object.get_container_panel()
    elif type(inspection_object) == Panel:
        container = None

    return container

def get_containers_of_inspection_object(inspection_object, unlinked_container=None, containers=None):
    """
    Retorna todos los contenedores posibles de obtener a partir de un objeto de 
    inspección.

    Por ejemplo, los contenedores de un algoritmo que se pueden obtener
    a partir de sólo el objeto del algoritmo son: punto de inspección y
    referencia; ya que el algoritmo puede acceder al punto de inspección, 
    que a su vez puede acceder a la referencia, pero la referencia no puede 
    acceder al tablero ni al panel.
    
    Un tablero puede obtener su panel.

    Args:
        inspection_object (dict or Board or Panel): Objeto del que se quiere
            obtener sus contenedores. Ejemplo, el diccionario de un
            algoritmo o el objeto de un panel.
        unlinked_container (None or dict or Board or Panel): Es el objeto
            del contenedor al que no se puede acceder desde el objeto de
            inspección. 
            
            Ejemplo: si se quiere obtener todos los contenedores
            de un algoritmo, se necesita tener el objeto del tablero, ya que
            no se puede acceder a él sólo con el objeto del algoritmo; este
            parámetro recibiría el objeto del tablero.
            
            Soluciona el problema mencionado en la descripción de la función.

            Defaults to None.
        containers (list, optional): Lista que almacena los contenedores del
            objeto. Se le debe asignar la lista creada en esta función 
            recursiva.

            Advertencia: No debe asignarse por default a una lista vacía; 
            al ser una lista un objeto mutable, ocasionará problemas si se 
            deja como un valor por defecto de un parámetro de función.

            Defaults to None.
    
    Returns:
        containers (list): Lista de contenedores del objeto de inspección en
            orden ascendente.
    """
    if not containers:
        containers = []

    # obtener contenedor de este objeto
    container = get_container_of_inspection_object(inspection_object)
    if not container:
        if unlinked_container:
            container = unlinked_container
            # no puede volver a utilizarse el unlinked_container; si se
            # pasara de nuevo en la función recursiva, provocaría recursión
            # infinita
            unlinked_container = None
        else:
            return containers
    containers.append(container)

    # agregar los demás contenedores con función recursiva
    get_containers_of_inspection_object(
        container, unlinked_container=unlinked_container, containers=containers,
    )

    return containers
