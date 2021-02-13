"""
Funciones para leer los datos de entrada: Variables de datos y configuraciones 
para cada una de las etapas (debug, registro, inspección).
"""
from inspector_package import (cv_func, excepts, 
    files_management, inspection_objects,)

def get_rotation_points_and_translation_point(rotation_point1_data, 
        rotation_point2_data, translation_point_data):
    """
    Retorna los diccionarios de los puntos de rotación y traslación utilizados
    en el método de registro <rotation_points_and_translation_point>.

    Args:
        rotation_point1_data (list): Lista con los datos del punto de 
            rotación 1 (obtenida directamente del archivo de datos de 
            entrada).
        rotation_point2_data (list): Lista con los datos del punto de 
            rotación 1 (obtenida directamente del archivo de datos de 
            entrada).
        translation_point_data (list): Lista con los datos del punto de 
            traslación (obtenida directamente del archivo de datos de 
            entrada).

    Returns:
        rotation_point1 (dict): Diccionario con los datos del punto
            de rotación 1.
        rotation_point2 (dict): Diccionario con los datos del punto
            de rotación 2.
        translation_point (dict): Diccionario con los datos del punto
            de traslación.
    """    
    # Punto de rotación 1
    [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
    filters, contours_filters] = rotation_point1_data

    rotation_point1 = cv_func.create_reference_point(
        rp_type=rp_type, name="ROTATION_POINT1", coordinates=coordinates,
        color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
        invert_binary=invert_binary, filters=filters, 
        contours_filters=contours_filters,
    )

    # Punto de rotación 2
    [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
    filters, contours_filters] = rotation_point2_data

    rotation_point2 = cv_func.create_reference_point(
        rp_type=rp_type, name="ROTATION_POINT2", coordinates=coordinates,
        color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
        invert_binary=invert_binary, filters=filters, 
        contours_filters=contours_filters,
    )

    # Punto de traslación
    [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
    filters, contours_filters] = translation_point_data

    translation_point = cv_func.create_reference_point(
        rp_type=rp_type, name="TRANSLATION_POINT", coordinates=coordinates,
        color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
        invert_binary=invert_binary, filters=filters, 
        contours_filters=contours_filters,
    )

    return rotation_point1, rotation_point2, translation_point

def get_registration_settings(registration_data):
    """
    Retorna un diccionario con la configuración de registro.

    Args:
        registration_data (list): Lista con los datos de registro (obtenida
            directamente del archivo de datos de entrada).

    Returns:
        registration_settings (dict): Diccionario con la configuración
            de registro.
    """
    registration_method, method_data = registration_data

    if registration_method == "rotation_points_and_translation_point":
        [rotation_point1_data, rotation_point2_data, translation_point_data,
        target_angle, [target_x, target_y], rotation_iterations] = method_data

        rotation_point1, rotation_point2, translation_point = \
            get_rotation_points_and_translation_point(
                rotation_point1_data, rotation_point2_data, 
                translation_point_data,
            )

        registration_settings = {
            "method":registration_method,
            "rotation_point1":rotation_point1,
            "rotation_point2":rotation_point2,
            "translation_point":translation_point,
            "target_x":target_x,
            "target_y":target_y,
            "target_angle":target_angle,
            "rotation_iterations":rotation_iterations,
        }

    return registration_settings

def get_registration_stage_data():
    """
    Retorna la configuración general y configuración de registro para 
    la etapa de registro.

    Raises:
        excepts.FatalError("WRONG_UV_INSPECTION_FLAG"): El parámetro de 
            configuración para definir si se usará luz ultravioleta «uv_inspection» 
            está mal escrito en los datos de entrada.

    Returns:
        settings (dict): Diccionario con la configuración general.
        registration_settings (dict): Diccionario con la configuración
            de registro.
    """
    data = files_management.read_file(
        r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_dt.di"
    )
    data = eval(data) # Convertir string data a una lista
    [settings_data, registration_data] = data


    # Datos de configuración
    [read_images_path, uv_inspection, panels_num, boards_per_panel, 
    threads_num_for_panels, threads_num_for_boards, boards_coordinates, 
    registration_mode] = settings_data

    if uv_inspection == "uv_inspection:True":
        uv_inspection = True
    elif uv_inspection == "uv_inspection:False":
        uv_inspection = False
    else:
        raise excepts.FatalError("WRONG_UV_INSPECTION_FLAG") # !FATAL_ERROR

    settings = {
        "read_images_path":read_images_path,
        "check_mode_images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/pd/",
        "uv_inspection":uv_inspection,
        "panels_num":panels_num,
        "boards_per_panel":boards_per_panel,
        "threads_num_for_panels":threads_num_for_panels,
        "threads_num_for_boards":threads_num_for_boards,
        "check_mode":"check:total",
        "boards_coordinates":boards_coordinates,
        "registration_mode":registration_mode,
    }


    # Datos de registro
    registration_settings = get_registration_settings(registration_data)

    return settings, registration_settings


def get_inspection_stage_data():
    """
    Retorna la configuración general, datos de las referencias y configuración de
    registro para inspección.

    Raises:
        excepts.FatalError("WRONG_UV_INSPECTION_FLAG"): El parámetro de 
            configuración para definir si se usará luz ultravioleta 
            «uv_inspection» está mal escrito en los datos de entrada.

    Returns:
        settings (dict): Diccionario con la configuración general.
        registration_settings (dict): Diccionario con la configuración
            de registro.
        references (list): Lista con los diccionarios de las referencias
            creados con 'create_reference'.
    """
    data = files_management.read_file(
        r"C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/dt.ii"
    )
    data = eval(data) # Convertir string data a una lista
    [settings_data, registration_data, references_data] = data


    # Datos de configuración
    [uv_inspection, boards_per_panel, threads_num_for_boards, 
    threads_num_for_references, check_mode, boards_coordinates, 
    registration_mode, skip_function_data] = settings_data

    if uv_inspection == "uv_inspection:True":
        uv_inspection = True
    elif uv_inspection == "uv_inspection:False":
        uv_inspection = False
    else:
        raise excepts.FatalError("WRONG_UV_INSPECTION_FLAG") # !FATAL_ERROR

    # Función de inspección para verificar que cada tablero esté en la imagen,
    # si un tablero no pasa la función de inspección, no se inspecciona el tablero
    skip_function = inspection_objects.create_algorithm(
        container_inspection_point=None, algorithm_data=skip_function_data,
    )

    settings = {
        "read_images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/",
        "check_mode_images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/inspections/bad_windows_results/",
        "uv_inspection":uv_inspection,
        "panels_num":1,
        "boards_per_panel":boards_per_panel,
        "threads_num_for_panels":1,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_references":threads_num_for_references,
        "check_mode":check_mode,
        "boards_coordinates":boards_coordinates,
        "registration_mode":registration_mode,
        "skip_function":skip_function,
    }


    # Datos de registro
    registration_settings = get_registration_settings(registration_data)

    # Referencias
    references = inspection_objects.create_references(references_data)

    return settings, registration_settings, references

def get_debug_stage_data():
    """
    Retorna la configuración general y los datos de las referencias para debugeo.

    Raises:
        excepts.FatalError("WRONG_UV_INSPECTION_FLAG"): El parámetro de 
            configuración para definir si se usará luz ultravioleta «uv_inspection» 
            está mal escrito en los datos de entrada.

    Returns:
        settings (dict): Diccionario con la configuración general.
        references (list): Lista con los diccionarios de las referencias
            creados con 'create_reference'.
    """
    data = files_management.read_file(
        r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/dbg_dt.di"
    )
    data = eval(data) # Convertir string data a una lista
    [settings_data, references_data] = data


    # Datos de configuración
    [read_images_path, uv_inspection, panels_num, boards_per_panel, 
    threads_num_for_panels, threads_num_for_boards, threads_num_for_references, 
    check_mode] = settings_data

    if uv_inspection == "uv_inspection:True":
        uv_inspection = True
    elif uv_inspection == "uv_inspection:False":
        uv_inspection = False
    else:
        raise excepts.FatalError("WRONG_UV_INSPECTION_FLAG") # !FATAL_ERROR

    settings = {
        "read_images_path":read_images_path,
        "check_mode_images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/pd/",
        "uv_inspection":uv_inspection,
        "panels_num":panels_num,
        "boards_per_panel":boards_per_panel,
        "threads_num_for_panels":threads_num_for_panels,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_references":threads_num_for_references,
        "check_mode":check_mode,
    }

    # Referencias
    references = inspection_objects.create_references(references_data)

    return settings, references

def get_data(stage):
    """Retorna las variables de configuración.

    Args:
        stage (str): Etapa que se ejecutará. 
            'debug', 'inspection', 'registration'.

    Returns:
        settings (dict): Diccionario con la configuración general.
        registration_settings (dict): Diccionario con la configuración
            de registro.
            En etapa de debugeo, tomará valor de {}
        references (list): Lista con los diccionarios de las referencias
            creados con 'create_reference'.
            En etapa de registro, tomará valor de []
    """
    if stage == 'inspection':
        settings, registration_settings, references = \
            get_inspection_stage_data()
    elif stage == 'debug':
        registration_settings = {}
        settings, references = get_debug_stage_data()
    elif stage == 'registration':
        references = []
        settings, registration_settings = \
            get_registration_stage_data()

    return settings, registration_settings, references
