from cv2 import imread, imwrite
from inspector_packageOptimizandoNuevo import inspection_objects

def force_read_image(path):
    img = imread(path)
    while img is None:
        # si no se almacenó correctamente la imagen, volver a intentarlo
        img = imread(path)
    return img

def read_photos_for_registration(settings, panel_number):
    path = "{0}{1}.bmp".format(settings["read_images_path"], panel_number)
    photo = imread(path)

    if photo is None:
        fail = "IMG_DOESNT_EXIST" # !BOARD_ERROR
        return fail, None, None

    if settings["uv_inspection"]:
        path = "{0}{1}-ultraviolet.bmp".format(settings["read_images_path"], panel_number)
        photo_ultraviolet = imread(path)

        if photo_ultraviolet is None:
            fail = "UV_IMG_DOESNT_EXIST" # !BOARD_ERROR
            return fail, None, None

    else:
        photo_ultraviolet = None

    return None, photo, photo_ultraviolet

def export_registered_board_image(image, image_uv, settings, board, stage):
    """Escribe la imagen del tablero registrado.
    Si hay imagen con luz ultravioleta, también la escribe."""
    if stage == 'inspection':
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/inspections/status/"
    elif stage == 'registration':
        path = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/"

    panel = inspection_objects.get_container_of_inspection_object(board)

    # luz blanca
    imwrite(
        "{0}{1}-{2}-white-registered.bmp".format(
            path, panel.get_number(), board.get_position_in_panel()),
        image,
    )

    if image_uv is not None:
        # luz ultravioleta
        imwrite(
            "{0}{1}-{2}-ultraviolet-registered.bmp".format(
                path, panel.get_number(), board.get_position_in_panel()),
            image_uv,
        )

def export_registration_images(images, panel_number, name, light, settings,
                              registration_fail,):
    """Escribe imágenes de un proceso de registro.

    Args:
        images (dict): Contiene las imágenes del proceso de registro.
            keys (str): Nombres de las imágenes.
            values (np.ndarray): Imágenes.
        panel_number (int): Número del panel.
        name (str or int): Si es registro global, será 'global_registration'.
            Si es registro local, será el número de posición del tablero dentro
            del panel.
        light (str): Luz utilizada.
        registration_fail (None or str): Código del fallo de registro.
            Si no hubo fallo de registro, debe ser None.
        settings - Ver ins_loop_func.start_inspection_loop().
    """
    if (settings["check_mode"] == "check:total" or
            (settings["check_mode"] == "check:yes" and registration_fail)
        ):
        for image_name, image in zip(images.keys(), images.values()):
            imwrite(
                "{0}{1}-{2}-{3}-{4}.bmp".format(
                    settings["check_mode_images_path"], 
                    panel_number, name, light, image_name,
                ),
                image,
            )

def export_local_registration_images(images, board, light, settings, 
                                     registration_fail):
    """Escribe imágenes de un proceso de registro local.

    Args:
        board (inspection_objects.Board): Contiene datos del tablero.
        images, light, registration_fail - 
            Ver images_operations.write_registration_images
        settings - Ver ins_loop_func.start_inspection_loop().
    """
    panel = inspection_objects.get_container_of_inspection_object(board)

    export_registration_images(images, panel.get_number(), 
        board.get_position_in_panel(), light, settings, registration_fail,
    )

def export_global_registration_images(images, panel, light, settings, 
                                      registration_fail):
    """Escribe imágenes de un proceso de registro global.

    Args:
        board (inspection_objects.Board): Contiene datos del tablero.
        images, light, registration_fail - 
            Ver images_operations.write_registration_images
        settings - Ver ins_loop_func.start_inspection_loop().
    """
    export_registration_images(images, panel.get_number(), 
        "global_registration", light, settings, registration_fail,
    )

def write_algorithm_image(image, images_path, panel_number, board_number, 
                          reference_name, inspection_point_name, 
                          algorithm_name, light, image_name):
    """Escribe una imagen de un algoritmo.

    Args:
        image (numpy.ndarray): Imagen que se escribirá.
        panel_number (int): Número de panel.
        board_number (int): Número de tablero.
        reference_name (str): Nombre de la referencia.
        inspection_point_name (str): Nombre de punto de inspección.
        algorithm_name (str): Nombre del algoritmo
        light (str): Luz utilizada.
        image_name (str): Nombre de la imagen.
        images_path - Ver images_operations.export_algorithm_images()
    """
    imwrite(
        "{0}{1}-{2}-{3}-{4}-{5}-{6}-{7}.bmp".format(
            images_path, panel_number, board_number, 
            reference_name, inspection_point_name, algorithm_name, 
            light, image_name
        ),
        image,
    )

def export_algorithm_images(images, algorithm, board, images_path):
    """
    Exporta las imágenes de un algoritmo.

    Args:
        images (dict): Contiene las imágenes del proceso de inspección del
            algoritmo.
            keys (str): Nombres de las imágenes.
            values (np.ndarray): Imágenes.
        algorithm (dict): Contiene datos del algoritmo.
        board (inspection_objects.Board): Contiene datos del tablero.
        images_path (str): Directorio donde se escribirán las imágenes.
    """
    [inspection_point, reference, board, panel] = \
        inspection_objects.get_containers_of_inspection_object(
            algorithm, unlinked_container=board
        )

    for image_name, image in zip(images.keys(), images.values()):
        write_algorithm_image(
            image, images_path, panel.get_number(), board.get_position_in_panel(), 
            reference["name"], inspection_point["name"], algorithm["name"], 
            algorithm["light"], image_name,
        )

def export_skip_function_images(images, skip_function, board, images_path):
    """
    Exporta las imágenes de la función de skip.

    Args:
        skip_function (dict): Contiene datos de la función de skip en forma
            de algoritmo.
        board, images, images_path - 
            Ver images_operations.export_algorithm_images()
    """
    panel = inspection_objects.get_container_of_inspection_object(board)
    reference_name, inspection_point_name = "skip_function", "skip_function"

    for image_name, image in zip(images.keys(), images.values()):
        write_algorithm_image(
            image, images_path, panel.get_number(), board.get_position_in_panel(), 
            reference_name, inspection_point_name, skip_function["name"], 
            skip_function["light"], image_name,
        )

def add_to_images_name(images, str_):
    """Es utilizado para agregar una cadena de texto al nombre de todas las
    imágenes que son retornadas por funciones de inspección y métodos de 
    registro para ser exportadas."""
    for image_name, image in zip(images.keys(), images.values()):
        new_name = image_name + str_
        # actualizar nombre
        del images[image_name]
        images[new_name] = image

    return images
