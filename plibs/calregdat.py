"""
Se utiliza para calcular ciertos datos de registro, como el ángulo entre los
puntos de rotación de la imagen que se introdujo, o las coordenadas de los centros
de los puntos de referencia.
"""

from inspector_package import (cv_func, reg_methods_func, files_management, 
    read_data)
import cv2

if __name__ == "__main__":
    data = files_management.read_file(
        "C:/Dexill/Inspector/Alpha-Premium/x64/pd/calregdat_dt.di"
    )
    data = eval(data) # Convertir string data a una lista de python
    [image_path, registration_data] = data

    # Datos del método de registro
    [registration_method, method_data] = registration_data

    if registration_method == "rotation_points_and_translation_point":
        [rotation_point1_data, rotation_point2_data, translation_point_data] = method_data

        rotation_point1, rotation_point2, translation_point = \
            read_data.get_rotation_points_and_translation_point(
                rotation_point1_data, rotation_point2_data, 
                translation_point_data,
            )

        method_settings = {
            "method":"rotation_points_and_translation_point",
            "rotation_point1":rotation_point1,
            "rotation_point2":rotation_point2,
            "translation_point":translation_point,
        }


    # obtener datos que faltan para crear todos los datos del registro,
    # según el método de registro: ángulo, centros de puntos de referencia...
    board_image = cv2.imread(image_path)

    fail, images_to_export, missing_data = \
        reg_methods_func.calculate_missing_registration_data(
            board_image, method_settings
        )

    for image_name in images_to_export.keys():
        image = images_to_export[image_name]
        cv2.imwrite(
            "{0}/{1}.bmp".format(
                "C:/Dexill/Inspector/Alpha-Premium/x64/pd/", image_name),
            image,
        )

    file = open(
        "C:/Dexill/Inspector/Alpha-Premium/x64/pd/calregdat_results.do", "w+"
    )

    if not fail:
        file.write(str(missing_data))
        file.close()
    else:
        file.write("%"+fail)
        file.close()
