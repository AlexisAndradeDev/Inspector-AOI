"""
Se utiliza para calcular ciertos datos de registro, como el ángulo entre los
puntos de rotación de la imagen que se introdujo, o las coordenadas de los centros
de los puntos de referencia.

* La imagen introducida debe estar previamente recortada, de tal forma que sólo
se muestre la región del tablero.
"""

from inspector_package import cv_func, reg_methods_func, operations
import cv2

if __name__ == "__main__":
    data = operations.read_file("C:/Dexill/Inspector/Alpha-Premium/x64/pd/calregdat_dt.di")
    # Convertir string data a una lista de python
    data = eval(data)
    [image_path, registration_method_data] = data

    board_image = cv2.imread(image_path)

    # Datos del método de registro
    [method_name, method_data] = registration_method_data

    if method_name == "rotation_points_and_translation_point":
        [rotation_point1_data, rotation_point2_data, translation_point_data] = method_data

        # Punto de rotación 1
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = rotation_point1_data

        rotation_point1 = cv_func.create_reference_point(
            rp_type=rp_type, name="ROTATION_POINT1", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        # Punto de rotación 2
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = rotation_point2_data

        rotation_point2 = cv_func.create_reference_point(
            rp_type=rp_type, name="ROTATION_POINT2", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        # Punto de traslación
        [rp_type, coordinates, color_scale, lower_color, upper_color, invert_binary,
        filters, contours_filters] = translation_point_data

        translation_point = cv_func.create_reference_point(
            rp_type=rp_type, name="TRANSLATION_POINT", coordinates=coordinates,
            color_scale=color_scale, lower_color=lower_color, upper_color=upper_color,
            invert_binary=invert_binary, filters=filters, contours_filters=contours_filters,
        )

        method_settings = {
            "method":"rotation_points_and_translation_point",
            "rotation_point1":rotation_point1,
            "rotation_point2":rotation_point2,
            "translation_point":translation_point,
        }


    # obtener datos que faltan para crear todos los datos del registro,
    # según el método de registro: ángulo, centros de puntos de referencia...
    fail, images_to_export, missing_data = reg_methods_func.calculate_missing_registration_data(board_image, method_settings)

    for image_data in images_to_export:
        image_name, image = image_data
        cv2.imwrite("{0}/{1}.bmp".format("C:/Dexill/Inspector/Alpha-Premium/x64/pd/", image_name), image)

    file = open("C:/Dexill/Inspector/Alpha-Premium/x64/pd/calregdat_results.do", "w+")

    if not fail:
        file.write(str(missing_data))
        file.close()
    else:
        file.write("%"+fail)
        file.close()