from inspector_package import ins_func, reg_methods_func, operations, cv_func, ins_loop_func

if __name__ == '__main__':
    data = operations.read_file(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/regallbrds_dt.di")
    # Convertir string data a una lista
    data = eval(data)
    [settings_data, registration_data] = data


    # Datos de configuración
    [uv_inspection, images_path, photos_num, boards_per_photo, threads_num_for_photos,
    threads_num_for_boards, boards_coordinates, registration_mode] = settings_data

    if images_path[-1] != "/":
        images_path += "/"

    settings = { # diccionario con datos de configuración
        "uv_inspection":uv_inspection,
        "read_images_path":images_path,
        "check_mode_images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/pd/",
        "photos_num":photos_num,
        "boards_per_photo":boards_per_photo,
        "threads_num_for_photos":threads_num_for_photos,
        "threads_num_for_boards":threads_num_for_boards,
        "boards_coordinates":boards_coordinates,
        "registration_mode":registration_mode,
        "check_mode":"check:total",
    }


    # Datos de registro del tablero (alineación de imagen)
    [registration_method, method_data] = registration_data

    if registration_method == "rotation_points_and_translation_point":
        [rotation_point1_data, rotation_point2_data, translation_point_data,
        objective_angle, [objective_x, objective_y], rotation_iterations] = method_data

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

        registration_settings = {
            "method":"rotation_points_and_translation_point",
            "rotation_point1":rotation_point1,
            "rotation_point2":rotation_point2,
            "translation_point":translation_point,
            "objective_x":objective_x,
            "objective_y":objective_y,
            "objective_angle":objective_angle,
            "rotation_iterations":rotation_iterations,
        }


    # Iniciar el bucle de inspección
    ins_loop_func.start_inspection_loop(references=None, registration_settings=registration_settings, settings=settings, stage="registration")
