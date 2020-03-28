from inspector_package import ins_func, reg_methods_func, operations, cv_func, ins_loop_func

if __name__ == '__main__':
    data = operations.read_file("C:/Dexill/Inspector/Alpha-Premium/x64/inspections/data/dt.ii")
    # Convertir string data a una lista de python
    data = eval(data)
    [settings_data, registration_data, references_data] = data


    # Datos de configuración
    [uv_inspection, boards_num, threads_num_for_boards, threads_num_for_references,
    check_mode, boards_coordinates, skip_function_data] = settings_data

    # Función de inspección para verificar que el tablero N esté en la imagen,
    # si no pasa la función, no se inspecciona el tablero
    skip_function = ins_func.create_algorithm(skip_function_data)

    settings = { # diccionario con datos de configuración
        "images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/inspections/bad_windows_results/",
        "uv_inspection":uv_inspection,
        "boards_num":boards_num,
        "boards_coordinates":boards_coordinates,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_references":threads_num_for_references,
        "check_mode":check_mode,
        "skip_function":skip_function,
    }


    # Datos de registro del tablero (alineación de imagen)
    [registration_method, method_data] = registration_data

    if registration_method == "circular_fiducials":
        [fiducials_windows, min_diameters, max_diameters, min_circle_perfections,
        max_circle_perfections, objective_angle, [objective_x, objective_y],
        fiducials_filters] = method_data

        # Crear 2 objetos con los datos de los fiduciales 1 y 2
        fiducial_1 = reg_methods_func.Fiducial(
            1, fiducials_windows[0], min_diameters[0],
            max_diameters[0], min_circle_perfections[0],
            max_circle_perfections[0], fiducials_filters[0])

        fiducial_2 = reg_methods_func.Fiducial(
            2, fiducials_windows[1], min_diameters[1],
            max_diameters[1], min_circle_perfections[1],
            max_circle_perfections[1], fiducials_filters[1])

        registration_settings = {
            "method":"circular_fiducials",
            "fiducial_1":fiducial_1,
            "fiducial_2":fiducial_2,
            "objective_x":objective_x,
            "objective_y":objective_y,
            "objective_angle":objective_angle,
        }

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


    # Referencias
    references = ins_func.create_references(references_data)

    # Iniciar el bucle de inspección
    results = "" # crear variable global para resultados
    ins_loop_func.start_inspection_loop(references, registration_settings, settings, "inspection")
