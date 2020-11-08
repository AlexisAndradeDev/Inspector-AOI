from inspector_package import ins_func, operations, ins_loop_func

if __name__ == '__main__':
    data = operations.read_file(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/dbg_dt.di")
    # Convertir string data a una lista de python
    data = eval(data)
    [settings_data, references_data] = data

    # Datos de configuración
    [images_path, uv_inspection, photos_num, boards_per_photo, threads_num_for_photos,
    threads_num_for_boards, threads_num_for_references, check_mode] = settings_data

    settings = { # diccionario con datos de configuración
        "read_images_path":images_path,
        "check_mode_images_path":"C:/Dexill/Inspector/Alpha-Premium/x64/pd/",
        "uv_inspection":uv_inspection,
        "photos_num":photos_num,
        "boards_per_photo":boards_per_photo,
        "threads_num_for_photos":threads_num_for_photos,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_references":threads_num_for_references,
        "check_mode":check_mode,
    }


    # Referencias
    references = ins_func.create_references(references_data)

    # Iniciar el bucle de inspección
    ins_loop_func.start_inspection_loop(references=references, registration_settings=None, settings=settings, stage="debug")
