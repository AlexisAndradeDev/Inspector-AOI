from inspector_package import ins_func, operations, ins_loop_func

if __name__ == '__main__':
    data = operations.read_file(r"C:/Dexill/Inspector/Alpha-Premium/x64/pd/dbg_dt.di")
    # Convertir string data a una lista de python
    data = eval(data)
    [settings_data, references_data] = data


    # Datos de configuración
    [images_path, uv_inspection, boards_num, threads_num_for_boards, threads_num_for_references] = settings_data

    settings = { # diccionario con datos de configuración
        "images_path":images_path,
        "uv_inspection":uv_inspection,
        "boards_num":boards_num,
        "threads_num_for_boards":threads_num_for_boards,
        "threads_num_for_references":threads_num_for_references,
        "check_mode":"check:total",
    }


    # Referencias
    references = ins_func.create_references(references_data)

    # Iniciar el bucle de inspección
    ins_loop_func.start_inspection_loop(references=references, registration_settings="not_register", settings=settings, stage="debug")
