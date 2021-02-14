"""Inicia el proceso de registro pre-debugeo."""
from inspector_package import (read_data, ins_loop_func,
    results_management, excepts,)

if __name__ == '__main__':
    stage = "registration"
    # variables globales
    try:
        settings, registration_settings, references = read_data.get_data(stage)
    except excepts.FatalError as e:
        # escribir código de fatal_error en el archivo de resultados
        results = "%"+str(e)
        results_management.write_results(results, stage)
    else:
        # Iniciar el bucle de inspección
        ins_loop_func.start_inspection_loop(
            settings, references, registration_settings, stage,
        )
