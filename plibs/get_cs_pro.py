"""Get contours properties
Obtiene diversas propiedades de los contornos encontrados en la imagen introducida
con los parámetros introducidos. Escribe las imágenes de cada propiedad.
"""
import cv2, numpy as np
from inspector_package import files_management, cv_func

reading_directory = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/"
writing_directory = "C:/Dexill/Inspector/Alpha-Premium/x64/pd/"

if __name__ == "__main__":
    data = files_management.read_file(f"{reading_directory}/get_cs_pro_dt.di")
    data = eval(data)

    image_path, filters, parameters = data

    lower, upper, color_scale, invert_binary, closing_shape, kernel_size = parameters
    if closing_shape == "rectangle":
        closing_shape = cv2.MORPH_RECT
    elif closing_shape == "ellipse":
        closing_shape = cv2.MORPH_ELLIPSE

    image = cv2.imread(image_path)
    filtered = cv_func.apply_filters(image, filters)

    # centros de cada contorno
    contours, binary = cv_func.find_contours(
        filtered, np.array(lower), np.array(upper), color_scale, invert_binary,
    )

    contours_properties = cv_func.get_contours_properties(contours)

    # agrupar contornos cercanos
    binary_with_close_operation = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(closing_shape, kernel_size),
    )

    _, contours, _ = cv2.findContours(
        binary_with_close_operation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    regions_properties = cv_func.get_contours_properties(contours)

    # dibujar en imágenes
    image_centers = image.copy()
    for center in contours_properties["centers"]:
        # dibujar centro
        image_centers[center[1]][center[0]] = [0, 0, 255]

    image_rectangles = image.copy()
    for rectangle in contours_properties["bounding_rectangles"]:
        cv2.rectangle(
            image_rectangles, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), 
            (0,0,255), thickness=1,
        )

    image_regions = image.copy()
    for rectangle in regions_properties["bounding_rectangles"]:
        cv2.rectangle(
            image_regions, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), 
            (0,0,255), thickness=1,
        )

    cv2.imwrite(f"{writing_directory}/original.bmp", image)
    cv2.imwrite(f"{writing_directory}/filtered.bmp", filtered)
    cv2.imwrite(f"{writing_directory}/binary.bmp", binary)
    cv2.imwrite(f"{writing_directory}/contours_closed.bmp", binary_with_close_operation)
    cv2.imwrite(f"{writing_directory}/centers.bmp", image_centers)
    cv2.imwrite(f"{writing_directory}/rectangles.bmp", image_rectangles)
    cv2.imwrite(f"{writing_directory}/regions.bmp", image_regions)

    results_to_write = [
        contours_properties["centers"], 
        contours_properties["bounding_rectangles"],
        regions_properties["bounding_rectangles"],
    ]
    files_management.write_file(f"{writing_directory}/get_cs_pro_results.do", results_to_write)
