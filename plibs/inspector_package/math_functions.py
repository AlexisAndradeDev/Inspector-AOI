import numpy as np
import math
import more_itertools as mit
from cv2 import getRotationMatrix2D

def elements_per_partition(number_of_elements, number_of_partitions, get_as_indexes=False):
    """Return a list of equally divided intervals."""
    if get_as_indexes:
        a = [list(c)[0] for c in mit.divide(number_of_partitions, range(0, number_of_elements))]
        b = [list(c)[-1] for c in mit.divide(number_of_partitions, range(0, number_of_elements))]
    else:
        a = [list(c)[0] for c in mit.divide(number_of_partitions, range(1, number_of_elements+1))]
        b = [list(c)[-1] for c in mit.divide(number_of_partitions, range(1, number_of_elements+1))]
    return list(zip(a, b))

def sum_lists(list1, list2):
    new_list = [list1[0]+list2[0], list1[1]+list2[1]]
    return new_list

def split_list(list, number_of_partitions):
    elements_per_partition_ = elements_per_partition(number_of_elements=len(list), number_of_partitions=number_of_partitions)

    splitted_list = []
    for partition_index in range(number_of_partitions):
        splitted_list.append(list[elements_per_partition_[partition_index][0]-1:elements_per_partition_[partition_index][1]-1])
    return splitted_list

def calculate_angle(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return math.degrees(math.atan2(yDiff, xDiff))

def rotate_coordinate_and_calculate_distance_to_center(w, h, distance_x, distance_y, angle):
    # Convertir ángulos de grados de más de una vuelta a menos de 360 (450 --> 90)
    if angle > 360:
        angle -= (angle//360)*360

    # Coordenadas a rotar
    coordinates = ( (w/2)+distance_x , (h/2)+distance_y )
    # Centro
    img_c = (w/2, h/2)
    center_x = w/2
    center_y = h/2

    trMat = getRotationMatrix2D(img_c, angle, 1)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    # new x and y
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    trMat[0, 2] += ((b_w / 2) - img_c[0])
    trMat[1, 2] += ((b_h / 2) - img_c[1])

    rotated_coordinates = multiply_matrices(coordinates, trMat)
    rotated_coordinates = (round(rotated_coordinates[0]), round(rotated_coordinates[1]))
    [rotated_x, rotated_y] = rotated_coordinates

    if angle == 0:
        distance_to_center_rotated_x = distance_x
        distance_to_center_rotated_y = distance_y
    elif 0 < angle <= 90:
        distance_to_center_rotated_x = rotated_x - center_y
        distance_to_center_rotated_y = rotated_y - center_x
    elif 90 < angle <= 180:
        distance_to_center_rotated_x = rotated_x - center_x
        distance_to_center_rotated_y = rotated_y - center_y
    elif 180 < angle <= 270:
        distance_to_center_rotated_x = rotated_x - center_y
        distance_to_center_rotated_y = rotated_y - center_x
    elif 270 < angle <= 360:
        distance_to_center_rotated_x = rotated_x - center_x
        distance_to_center_rotated_y = rotated_y - center_y

    return int(distance_to_center_rotated_x), int(distance_to_center_rotated_y)

def multiply_matrices(matrix1, matrix2):
    res = (
        np.dot(np.array([matrix1[0], matrix1[1], 1]), matrix2[0]),
        np.dot(np.array([matrix1[0], matrix1[1], 1]), matrix2[1])
        )
    return res

def count_array_items(array, axis=None):
    """
    Función copiada de Numpy:
    pkgs\numpy-base-1.15.4-py37hc3f5095_0\Lib\site-packages\numpy\core\_methods.py
    Función original: _count_reduce_items
    """
    if axis is None:
        axis = tuple(range(array.ndim))
    if not isinstance(axis, tuple):
        axis = (axis,)
    items = 1
    for ax in axis:
        items *= array.shape[ax]
    return items

def average_array(array):
    pixels_value_sum = np.sum(array)
    cnt = count_array_items(array)
    average = pixels_value_sum * 1. / cnt
    return average

def calculate_circularity(area, perimeter):
    circularity = 4*math.pi*(area/(perimeter*perimeter))
    return circularity
