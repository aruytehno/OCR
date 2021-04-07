import os
import cv2
import numpy as np



# Возвращает повёрнутое изображение (матрицу) на угол "angle"
# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def rotate_image(image):
    gray = cv2.bitwise_not(cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY))
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    print("[INFO] angle: {:.3f}".format(angle))
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# сортирует контруры, возвращает вложенный массив, табличное представление
# [[((x,y),(w,h),angle)),((x,y),(w,h),angle)),...],[((x,y),(w,h),angle)),((x,y),(w,h),angle)),...],]
def sorting_contours(contours):
    contours2 = contours.copy()
    cell_table = []
    while len(contours2) > 0:
        t = contours2[0]  # первый попавшийся
        y = contours2[0][0][1]  # смотрим кординату y
        line_outlines = []  # здесь будут контрура принадлежащие одной строке
        for _ in contours2[:]:
            nextY = _[0][1]
            if y + 5 > nextY > y - 5:
                y = nextY
                line_outlines.append(_)
                contours2.remove(_)
        cell_table.append(line_outlines)
    return cell_table


def show_row(point_table, n):
    row = point_table[n - 1]

    maxY = 0
    for _ in row:
        for __ in _:
            if __[1] > maxY:
                maxY = __[1]
    minY = row[0][0][1]
    for _ in row:
        for __ in _:
            if __[1] < minY:
                minY = __[1]

    maxX = 0

    for _ in row:
        for __ in _:
            if __[0] > maxX:
                maxX = __[0]
    minX = row[0][0][0]

    for _ in row:
        for __ in _:
            if __[0] < minX:
                minX = __[0]

    return rotated_img[int(minY):int(maxY), int(minX):int(maxX)]


def cells(contours):
    # здесь преобразование беспорядочных точек контуров в отсортированые 4 точки углов каждой ячейки:
    n = 0
    cells = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == 1:
            n += 1
            rect = cv2.minAreaRect(contour)  # выход:((x,y),(w,h),angle))
            cells.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # show color contours
            # cv2.drawContours(rotated_img, [box], 0, (0, 0, 255), 1)
    return cells

def places_intersection_points(point_table):
    # Расставляет точки пересечения
    for _ in point_table:
        for __ in _:
            for ___ in __:
                cv2.circle(rotated_img, (int(___[0]), int(___[1])), 2, (255, 0, 255),
                           2)  # цвет точек пересечения (феолет.)
    return point_table

def point_in_table(table):
    point_table = []
    for _ in table:
        row = []
        for __ in _:
            row.append(cv2.boxPoints(__))
        point_table.append(row)
    return point_table


def show_cell_numbers():
    # отображение номеров ячеек
    n = 1
    d = 1
    for _ in table:
        for __ in _:
            cv2.putText(rotated_img, str(n), (int(__[0][0]), int(__[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                        1,
                        cv2.LINE_AA)  # цвет нумерации ячеек (синий)
            n += 1
        d += 1
    return table



"""
начало работы
"""
# чтение изображение в img
image_orig = cv2.imread('examples' + os.sep + 'rotated' + os.sep + 'example7.png')
# автоповорот изображения
rotated_img = rotate_image(image_orig)
cv2.imwrite('out' + os.sep + 'Input.png', rotated_img)
cv2.imwrite('out' + os.sep + 'Input_orig.png', image_orig)
# конвертирование в оттенки серого результат в "gray"
gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
# пороговое осветление (все пиксели ярче 200 становятся белыми (255)остальные чёрными (0)) результат в "trash"
# первое число очень важно - от его выбора зависит как определятся контуры
ret, thresh = cv2.threshold(gray, 200, 255, 0, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8),
                      iterations=1)  # эрозия)) - каждый чёрный пиксель закрашивает соседние вокруг себя
# поиск контуров в "contours" изображение контуров, а в "hierarhy" иерархия (инф о вложенности контуров)
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# копирование входного изображение чтобы потом наложить на него контуры
output = rotated_img.copy()
cnt = contours[1]  # самый большой контур копировать в cnt

# поиск контуров (contours - координаты точек контуров, hierarchy - иерархия)
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

table = sorting_contours(cells(contours))


point_table = point_in_table(table)

# Функция показывает точки пересечения
# places_intersection_points(point_table)
# Функция показывает номера ячеек
# show_cell_numbers()

# Отображает строку по номеру
cv2.imwrite('out' + os.sep + 'cropped.png', show_row(point_table, 6))

