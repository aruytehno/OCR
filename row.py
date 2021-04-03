import cv2
import numpy as np
import math


# Возвращает повёрнутое изображение (матрицу) на угол "angle"
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


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

    crop_img = rotate_img[int(minY):int(maxY), int(minX):int(maxX)]
    cv2.imwrite('out/cropped.png', crop_img)


# начало работы
# чтение изображение в img
img = cv2.imread("example6.jpg")
# конвертирование в оттенки серого результат в "gray"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# пороговое осветление (все пиксели ярче 20 становятся белыми (255)) результат в "trash"
ret, thresh = cv2.threshold(gray, 20, 255,
                            cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8),
                      iterations=1)  # эрозия)) - каждый чёрный пиксель закрашивает соседние вокруг себя

# поиск контуров в "contours" изображение контуров, а в "hierarhy" иерархия (инф о вложенности контуров)
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# копирование входного изображение чтобы потом наложить на него контуры
output = img.copy()

cnt = contours[1] # самый большой контур копировать в cnt
ell = cv2.fitEllipse(cnt) # нахождение эллипса из контура
angle = ell[2] # угол поворота этого эллипса
rotate_img = rotate_image(img, angle - 90) # поворачивает изначсальное изображение на угол поворота главного контура
cv2.imshow("rotate", rotate_img) # показывает его
roterode = rotate_image(img_erode, angle - 90) # поворачивает изображение (img_erode) которое после преобразования

# поиск контуров (contours - координаты точек контуров, hierarchy - иерархия)
contours, hierarchy = cv2.findContours(roterode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        cv2.drawContours(rotate_img, [box], 0, (0, 0, 255), 1)  # цвет определённых контуров (красный)

table = sorting_contours(cells)

point_table = []
for _ in table:
    row = []
    for __ in _:
        row.append(cv2.boxPoints(__))
    point_table.append(row)

# Расставляет точеки пересечения
for _ in point_table:
    for __ in _:
        for ___ in __:
            cv2.circle(rotate_img, (int(___[0]), int(___[1])), 2, (255, 0, 255), 2)  # цвет точек пересечения (феолет.)

print(point_table.__sizeof__())

show_row(point_table, 4) # отображает строку по номеру

# отображение номеров ячеек
n = 1
d = 1
for _ in table:
    for __ in _:
        cv2.putText(rotate_img, str(n), (int(__[0][0]), int(__[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)  # цвет нумерации ячеек (синий)
        n += 1
    d += 1
# сохранение всех изображений
cv2.imwrite('out/Input.png', img)
# cv2.imwrite('out/gray.png', gray)
# cv2.imwrite('out/thresh.png', thresh)
# cv2.imwrite('out/Enlarged.png', img_erode)
cv2.imwrite('out/rotimg.png', rotate_img)
