import cv2
import numpy as np
import math


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

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


# возвращает вложенный массив
# [[((x,y),(w,h),angle)),((x,y),(w,h),angle)),...],[((x,y),(w,h),angle)),((x,y),(w,h),angle)),...],]
def сортировка_контуров(контуры):
    контуры2 = контуры.copy()
    таблицаЯчеек = []
    # print(len(контуры2))
    # input()
    while len(контуры2) > 0:
        t = контуры2[0]  # первый попавшийся
        y = контуры2[0][0][1]  # смотрим кординату y
        строкаКонтуров = []  # здесь будут контрура принадлежащие одной строке
        for _ in контуры2[:]:
            # print(i)
            # input()
            nextY = _[0][1]
            if y + 5 > nextY > y - 5:
                y = nextY
                строкаКонтуров.append(_)
                контуры2.remove(_)
        таблицаЯчеек.append(строкаКонтуров)
    return таблицаЯчеек


def показать_строку(таблицаТочек, n):
    строка = таблицаТочек[n - 1]

    maxY = 0
    for _ in строка:
        for __ in _:
            if __[1] > maxY:
                maxY = __[1]
    minY = строка[0][0][1]
    for _ in строка:
        for __ in _:
            if __[1] < minY:
                minY = __[1]

    maxX = 0
    for _ in строка:
        for __ in _:
            if __[0] > maxX:
                maxX = __[0]
    minX = строка[0][0][0]
    for _ in строка:
        for __ in _:
            if __[0] < minX:
                minX = __[0]

    # print(maxY,minY,maxX,minX)
    # input()

    crop_img = rotimg[int(minY):int(maxY), int(minX):int(maxX)]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)


image_file = "3.5.png"
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()

cnt = contours[1]
ell = cv2.fitEllipse(cnt)
angle = ell[2]
# print(angle)
rotimg = rotate_image(img, angle - 90)
cv2.imshow("rotate", rotimg)
roterode = rotate_image(img_erode, angle - 90)

contours, hierarchy = cv2.findContours(roterode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

n = 0
ячейки = []
for idx, contour in enumerate(contours):
    # (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 1:
        n += 1
        rect = cv2.minAreaRect(contour)  # выход:((x,y),(w,h),angle))
        # print(contour[0])
        # input()
        ячейки.append(rect)
        # print(rect)
        # input()
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print(box)
        # input()
        cv2.drawContours(rotimg, [box], 0, (250, 0, 0), 1)

таблица = сортировка_контуров(ячейки)
print(таблица[0])
print("-----")

таблицаТочек = []
for _ in таблица:
    строка = []
    for __ in _:
        # print(__)
        # input()
        строка.append(cv2.boxPoints(__))
    таблицаТочек.append(строка)

print(таблицаТочек[0])
for _ in таблицаТочек:
    for __ in _:
        for ___ in __:
            # print(___)
            # input()
            cv2.circle(rotimg, (int(___[0]), int(___[1])), 2, (0, 0, 255), 2)

показать_строку(таблицаТочек, 6)

n = 1
d = 1
for _ in таблица:
    for __ in _:
        # cv2.circle(rotimg,(int(__[0][0]),int(__[0][1])) , 2, (0,0,255), 2)
        cv2.putText(rotimg, str(n), (int(__[0][0]), int(__[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    cv2.LINE_AA)
        n += 1
    d += 1
# print(n)


cv2.imshow("Input", img)
cv2.imshow("gray", gray)
cv2.imshow("thresh", thresh)
cv2.imshow("Enlarged", img_erode)
cv2.imshow("rotimg", rotimg)
cv2.waitKey(0)