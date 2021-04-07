# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
import os

import numpy as np
import cv2

# построить аргумент, синтаксический анализ и синтаксический анализ аргументов
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image file")
# args = vars(ap.parse_args())
# load the image from disk
image = cv2.imread('examples' + os.sep + 'rotated' + os.sep + 'example7.png')
# преобразовать изображение в оттенки серого и перевернуть передний план
# и фон, чтобы убедиться, что передний план теперь "белый" и фон "черный"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
# порог изображения, устанавливая для всех пикселей переднего плана значение
# 255 и все пиксели фона на 0
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# получаем координаты (x, y) всех значений пикселей, которые
# больше нуля, используйте эти координаты для
# вычислить повернутую ограничивающую рамку, которая содержит все координаты
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
# функция `cv2.minAreaRect` возвращает значения в
# диапазон [-90, 0); при вращении прямоугольника по часовой стрелке
# вернули угловые тренды на 0 - в этом особом случае мы
# нужно добавить 90 градусов к углу
if angle < -45:
    angle = -(90 + angle)
# в противном случае просто возьмите угол, обратный углу, чтобы сделать его положительным
else:
    angle = -angle
# повернуть изображение, чтобы выровнять его
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# рисуем угол коррекции на изображении, чтобы мы могли его проверить
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# сохраняем выходное изображение
print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("Input", image)
# cv2.imshow("Rotated", rotated)
cv2.imwrite('out' + os.sep + 'Input.png', image)
cv2.imwrite('out' + os.sep + 'Rotated.png', rotated)
# cv2.waitKey(0)
