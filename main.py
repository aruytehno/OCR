import cv2
import pytesseract
import math

# https://itproger.com/news/raspoznavanie-teksta-s-kartinki-python-tesseract-orc-opencv
# Путь для подключения tesseract
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

# Подключение фото
img = cv2.imread('example.png')

# img = rotate_image(img, -8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(cv2.minAreaRect())
# Будет выведен весь текст с картинки
config = r'--oem 3 --psm 4'
# print(pytesseract.image_to_string(img, lang='rus', config=config))

# Делаем нечто более крутое!!!

data = pytesseract.image_to_data(img, lang='rus', config=config)

# Перебираем данные про текстовые надписи
for i, el in enumerate(data.splitlines()):
    if i == 0:
        continue

    el = el.split()
    try:
        # Создаем подписи на картинке
        # print(el)
        x, y, w, h = int(el[6]), int(el[7]), int(el[8]), int(el[9])
        # cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
        cv2.putText(img, el[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        print(cv2.FONT_HERSHEY_COMPLEX)
    except IndexError:
        print("Операция была пропущена")

cv2.imwrite('out/out.png', img)

