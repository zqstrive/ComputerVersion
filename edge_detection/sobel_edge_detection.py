import cv2
import numpy as np


def sobel(img):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    row, col = img.shape
    out = np.zeros(img.shape)
    out_x = np.zeros(img.shape)
    out_y = np.zeros(img.shape)

    for i in range(row - 2):
        for j in range(col - 2):
            out_x[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * sobel_x))
            out_y[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * sobel_y))
            out[i + 1, j + 1] = (out_x[i + 1, j + 1] ** 2 + out_y[i + 1, j + 1] ** 2) ** 0.5

    out = np.clip(out, 0, 255)
    return np.uint8(out)


img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('img',img)
sobel_out = sobel(img)
cv2.imwrite('sobel_img.jpg',sobel_out)
cv2.imshow('sobel_img', sobel_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
