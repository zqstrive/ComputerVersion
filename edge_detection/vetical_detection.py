import cv2
import numpy as np

# Gradient filter
def gradient_filter(img,Kernel, K_size=3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)


    tmp = out.copy()

    # filtering
    for row in range(H):
        for col in range(W):
            for c in range(C):
                out[pad + row, pad + col, c] = np.sum(Kernel * tmp[row: row + K_size, col: col + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out


# Read image
img = cv2.imread("lena.jpg",cv2.IMREAD_GRAYSCALE)

# vertical_filter
vertical_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float)
vertical_kernel /= 3

# horizontal_filter
horizontal_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],dtype=np.float)
horizontal_kernel /= 3

# Gradient Filter
out_vertical = gradient_filter(img,vertical_kernel,K_size=3)
out_horizontal = gradient_filter(img,horizontal_kernel,K_size=3)

# Save result
cv2.imwrite('out_vertical.jpg',out_vertical)
cv2.imwrite('out_horizontal.jpg',out_horizontal)
# cv2.imwrite("out.jpg", out)
# cv2.imshow("result",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
