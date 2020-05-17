import cv2
import numpy as np
import matplotlib.pyplot as plt

# to make the image equalization
def histogram_eq(img,gray_level_sum):

    count = np.zeros(gray_level_sum)

    for row in img:
        for pixel in row:
            count[pixel] += 1
    row,col = img.shape
    prob = count / (row * col)  # calculate the probability of each pixel
    prob_heap = np.cumsum(prob) # calculate the heap probability
    # map function
    img_map = [int((gray_level_sum - 1) * prob_heap[i]) for i in range(gray_level_sum)]

    for i in range(row):
        for j in range(col):
            img[i, j] = img_map[img[i, j]]

    return img




if __name__ == '__main__':
    gray_level_sum = 256    # gray level sum
    fig,axes = plt.subplots(2,2)

    img = cv2.imread('montain.jpg',0)
    # print(img.shape)

    axes[0,0].set_title('Raw_Image')
    axes[0,0].imshow(img,'gray')
    axes[0,0].set_xticks([]), axes[0,0].set_yticks([])

    axes[0,1].set_title('Raw_Histogram')
    axes[0,1].hist(img.reshape([img.size]),256)

    img_his_eq = histogram_eq(img,gray_level_sum)
    axes[1, 0].set_title('Histogram_Image')
    axes[1, 0].imshow(img_his_eq, 'gray')
    axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])

    axes[1, 1].set_title('Histogram_equalization')
    axes[1, 1].hist(img_his_eq.reshape([img_his_eq.size]), 256)

    plt.show()