import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_pixel_prob(img,gray_level_sum):

    count = np.zeros(gray_level_sum)

    for row in img:
        for pixel in row:
            count[pixel] += 1

    row, col = img.shape
    prob = count / (row * col)

    return prob


def his_eq(img, prob,gray_level_sum):

    prob_heap = np.cumsum(prob)  # 累计概率

    img_map = [int((gray_level_sum-1)* prob_heap[i]) for i in range(gray_level_sum)]  # 像素值映射

    # 像素值替换
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            img[i, j] = img_map[img[i, j]]

    return img


# def plot(y, name):
#     """
#     画直方图，len(y)==gray_level
#     :param y: 概率值
#     :param name:
#     :return:
#     """
#     plt.figure(num=name)
#     plt.bar([i for i in range(gray_level)], y, width=1)


if __name__ == '__main__':
    gray_level_sum = 256  # 灰度级总数
    fig,axes = plt.subplots(2,2)

    img = cv2.imread("montain.jpg", 0)  # 读取灰度图
    print(img.shape)

    axes[0,0].set_title('Raw_Image')
    axes[0,0].imshow(img,'gray')
    axes[0,0].set_xticks([]), axes[0,0].set_yticks([])

    axes[0,1].set_title('Raw_Histogram')
    # axes[0,1].hist(prob,orientation='horizontal')
    axes[0,1].hist(img.reshape([img.size]),256)


    prob = get_pixel_prob(img,gray_level_sum)
    # plot(prob, "原图直方图")
    # print(prob)
    # 直方图均衡化
    img_his_eq = his_eq(img, prob,gray_level_sum)
    print(img_his_eq)

    # cv2.imwrite("source_hist.jpg", img)  # 保存图像

    prob_his_eq = get_pixel_prob(img_his_eq,gray_level_sum)
    # plot(prob, "直方图均衡化结果")

    axes[1, 0].set_title('Histogram_Image')
    axes[1, 0].imshow(img_his_eq, 'gray')
    axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])

    axes[1, 1].set_title('Histogram_equalization')
    # axes[0,1].hist(prob,orientation='horizontal')
    axes[1, 1].hist(img_his_eq.reshape([img_his_eq.size]), 256)



    plt.show()
