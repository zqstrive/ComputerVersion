import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_image():

    img = cv2.imread('image.bmp')
    img = img / 255
    return img


def initialize_means(img, clusters):

    points = np.reshape(img, (img.shape[0] * img.shape[1],
                              img.shape[2]))
    m, n = points.shape
    means = np.zeros((clusters, n))

    # random initialization of means.
    for i in range(clusters):
        rand_1 = int(np.random.random(1) * 10)
        rand_2 = int(np.random.random(1) * 8)
        means[i, 0] = points[rand_1, 0]
        means[i, 1] = points[rand_2, 1]

    return points, means

def compute_distance(x1, y1, x2, y2):
    distance = np.square(x1 - x2) + np.square(y1 - y2)
    distance = np.sqrt(distance)

    return distance


def k_means(points, means, clusters):
    iterations = 10  # the number of iterations
    m, n = points.shape

    # these are the index values that
    # correspond to the cluster to
    # which each pixel belongs to.
    index = np.zeros(m)

    # k-means algorithm.
    while (iterations > 0):

        for j in range(len(points)):

            # initialize minimum value to a large value
            min_value = 1000
            temp = None

            for k in range(clusters):

                x1 = points[j, 0]
                y1 = points[j, 1]
                x2 = means[k, 0]
                y2 = means[k, 1]

                if (compute_distance(x1, y1, x2, y2) < min_value):
                    min_value = compute_distance(x1, y1, x2, y2)
                    temp = k
                    index[j] = k

        for k in range(clusters):

            sumx = 0
            sumy = 0
            count = 0

            for j in range(len(points)):

                if (index[j] == k):
                    sumx += points[j, 0]
                    sumy += points[j, 1]
                    count += 1

            if (count == 0):
                count = 1

            means[k, 0] = float(sumx / count)
            means[k, 1] = float(sumy / count)

        iterations -= 1

    return means, index


def compress_image(means, index, img):
    # recovering the compressed image by
    # assigning each pixel to its corresponding centroid.
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]

    # getting back the 3d matrix (row, col, rgb(3))
    recovered = np.reshape(recovered, (img.shape[0], img.shape[1],
                                       img.shape[2]))

    # plotting the compressed image.
    plt.imshow(recovered)
    plt.show()

    # # saving the compressed image.
    # cv2.imwrite('image_seg.png', recovered)


# Driver Code
if __name__ == '__main__':
    img = read_image()

    clusters = 64

    points, means = initialize_means(img, clusters)
    means, index = k_means(points, means, clusters)
    compress_image(means, index, img)
