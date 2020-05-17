import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    print ("histogram: ", histogram)
    # plt.hist(histogram,30,[0,256])

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    print ("uniform hist: ", uniform_hist)
    plt.hist(uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray


if __name__ == '__main__':
    fname = "montain.jpg" # Gray image
    # Note, matplotlib natively only read png images.
    gray = plt.imread(fname, format=np.uint8)
    if gray is None:
        print ("Image {} does not exist!".format(fname))
        exit(-1)

    # Histogram equalization
    uniform_gray = histogram_equalization(gray)

    # Display the result
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Raw Image")
    ax1.imshow(gray, 'gray')
    ax1.set_xticks([]), ax1.set_yticks([])

    ax2.set_title("Histogram Equalized Image")
    ax2.imshow(uniform_gray, 'gray')
    ax2.set_xticks([]), ax2.set_yticks([])

    fig.tight_layout()
    plt.show()