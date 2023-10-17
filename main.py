import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure


def compute_histogram(image):
    histogram = np.zeros(256)
    for pixel in image.ravel():
        histogram[pixel] += 1
    return histogram


def match_histograms(original, reference):
    orig_cumul = np.cumsum(original) / original.sum()
    ref_cumul = np.cumsum(reference) / reference.sum()

    # lut = np.interp(orig_cumul, ref_cumul, np.arange(256))
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        # Find the index where the difference between the input CDF and the desired CDF is the smallest.
        mapping[i] = np.argmin(np.abs(orig_cumul[i] - ref_cumul))

    return mapping.astype(np.uint8)


def method_1(input_image, reference):
    # Match the input image to the desired histogram
    specified_image = exposure.match_histograms(input_image, reference)
    return specified_image


def method_2(input_image, reference):
    hist_input = compute_histogram(input_image)
    hist_ref = compute_histogram(reference)

    # Look up table
    lut = match_histograms(hist_input, hist_ref)
    matched_img = np.reshape(lut[input_image.ravel()], input_image.shape)
    return matched_img


if __name__ == '__main__':
    # Data type
    # brain : CT images of brain
    # human : Different human faces
    data_type = 'human'
    data_path = os.path.join('data', data_type)

    # Load images
    input_image = Image.open(os.path.join(data_path, 'original.png')).convert('L')
    input_image = np.asarray(input_image)
    reference = Image.open(os.path.join(data_path, 'reference.png')).convert('L')
    reference = np.asarray(reference)

    # Select method
    # method == 1 : Using a Scikit-image library
    # method == 2 : Traditional method
    method = 2

    # Run
    if method == 1:
        specified_image = method_1(input_image, reference)

    elif method == 2:
        specified_image = method_2(input_image, reference)

    # Visualize
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title("Input Image")
    plt.subplot(2, 3, 2)
    plt.imshow(reference, cmap='gray')
    plt.title("Reference Image")
    plt.subplot(2, 3, 3)
    plt.imshow(specified_image, cmap='gray')
    plt.title("Specified Image")
    plt.subplot(2, 3, 4)
    plt.hist(input_image.ravel(), 256, [0, 256])
    plt.subplot(2, 3, 5)
    plt.hist(reference.ravel(), 256, [0, 256])
    plt.subplot(2, 3, 6)
    plt.hist(specified_image.ravel(), 256, [0, 256])
    plt.show()


