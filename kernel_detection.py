import numpy as np
import cv2
import argparse


def matrix_summ(img_matrix: np.ndarray) -> int:
    s = 0
    for row in range(img_matrix.shape[0]):
        for col in range(img_matrix.shape[1]):
            s = s + img_matrix[row][col]
    return s


def kernel_detector(kernel_size: int, image_height: int, image_width:int, img : np.ndarray,
                    img_copy: np.ndarray, blobs_list: list) -> None:
    '''

    :param kernel_size: size of the applied kernel, single integer number, i.e. 3
    :param image_height: height of the input image
    :param image_width: width of the input image
    :param img: input black and white image(after thresholding operation) in the form of numpy array
    :param img_copy: colored image(converted from grayscale to BGR domain) to draw detected blobs
    :param blobs_list: a list of coordinates for detectded blobs, will be used for drawing blobs further
    :return:
    '''
    if kernel_size == 3:
        for i in range(image_height):
            for j in range(image_width):
                if img[i, j] == 255:
                    img_slice = img[i - 1:i + 2, j - 1:j + 2]
                    if ((matrix_summ(img_slice) - 255) == 0):
                        img_copy[i, j] = (0, 0, 0)
                        blobs_list.extend([j, i])

    if kernel_size == 5:
        for i in range(image_height):
            for j in range(image_width):
                if img[i, j] == 255:
                    img_slice = img[i - 2:i + 3, j - 2:j + 3]
                    if ((matrix_summ(img_slice) - 255) == 0):
                        img_copy[i, j] = (0, 0, 0)
                        blobs_list.extend([j, i])
    elif kernel_size == 7:
        for i in range(image_height):
            for j in range(image_width):
                if img[i, j] == 255:
                    img_slice = img[i - 3:i + 4, j - 3:j + 4]
                    if ((matrix_summ(img_slice) - 255) == 0):
                        img_copy[i, j] = (0, 0, 0)
                        blobs_list.extend([j, i])

    elif kernel_size == 13:
        for i in range(image_height - 1):
            for j in range(image_width - 1):
                if img[i][j] == 255:
                    img_slice = img[i - 6:i + 7, j - 6:j + 7]
                    if ((matrix_summ(img_slice) - 255) <= 255):
                        img_copy[i, j] = (0, 0, 0)
                        blobs_list.extend([j, i])

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Input and output file paths')

    parser.add_argument('input',type = str, help = 'Path for input file')
    parser.add_argument('--save_result', help = 'Save output result', action = 'store_true')
    parser.add_argument('--output',type = str, help = 'Path for output file')

    args = parser.parse_args()
    print(args)

    #sample = cv2.imread('NC/S07_M0044-01MS.tif')
    sample = cv2.imread(args.input)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    sample = cv2.bilateralFilter(sample, 0, 10, 10)

    thresh = 190
    maxValue = 255
    ret, img = cv2.threshold(sample, thresh, maxValue, cv2.THRESH_BINARY)

    print(f'Image size:{img.shape}')

    height = img.shape[0]
    width = img.shape[1]
    blobs_list = []

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image_copy = color.copy()

    cv2.imshow("THRESHOLDING", img)
    cv2.waitKey(0)


    kernel_detector(kernel_size = 7, image_height = height, image_width = width, img=img, img_copy= image_copy, blobs_list=blobs_list)
    cv2.imshow("Removing blobs with 7x7 filter", image_copy)
    cv2.waitKey(0)

    kernel_detector(kernel_size = 13, image_height = height, image_width = width, img=img, img_copy = image_copy, blobs_list= blobs_list)
    cv2.imshow("Removing blobs with with 13x13 filter", image_copy)
    cv2.waitKey(0)

    if args.save_result:
        cv2.imwrite(args.output, image_copy)


    array = np.array(blobs_list, dtype = int)
    res = array.reshape(int(len(blobs_list)/2), 2)
    print("Количество выбросов:",res.shape[0])

    for k in range(int((res.size)/2)):
        center = (res[k][0],res[k][1])
        image_copy = cv2.circle(image_copy, center, 3,(0,0,255),-1)

    cv2.imshow("Highlighted blobs", image_copy)
    cv2.waitKey(0)




