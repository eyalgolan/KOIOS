from typing import List, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
import cv2.cv2
import sklearn.decomposition as dec
import scipy
import logging
import scipy.signal
from sensorData import SensorData
import platform
import imutils  # why do we need this?

FORMAT = '[%(asctime)s] [%(levelname)s] [%(funcName)s] [%(lineno)d] : %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def parse_roi(image):
    """
    Upon receiving an image, finds a face (if exists) and writes it as an image
    :param image: the image to be parsed
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # perform grayscale
    flag_face_detected = False
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in face:
        flag_face_detected = True
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        # print("[INFO] Object found. Saving locally.")
        cv2.imwrite('faces_detected.jpg', roi_color)
        roi_color_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        plt.imshow(roi_color_rgb)
        # plt.show()
    if not flag_face_detected:
        logging.warning("No face detected in image")


def parse_RGB(image, vidcap, greens, blues, reds):
    """
    Parses an image to its RGB channels
    :param image: the image to be parsed
    :param vidcap:
    :param greens: array containing green channel values
    :param blues: array containing blue channel values
    :param reds: array containing red channel values
    :return: a flag indicating if there is a next image, and the next image
    """
    blue, green, red = cv2.split(image)
    greens.append(np.mean(green))
    blues.append(np.mean(blue))
    reds.append(np.mean(red))
    success, image = vidcap.read()
    # print('Read a new frame: ', success)
    return success, image


def plot_results(greens, reds, blues, title=""):
    """
    Plot results
    :param title: Title of the results
    :param greens: array containing green channel values
    :param reds: array containing red channel values
    :param blues: array containing blue channel values
    """

    plt.figure(figsize=(6, 5))
    plt.title(title)
    plt.subplot(3, 1, 1)
    plt.plot(greens, "green")
    plt.subplot(3, 1, 2)
    plt.plot(reds, "red")
    plt.subplot(3, 1, 3)
    plt.plot(blues, "blue")
    plt.show()


def apply_fir(signal, fps=30, filter_size=2, min_f=0.75, max_f=3.3):
    '''
        This function filter a signal.
    :param signal: the sig to be filter
    :param fps: fps of the video
    :param filter_size: size of the filter
    :param min_f: minimum frequency  to filter. 0.75 means 45 bmp
    :param max_f: Max frequency to filter. 0.3.3 means 200 bmp
    :return: filtered signal.
    '''
    logging.info("Run FIR filter on green signal")
    FIR_filter = scipy.signal.firwin(numtaps=filter_size * fps + 1, cutoff=[min_f * 2 / fps, max_f * 2 / fps],
                                     window='hamming', pass_zero=False) # runs scipy FIR filter method with some params from the web.
    filtered_signal = np.convolve(signal, FIR_filter, mode='valid')
    return filtered_signal


def extract_hr(greens, reds, blues):
    '''
    This function is THE function that runs the method to extract the heart beat.
    :param greens: green signal
    :param reds: red signal
    :param blues: blue signal
    :return: For now nothing, should return the bbp of the person in the video for every 15 seconds window.
    '''
    logging.info("Extracting HR ...")
    rgb_after_ica = perform_ica(greens, reds, blues)
    filtered_green_sig = apply_fir(rgb_after_ica[1, :])
    # TODO: run fft on the filtered sig in every 15 second window and calc the heart rate.
    # A sanity check of the filtering:
    plt.figure(figsize=(6, 5))
    plt.title("filtered_green_sig")
    plt.subplot(2, 1, 1)
    plt.plot(rgb_after_ica[1, :][:200])
    plt.subplot(2, 1, 2)
    plt.plot(filtered_green_sig[:200])
    plt.show()


def standardize(lst):
    """
    This function get a list of values, makes an array and standardize it.
    :param lst: List of values to be standardize.
    :return: An standardized array out of the list.
    """
    logging.info("Stabilizing results ...")
    mean_val = np.mean(lst)
    std = np.std(lst)
    standardized_arr = (lst - mean_val) / std
    return standardized_arr


def perform_ica(greens, reds, blues):
    """
    This function performs ICA on the signals.
    :param greens: Green signal.
    :param reds: Red signal.
    :param blues: Blue signal value.
    :return: --
    """
    logging.info("Performing ICA ...")
    # ICA process assume that all the values are normalized!.
    normalized_green = standardize(greens)
    normalized_red = standardize(reds)
    normalized_blue = standardize(blues)
    rgb_mat = np.vstack((normalized_red, normalized_green, normalized_blue)).T  # builds a matrix from the signals.
    ica_ = dec.FastICA()
    s = ica_.fit_transform(rgb_mat).T  # s is a matrix of the original signals, each row is a component.
    plot_results(s[1, :], s[0, :], s[2, :], "After ICA")
    return s


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


# This part convert the video to images, every image is a frame
def main():
    """
    :return:
    """
    logging.info("Starting ...")

    if platform.system() == "Windows":
        seperator = "\\"
    else:
        seperator = "/"

    dataset_location = ".." + seperator + "dataset" + seperator + "good_sync" + seperator
    dir = "perry-all-2"
    logging.info("Obtaining collected data ...")
    sd = SensorData(dataset_location + dir)
    # video_location = dataset_location + seperator + dir + seperator + sd.get_video_filename()
    video_location = dataset_location + seperator + dir + seperator + "ptest.mp4"
    logging.info("Working on video " + video_location)
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    # image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    num_of_frames = 1
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    greens = []
    blues = []
    reds = []

    logging.info("Parsing images ...")
    while success:
        image = rotate_image(image, 90)
        parse_roi(image)  # build image ROI
        image = cv2.imread("faces_detected.jpg")
        success, image = parse_RGB(image, vidcap, greens, blues, reds)
        # image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imshow("Rotated (Correct)", image)
        num_of_frames += 1

    logging.info("Plotting results ...")
    plot_results(greens, reds, blues, "The first results")
    extract_hr(greens, reds, blues)  # didnt understand that warning.
    logging.info("Done")


if __name__ == "__main__":
    main()
