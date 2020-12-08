from typing import List, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
import cv2.cv2
import sklearn.decomposition as dec
import scipy
import logging
from sensorData import SensorData

FORMAT = '[%(asctime)s] [%(levelname)s] [%(funcName)s] [%(lineno)d] : %(message)s'
logging.basicConfig(format=FORMAT, level = logging.INFO)

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
        #print("[INFO] Object found. Saving locally.")
        cv2.imwrite('faces_detected.jpg', roi_color)
        roi_color_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        plt.imshow(roi_color_rgb)
        plt.show()
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
    #print('Read a new frame: ', success)
    return success, image


# TODO: need to fixed the tittle bug
def plot_results(greens, reds, blues, title=""):
    """
    Plots the results
    :param title: Title of the results
    :param greens: array containing green channel values
    :param reds: array containing red channel values
    :param blues: array containing blue channel values
    """
    plt.title(title)
    plt.subplot(3, 1, 1)
    plt.plot(greens, "green")
    plt.subplot(3, 1, 2)
    plt.plot(reds, "red")
    plt.subplot(3, 1, 3)
    plt.plot(blues, "blue")
    plt.show()


# A failed attempt to run fft :(
# def apply_fft(second_component):
#     n = 90
#     fhat = np.fft.fft(second_component, n)
#     PSD = fhat * (np.conj(fhat) / n)
#     freq = 3 * np.arange(n)
#     L = np.arange(1, np.floor(n / 2), dtype='int')
#     indices = PSD > 100
#     PSDclean = PSD * indices
#     fhat = indices * fhat
#     ffilt = np.fft.ifft(fhat)
#     plt.plot(ffilt, "pink")
#     plt.show()
#     # x_fft = scipy.fft.fft(second_component)
#     # plt.plot(x_fft)
#     # FIR

# A failed attempt to run signal filter :(
# def apply_FIR_filter(signal, fps=30, filter_size=2, min_f=0.75, max_f=3.3):
#     import scipy.signal
#     FIR_filter = scipy.signal.firwin(numtaps=filter_size * fps + 1, cutoff=[min_f * 2 / fps, max_f * 2 / fps],
#                                      window='hamming', pass_zero=True)
#     filtered_signal = np.convolve(signal, FIR_filter, mode='valid')
#     return filtered_signal


def extract_hr(greens, reds, blues):
    logging.info("Extracting HR ...")
    perform_ica(greens, reds, blues)


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


# This part convert the video to images, every image is a frame
def main():
    """
    :return:
    """
    dataset_location = "../dataset/good_sync/"
    # dataset_location = "..\\dataset\\good_sync\\"
    dir = "perry-all-2"
    sd = SensorData(dataset_location + dir)
    logging.info("Starting ...")
    # video_location = dataset_location + "\\" + dir + "\\" + sd.get_video_filename()
    video_location = dataset_location + "/" + dir + "/" + sd.get_video_filename()
    logging.info("Working on video " + video_location)
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    num_of_frames = 1
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    greens = []
    blues = []
    reds = []

    logging.info("Parsing images ...")
    while success:
        parse_roi(image)  # build image ROI
        image = cv2.imread("faces_detected.jpg")
        success, image = parse_RGB(image, vidcap, greens, blues, reds)
        num_of_frames += 1

    logging.info("Plotting results ...")
    plot_results(greens, reds, blues, "The first results")
    extract_hr(greens, reds, blues)  # didnt understand that warning.
    logging.info("Done")

if __name__ == "__main__":
    main()
