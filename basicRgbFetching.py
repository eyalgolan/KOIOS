import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2

def parse_roi(image):
    """
    Upon receiving an image, finds a face (if exists) and writes it as an image
    :param image: the image to be parsed
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # perform grayscale

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        cv2.imwrite('faces_detected.jpg', roi_color)

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
    red, green, blue = cv2.split(image)
    greens.append(np.mean(green))
    blues.append(np.mean(blue))
    reds.append(np.mean(red))
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    return success, image

def plot_results(greens, reds, blues):
    """
    Plots the results
    :param greens: array containing green channel values
    :param reds: array containing red channel values
    :param blues: array containing blue channel values
    """
    plt.subplot(3, 1, 1)
    plt.plot(greens, "green")
    plt.subplot(3, 1, 2)
    plt.plot(reds, "red")
    plt.subplot(3, 1, 3)
    plt.plot(blues, "blue")
    plt.show()

# This part convert the video to images, every image is a frame
def main():
    """
    :return:
    """
    vidcap = cv2.VideoCapture('test.mp4')
    success, image = vidcap.read()
    num_of_frames = 1
    greens = []
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    blues = []
    reds = []

    while success:
        parse_roi(image) # build image ROI
        image = cv2.imread("faces_detected.jpg")
        success,image = parse_RGB(image, vidcap, greens, blues, reds)
        num_of_frames += 1

    plot_results(greens, reds, blues)

if __name__ == "__main__":
    main()