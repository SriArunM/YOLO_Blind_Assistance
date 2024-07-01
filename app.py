from flask import Flask, render_template, Response

app = Flask(__name__)

import cv2 as cv
import numpy as np
import pyttsx3

# Distance constants
KNOWN_DISTANCE = 40  # INCHES
BOTTLE_KNOWN_DISTANCE = 27.5
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
BOTTLE_WIDTH = 3
LAPTOP_WIDTH = 11.77
CHAIR_WIDTH = 18.50
CAT_WIDTH = 4.3
# DOG_WIDTH=7.5
TVMONITOR_WIDTH = 35
# BUS_WIDTH=100.39
# FIRE_HYDRANT_WIDTH=12
BOOK_WIDTH = 7.5
# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
# colors for object detected
COLORS = [
    (255, 0, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 0),
]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for classid, score, box in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s:%f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 39:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 56:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 63:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 15:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])
        elif classid == 62:
            data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])

        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


# reading the reference image from dir
ref_person = cv.imread("ReferenceImages/image1.png")
ref_mobile = cv.imread("ReferenceImages/image2.png")
ref_bottle = cv.imread("ReferenceImages/image3.png")
ref_laptop = cv.imread("ReferenceImages/image4.png")
ref_chair = cv.imread("ReferenceImages/image5.png")
ref_cat = cv.imread("ReferenceImages/image6.jpg")
ref_tvmonitor = cv.imread("ReferenceImages/image7.jpg")

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

bottle_data = object_detector(ref_bottle)
bottle_width_in_rf = bottle_data[0][1]

laptop_data = object_detector(ref_laptop)
laptop_width_in_rf = laptop_data[0][1]

chair_data = object_detector(ref_chair)
chair_width_in_rf = chair_data[0][1]

cat_data = object_detector(ref_cat)
cat_width_in_rf = cat_data[0][1]

tvmonitor_data = object_detector(ref_tvmonitor)
tvmonitor_width_in_rf = tvmonitor_data[0][1]

print(
    f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf} bottle width in pixel: {bottle_width_in_rf} laptop width in pixels : {laptop_width_in_rf} chair width in pixels : {chair_width_in_rf}"
)

# finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_width_in_rf)
focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)
focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
focal_cat = focal_length_finder(KNOWN_DISTANCE, CAT_WIDTH, cat_width_in_rf)
focal_tvmonitor = focal_length_finder(
    KNOWN_DISTANCE, TVMONITOR_WIDTH, tvmonitor_width_in_rf
)

cap = cv.VideoCapture(0)


# speak
def speak(distance, obj_name):
    engine = pyttsx3.init()
    engine.say(
        "The object "
        + obj_name
        + " is "
        + str(int(distance))
        + " inches in front of you"
    )
    engine.runAndWait()


"""def gen(camera):
    while True:
        frame = camera.main()
        if frame != "":
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')"""


def gen_frames():
    while True:
        ret, frame = cap.read()
        x, y, distance = 0, 0, 0
        data = object_detector(frame)
        for d in data:
            if d[0] == "person":
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == "cell phone":
                distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == "bottle":
                distance = distance_finder(focal_bottle, BOTTLE_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == "chair":
                distance = distance_finder(focal_chair, CHAIR_WIDTH, d[1])
                x, y = d[2]
            elif d[0] == "laptop":
                distance = distance_finder(focal_laptop, LAPTOP_WIDTH, d[1])
                x, y = d[2]

            elif d[0] == "tvmonitor":
                distance = distance_finder(focal_tvmonitor, TVMONITOR_WIDTH, d[1])
                x, y = d[2]

            if distance < 10000:
                speak(distance, d[0])
                pass
            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(
                frame,
                f"Dis: {round(distance,2)} inch",
                (x + 5, y + 13),
                FONTS,
                0.48,
                GREEN,
                2,
            )

        ret, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        key = cv.waitKey(1)
        if key == ord("q"):
            break


# cap.release()
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
