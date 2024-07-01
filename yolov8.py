from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np
import pyttsx3
from ultralytics import YOLO

app = Flask(__name__)

# Distance constants
KNOWN_DISTANCE = 40  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
BOTTLE_WIDTH = 3
LAPTOP_WIDTH = 11.77
CHAIR_WIDTH = 18.50
CAT_WIDTH = 4.3
TVMONITOR_WIDTH = 35

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

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 Nano model pre-trained on COCO


# Object detection function
def object_detector(image):
    results = model(image)
    data_list = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        names = result.names  # Class names

        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id = int(class_id)
            label = names[class_id]
            color = COLORS[class_id % len(COLORS)]

            # Draw rectangle and label
            cv.rectangle(
                image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2
            )
            cv.putText(
                image,
                f"{label}:{score:.2f}",
                (int(box[0]), int(box[1]) - 14),
                FONTS,
                0.5,
                color,
                2,
            )

            if label in [
                "person",
                "cell phone",
                "bottle",
                "chair",
                "laptop",
                "cat",
                "tvmonitor",
            ]:
                data_list.append(
                    [label, int(box[2] - box[0]), (int(box[0]), int(box[1]) - 2)]
                )

    return data_list


# Focal length and distance finder functions
def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width


def distance_finder(focal_length, real_object_width, width_in_frame):
    return (real_object_width * focal_length) / width_in_frame


# Reading reference images
ref_images = {
    "person": cv.imread("ReferenceImages/image1.png"),
    "mobile": cv.imread("ReferenceImages/image2.png"),
    "bottle": cv.imread("ReferenceImages/image3.png"),
    "laptop": cv.imread("ReferenceImages/image4.png"),
    "chair": cv.imread("ReferenceImages/image5.png"),
    "cat": cv.imread("ReferenceImages/image6.jpg"),
    "tvmonitor": cv.imread("ReferenceImages/image7.jpg"),
}

# Get width in reference frame for each object
ref_data = {key: object_detector(image)[0][1] for key, image in ref_images.items()}

# Finding focal lengths
focal_lengths = {
    "person": focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, ref_data["person"]),
    "mobile": focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, ref_data["mobile"]),
    "bottle": focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, ref_data["bottle"]),
    "laptop": focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, ref_data["laptop"]),
    "chair": focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, ref_data["chair"]),
    "cat": focal_length_finder(KNOWN_DISTANCE, CAT_WIDTH, ref_data["cat"]),
    "tvmonitor": focal_length_finder(
        KNOWN_DISTANCE, TVMONITOR_WIDTH, ref_data["tvmonitor"]
    ),
}

cap = cv.VideoCapture(0)


def speak(distance, obj_name):
    engine = pyttsx3.init()
    engine.say(f"The object {obj_name} is {int(distance)} inches in front of you")
    engine.runAndWait()


def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        data = object_detector(frame)
        for d in data:
            if d[0] in focal_lengths:
                distance = distance_finder(
                    focal_lengths[d[0]], globals()[f"{d[0].upper()}_WIDTH"], d[1]
                )
                x, y = d[2]
                if distance > 1000:
                    speak(distance, d[0])
                cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
                cv.putText(
                    frame,
                    f"Dis: {round(distance, 2)} inch",
                    (x + 5, y + 13),
                    FONTS,
                    0.48,
                    GREEN,
                    2,
                )

        ret, buffer = cv.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
