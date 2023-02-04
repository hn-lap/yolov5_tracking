import cv2

from object_tracking import draw_polygon, infer


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


points = []
detect = False
cap = cv2.VideoCapture("video.mp4")
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = draw_polygon(frame, points)
    if detect:
        frame = infer(points=points)
    key = cv2.waitKey(100)
    if key == ord("q"):
        break
    elif key == ord("d"):
        points.append(points[0])
        detect = True

    cv2.imshow("Video", frame)
    cv2.setMouseCallback("Video", handle_left_click, points)
