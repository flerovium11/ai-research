import cv2
from ultralytics import YOLO

yolo = YOLO('models/yolov5_finetuned_200ep.pt')
cap = cv2.VideoCapture(0)  # Open the default camera (0)


def preprocess(image, dims=(512, 512)):
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # crop image to a centered square
    min_dim = min(image.shape[0], image.shape[1])
    start_x = (image.shape[1] - min_dim) // 2
    start_y = (image.shape[0] - min_dim) // 2
    image = image[start_y:start_y+min_dim, start_x:start_x+min_dim]

    image = cv2.resize(image, dims)
    return image


while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Perform your ML processing here on 'frame'
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = preprocess(frame)
    results = yolo(frame)
    image = results[0].plot()
    image = cv2.resize(image, (1024, 1024))
    cv2.imshow('Webcam', image)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
