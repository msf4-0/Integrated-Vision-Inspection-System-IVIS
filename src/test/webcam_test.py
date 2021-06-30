# import the opencv library
import cv2
import sys

# define a video capture object
link = "http://192.168.1.105:4747/video"
local_link = "http://127.0.0.1:4747"
cap = cv2.VideoCapture(link)
if not cap.isOpened():
    sys.exit(1)

while cap.isOpened():

    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        continue

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

# import cv2
# import glob

# for camera in glob.glob("/dev/video?"):
#     c = cv2.VideoCapture(camera)
#     print(camera)
