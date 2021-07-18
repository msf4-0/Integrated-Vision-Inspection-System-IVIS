# import the opencv library
import cv2
import sys
import streamlit as st
st.title("Webcam Live Stream")

# button to activate run
run_button = st.checkbox("Run", value=False, key='run')
FRAME_WINDOW = st.image([])  # create empty image window


def set_resolution(cap,width, height, fps):
    cap.set(3, width)
    cap.set(4, height)
    cap.set(5, fps)

    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(video_width, video_height, video_fps)

if 'cap' not in st.session_state:
    st.session_state.cap=cv2.VideoCapture(1)

# define a video capture object
link = "http://192.168.1.104:4747/video"
local_link = "http://127.0.0.1:4747"
# cap = cv2.VideoCapture(1)
cap=st.session_state.cap
if not cap.isOpened():
    sys.exit(1)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_backend = cap.get(cv2.CAP_PROP_BACKEND)
print(video_width, video_height, video_fps, video_backend)

# set_resolution(1280, 720, 5)

while run_button and cap.isOpened():

    # Capture the video frame
    # by frames
    ret, frame = cap.read()
    print(cap.read()[0])
    if not ret:
        cap.release()
        continue
    print(frame.shape)

    # Convert frame colorspace to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    FRAME_WINDOW.image(frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

random_button=st.button("random")
st.write(random_button)
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

# import cv2
# import glob

# for camera in glob.glob("/dev/video?"):
#     c = cv2.VideoCapture(camera)
#     print(camera)
