import cv2
from threading import Thread, Lock, Event
from time import sleep
from typing import Union, Tuple


class CameraFailError(Exception):
    pass


class WebcamVideoStream:
    # based on https://github.com/xavialex/Streamlit-TF-Real-Time-Object-Detection/blob/master/video_utils.py
    def __init__(
            self, src: Union[int, str], shape: Tuple[int, int] = None,
            is_usb_camera: bool = True, name: str = "WebcamVideoStream"):
        """Initialize the video stream from a video

        Args:
            src (int | str): port or address of the camera to use.
            name (str, default='WebcamVideoStream'): Name for the thread.
            shape (Tuple[int, int]): set the resolution for the camera (might not work)
            is_usb_camera (bool, default=True): whether it is a USB camera or other types of camera,
                such as IP camera, to avoid issues when closing the camera.
        Attributes:
            name (str, default='WebcamVideoStream'): Name for the thread.
            stream (cv2.VideoCapture): Webcam video stream.
            grabbed (bool): Tells if the current frame's been correctly read.
            frame (nparray): OpenCV image containing the current frame.
            lock (_thread.lock): Lock to avoid race condition.
            _stop_event (Event): Event used to gently stop the thread.
        """
        self.name = name
        self.is_usb_camera = is_usb_camera
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise CameraFailError(f"Error opening camera from source: {src}")
        else:
            print(f"[INFO] Camera opened successfuly for src: {src}")
        self.shape = shape
        if self.shape is not None:
            self.stream.set(3, shape[0])
            self.stream.set(4, shape[1])
        self.grabbed, self.frame = self.stream.read()
        if not self.grabbed:
            raise CameraFailError(f"Cannot grab frame from source: {src}")
        else:
            print("[INFO] Camera is reading frames")
        self.lock = Lock()
        self._stop_event = Event()

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, daemon=True,
               name=self.name).start()
        return self

    def update(self):
        # Continuosly iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                self.grabbed, self.frame = self.stream.read()
                # FPS = 60, i.e., 1 / 60 = 0.016667
                sleep(0.0166)
            else:
                print(
                    f"[INFO] The thread for reading the camera {self.name} is stopped")
                return
        print(
            f"[INFO] The camera {self.name} is not opened anymore, stopping the thread now.")
        self.stop()

    def read(self):
        return self.frame

    def stop(self):
        self.lock.acquire()
        if self.is_usb_camera:
            # cannot release for IP camera or error would occur
            self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()
