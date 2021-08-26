"""
    TEST
    """
from pathlib import Path
import sys
from threading import Thread
import streamlit as st
from streamlit import session_state as session_state
SRC = Path(__file__).resolve().parents[1]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from core.utils.log import log_info
st.write(Path.cwd())

from typing import Union
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from mimetypes import guess_type
from threading import Thread
from timeit import timeit
from base64 import b64encode
from io import BytesIO
from streamlit.report_thread import add_report_ctx

data_path = '/home/rchuzh/.local/share/integrated-vision-inspection-system/app_media/dataset/my-third-dataset/IMG_20210315_184149.jpg'


class Task:
    def __init__(self, image_path: Path, opencv_flag: bool = False):
        self.image_path: Path = Path(image_path)
        self.image: Union[Image.Image, np.ndarray] = None
        self.data_url: str = None
        self.opencv_flag: bool = opencv_flag

    def get_image(self):
        if self.opencv_flag:
            image_path = str(self.image_path)
            self.image = cv2.imread(image_path)

        else:

            self.image = Image.open(self.image_path)

        return self.image

    def data_url_encoder(self):
        """Load Image and generate Data URL in base64 bytes

        Args:
            image (bytes-like): BytesIO object

        Returns:
            bytes: UTF-8 encoded base64 bytes
        """
        if isinstance(self.image, np.ndarray):
            image_name = Path(self.image_path).name

            log_info(f"Encoding image into bytes: {str(image_name)}")
            extension = Path(image_name).suffix
            _, buffer = cv2.imencode(extension, self.image)
            log_info("Done enconding into bytes")

            log_info("Start B64 Encoding")

            b64code = b64encode(buffer).decode('utf-8')
            log_info("Done B64 encoding")

        elif isinstance(self.image, Image.Image):
            img_byte = BytesIO()
            image_name = Path(self.image.filename).name  # use Path().name
            log_info(f"Encoding image into bytes: {str(image_name)}")
            self.image.save(img_byte, format=self.image.format)
            log_info("Done enconding into bytes")
            log_info("Start B64 Encoding")
            bb = img_byte.getvalue()
            b64code = b64encode(bb).decode('utf-8')
            log_info("Done B64 encoding")

        mime = guess_type(image_name)[0]
        log_info(f"{image_name} ; {mime}")
        self.data_url = f"data:{mime};base64,{b64code}"
        log_info("Data url generated")

    def data_url_pipeline(self):
        self.get_image()
        self.data_url_encoder()


def perf_metric():
    load_time = timeit(
        globals=globals(),
        setup="""
import streamlit as st;
from PIL import Image;
import cv2;
from pathlib import Path;
from mimetypes import guess_type;
from threading import Thread;
data_path='/home/rchuzh/.local/share/integrated-vision-inspection-system/app_media/dataset/my-third-dataset/IMG_20210315_184149.jpg'
""",
        stmt="""
some_task=Task(image_path=data_path,opencv_flag=True)
data_uri_gen_thread=Thread(target=some_task.data_url_pipeline)
add_report_ctx(data_uri_gen_thread)
# some_task.data_url_pipeline()
data_uri_gen_thread.start()
st.write("Hello")
st.write("Outside thread")
data_uri_gen_thread.join()
    """, number=1)

    st.write(load_time)


st.button("Start Metric", key='perf_metrics', on_click=perf_metric)
