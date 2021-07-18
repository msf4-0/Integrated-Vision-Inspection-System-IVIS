"""
Title: Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
import streamlit as st

from pathlib import Path
from time import sleep
import sys
import logging


# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[2]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

for path in sys.path:
    if str(LIB_PATH) not in sys.path:
        sys.path.insert(0, str(LIB_PATH))  # ./lib
    else:
        pass
from path_desc import chdir_root
layout = 'wide'
# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<
# DEFINE Web APP page configuration
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)


import psycopg2


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])


#--------------------Logger-------------------------#
FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.INFO)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#
# PROJECT_PAGE_OPTIONS = {"All Projects":, "New Project":}

def show():
    chdir_root()

    # #------------------START------------------------#
    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
    st.markdown("""___""")

    #-------------------------------------------#
    conn = init_connection()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "All Projects"
        st.session_state.previous_page = "All Projects"

    # >>>> Project Sidebar >>>>
    project_page_options = ("All Projects", "New Project")
    with st.sidebar.beta_expander("Project Page", expanded=True):
        st.session_state.current_page = st.radio("project_page_select", options=project_page_options,
                                                 index=0)

    # # <<<<<<<<<<<<<<<<<<<<<<<<<
    # st.write("")


show()


# if __name__ == "__main__":
#     if st._is_running_with_streamlit:

#         main()
#     else:
#         sys.argv = ["streamlit", "run", sys.argv[0]]
#         sys.exit(stcli.main())
