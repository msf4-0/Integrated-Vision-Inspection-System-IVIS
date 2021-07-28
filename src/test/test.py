"""
TEST
"""

import sys
from pathlib import Path
import base64
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as SessionState

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[1]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined Modules >>>>
from path_desc import chdir_root
from core.utils.log import log_info, log_error  # logger
from data_manager.database_manager import init_connection

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<

# <<<< Variable Declaration <<<<


def main():
    # >>>> Template >>>>
    chdir_root()  # change to root directory
    # initialise connection to Database
    conn = init_connection(**st.secrets["postgres"])
    with st.sidebar.beta_container():
        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')
        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")
    # <<<< Template <<<<
    upload = st.file_uploader("File", type=[
        'zip', 'tar.gz', 'tar.bz2', 'tar.xz'], key='user_custom_upload')
    if upload:
        st.write(upload.type)
        bb = upload.getvalue()
        b64code = base64.b64encode(bb).decode('utf-8')
        data_url = 'data:' + upload.type + ';base64,' + b64code

        html = f"""<a href="{data_url}" >
  <img src="https://images.unsplash.com/photo-1560958089-b8a1929cea89?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1351&q=80" alt="W3Schools" width="104" height="142">
</a>"""
        st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
