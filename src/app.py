"""
Title: Integrated Vision Inspection System
AUthor: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""
# ----------Add sys path for modules----------------#

import sys
from os import path, chdir
from pathlib import Path
import psycopg2  # for PostgreSQL

SRC = Path(__file__).parent.resolve()  # ROOT folder -> ./src
print(SRC)
sys.path.insert(0, str(Path(SRC, 'lib')))  # ./lib

# Change to Project Directory
ROOT = SRC.parent.resolve()  # ROOT folder -> ./image_labelling_shrdc


#--------------------Logger-------------------------#
import logging

FORMAT = '[%(levelname)s] %(asctime)s - %(message)s'
DATEFMT = '%d-%b-%y %H:%M:%S'

# logging.basicConfig(filename='test.log',filemode='w',format=FORMAT, level=logging.DEBUG)
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    stream=sys.stdout, datefmt=DATEFMT)

log = logging.getLogger()

#----------------------------------------------------#
import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
layout='wide'

# DEFINE Web APP page configuration
try:
    st.set_page_config(page_title="Integrated Vision Inspection System",
                       page_icon="static/media/shrdc_image/shrdc_logo.png", layout='wide')
except:
    st.beta_set_page_config(page_title="Label Studio Test",
                            page_icon="random", layout=layout)

#---------------Connection to db------------------#


@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])


conn = init_connection()

# Perform query


@st.cache(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


rows = run_query("SELECT * from playground;")

for row in rows:
    st.write(row)
#----------------------------------------------------#


#------------------IMPORT for PAGES-------------------#
from pages import login, dashboard, project, dataset, inference
#----------------------------------------------------#


# PAGES Dictionary
# Import as modules from "./lib/pages"
PAGES = {
    "LOGIN": login,
    "DASHBOARD": dashboard,
    "PROJECT": project,
    "DATASET": dataset,
    "INFERENCE": inference
}


def main():

    #------------------START------------------------#
    with st.sidebar.beta_container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    with st.beta_container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
    st.markdown("""___""")
#-------------------------------------------#

    PAGES["LOGIN"].write()


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        chdir(ROOT)
        main()

    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        
        sys.exit(stcli.main())
        chdir(ROOT)
