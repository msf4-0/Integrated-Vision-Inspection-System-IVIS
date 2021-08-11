"""
Title: New Project Page
Date: 5/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from pathlib import Path
from enum import IntEnum
from time import sleep
from colorutils import hex_to_hsv
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
st.set_page_config(page_title="Integrated Vision Inspection System",
                   page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
# TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

from path_desc import chdir_root
from core.utils.code_generator import get_random_string
from core.utils.log import log_info, log_error  # logger
from core.utils.helper import create_dataframe, get_df_row_highlight_color
from data_manager.database_manager import init_connection
from data_manager.annotation_type_select import annotation_sel
from data_manager.dataset_management import query_dataset_list, get_dataset_name_list
from project.project_management import NewProject, ProjectPagination
from frontend.editor_manager import NewEditor
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_project = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


class AnnotationType(IntEnum):
    Image_Classification = 1
    BBox = 2
    Polygons = 3
    Masks = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return AnnotationType[s]
        except KeyError:
            raise ValueError()


from color_extract import color_extract
from core.utils.helper import get_df_row_highlight_color


def show():

    chdir_root()  # change to root directory

    with st.sidebar.container():

        st.image("resources/MSF-logo.gif", use_column_width=True)
    # with st.container():
        st.title("Integrated Vision Inspection System", anchor='title')

        st.header(
            "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
        st.markdown("""___""")

    # ******** SESSION STATE ***********************************************************

    if "new_project" not in session_state:
        session_state.new_project = NewProject(get_random_string(length=8))
    if 'new_editor' not in session_state:
        session_state.new_editor = NewEditor(get_random_string(length=8))
        # set random project ID before getting actual from Database
    # NOTE
    if 'dataset_page' not in session_state:
        session_state.dataset_page = 0
    # ******** SESSION STATE *********************************************************


# >>>> New Project INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Page title
    st.write("# __Add New Project__")
    st.markdown("___")

    # right-align the project ID relative to the page
    id_blank, id_right = st.columns([3, 1])
    id_right.write(
        f"### __Project ID:__ {session_state.new_project.id}")

    outercol1, outercol2, outercol3 = st.columns([1.5, 3.5, 0.5])
    outercol1.write("## __Project Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):
        context = ['name', session_state.name]
        if session_state.name:
            if session_state.new_project.check_if_exist(context, conn):
                field_placeholder['name'].error(
                    f"Project name used. Please enter a new name")
                sleep(1)
                log_error(f"Project name used. Please enter a new name")
            else:
                session_state.new_project.name = session_state.name
                log_info(f"Project name fresh and ready to rumble")

    outercol2.text_input(
        "Project Title", key="name", help="Enter the name of the project", on_change=check_if_name_exist, args=(place, conn,))
    place["name"] = outercol2.empty()

    # **** Project Description (Optional) ****
    description = outercol2.text_area(
        "Description (Optional)", key="desc", help="Enter the description of the project")
    if description:
        session_state.new_project.desc = description
    else:
        pass

    with outercol2:
        v = annotation_sel()
        if None not in v:
            (deployment_type, editor_base_config) = v

            session_state.new_editor.editor_config = editor_base_config['config']
    # deployment_type = outercol2.selectbox(
    #     "Deployment Type", key="deployment_type", options=DEPLOYMENT_TYPE, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")

            if deployment_type is not None:
                session_state.new_project.deployment_type = deployment_type
                session_state.new_project.query_deployment_id()
                # st.write(session_state.new_project.deployment_id)

            else:
                pass

        place["deployment_type"] = outercol2.empty()

# <<<<<<<< New Project INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>> Choose Dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # include options to create new dataset on this page
    # create 2 columns for "New Data Button"
    outercol1, outercol2, outercol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    outercol1.write("## __Dataset :__")
    # outercol1, outercol2, outercol3 = st.columns([1, 2, 2])

    data_left, data_right = st.columns(2)
    # >>>> Right Column to select dataset >>>>
    # TODO #51 Utilise dataset query from dataset_management
    with outercol3:
        existing_dataset, dataset_table_column_names = query_dataset_list()
        dataset_dict = get_dataset_name_list(existing_dataset)

        session_state.new_project.dataset_chosen = st.multiselect(
            "Dataset List", key="dataset_chosen", options=dataset_dict, help="Assign dataset to the project")
        place["dataset_chosen"] = st.empty()
        # Button to create new dataset
        new_data_button = st.button("Create New Dataset")

        # print choosen dataset
        st.write("### Dataset choosen:")
        if len(session_state.new_project.dataset_chosen) > 0:
            for idx, data in enumerate(session_state.new_project.dataset_chosen):
                st.write(f"{idx+1}. {data}")
        elif len(session_state.new_project.dataset_chosen) == 0:
            st.info("No dataset selected")
    # <<<< Right Column to select dataset <<<<

    # >>>> Left Column to show full list of dataset and selection >>>>
    if "dataset_page" not in session_state:
        session_state.dataset_page = 0

    def next_page():
        session_state.dataset_page += 1

    def prev_page():
        session_state.dataset_page -= 1

    with outercol2:
        start = 10 * session_state.dataset_page
        end = start + 10

        df = create_dataframe(existing_dataset,
                              column_names=dataset_table_column_names, date_time_format=True)

        df_loc = df.loc[:, "ID":"Date/Time"]
        df_slice = df_loc.iloc[start:end]

        # GET color from active theme
        from_config = st.get_option('theme.backgroundColor')

        df_row_highlight_color = get_df_row_highlight_color()

        def highlight_row(x, selections):

            if x.Name in selections:

                return [f'background-color: {df_row_highlight_color}'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df_slice.style.apply(
            highlight_row, selections=session_state.dataset_chosen, axis=1)

        # >>>>DATAFRAME
        st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]))

    # <<<< Left Column to show full list of dataset and selection <<<<

    # >>>> Dataset Pagination >>>>
    _, col1, _, col2, _, col3, _ = st.columns(
        [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])
    num_dataset_per_page = 10
    num_dataset_page = len(
        dataset_dict) // num_dataset_per_page
    # st.write(num_dataset_page)
    if num_dataset_page > 1:
        if session_state.dataset_page < num_dataset_page:
            col3.button(">", on_click=next_page)
        else:
            col3.write("")  # this makes the empty column show up on mobile

        if session_state.dataset_page > 0:
            col1.button("<", on_click=prev_page)
        else:
            col1.write("")  # this makes the empty column show up on mobile

        col2.write(
            f"Page {1+session_state.dataset_page} of {num_dataset_page}")
    # <<<< Dataset Pagination <<<<
    # TODO #52 Add Editor Config
    # **** Submit Button ****
    success_place = st.empty()
    field = [session_state.new_project.name,
             session_state.new_project.deployment_id, session_state.new_project.dataset_chosen]
    st.write(field)
    col1, col2 = st.columns([3, 0.5])
    submit_button = col2.button("Submit", key="submit")

    if submit_button:
        session_state.new_project.has_submitted = session_state.new_project.check_if_field_empty(
            field, field_placeholder=place)

        if session_state.new_project.has_submitted:
            # TODO #13 Load Task into DB after creation of project
            if session_state.new_project.initialise_project():
                session_state.new_editor.project_id = session_state.new_project.id
                if session_state.new_editor.init_editor():
                    success_place.success(
                        f"Successfully stored **{session_state.new_project.name}** project information in database")
                else:
                    success_place.error(
                        f"Failed to stored **{session_state.new_editor.name}** editor config in database")

            else:
                success_place.error(
                    f"Failed to stored **{session_state.new_project.name}** project information in database")

    col1, col2 = st.columns(2)
    col1.write(vars(session_state.new_project))
    col2.write(vars(session_state.new_editor))


def main():
    show()


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
