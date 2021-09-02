"""
Title: New Training Page
Date: 23/7/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
"""

import sys
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import cli as stcli
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

import numpy as np  # TEMP for table viz
from core.utils.code_generator import get_random_string
from core.utils.helper import create_dataframe, get_df_row_highlight_color
from core.utils.log import log_error, log_info  # logger
from data_manager.database_manager import init_connection
from path_desc import MEDIA_ROOT, chdir_root
from training.model_management import MODEL_TYPE, Model, ModelType, PreTrainedModel
from project.project_management import Project
from training.training_management import NewTraining, TrainingParam
from deployment.deployment_management import DeploymentType
from user.user_management import User
from core.utils.form_manager import remove_newline_trailing_whitespace
# >>>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])


# >>>> Variable Declaration >>>>
new_training = {}  # store
place = {}
DEPLOYMENT_TYPE = ("", "Image Classification", "Object Detection with Bounding Boxes",
                   "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")


def new_training_page():

    # >>>> INIT >>>>
    chdir_root()  # change to root directory

    # ******** SESSION STATE ***********************************************************

    if "new_training" not in session_state:
        session_state.new_training = NewTraining(get_random_string(
            length=8), session_state.project)
        # set random project ID before getting actual from Database
    if 'pt_model' not in session_state:
        session_state.pt_model = PreTrainedModel()
    if 'model' not in session_state:
        session_state.model = Model()

    # ******** SESSION STATE *********************************************************

    # TODO #107 move to Training Dashboard
    # >>>> TRAINING SIDEBAR >>>>
    training_page_options = ("All Trainings", "New Training")
    with st.sidebar.expander("Training Page", expanded=True):
        session_state.current_page = st.radio("", options=training_page_options,
                                              index=0)
    # <<<< TRAINING SIDEBAR <<<<

    # Page title
    st.write("# __Add New Training__")

    # ************COLUMN PLACEHOLDERS *****************************************************
    dt_place, _ = st.columns([3, 1])

    # right-align the training ID relative to the page
    id_blank, id_right = st.columns([3, 1])

    infocol1, infocol2, infocol3 = st.columns([1.5, 3.5, 0.5])

    info_dataset_divider = st.empty()

    # create 2 columns for "New Data Button"
    datasetcol1, datasetcol2, datasetcol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    # COLUMNS for Dataset Dataframe buttons
    _, dataset_button_col1, _, dataset_button_col2, _, dataset_button_col3, _ = st.columns(
        [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    model_dataset_divider = st.empty()

    # COLUMNS for Model section
    modelcol1, modelcol2, modelcol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    # COLUMNS for Model Dataframe buttons
    _, _, model_button_col1, _, model_button_col2, _, model_button_col3, _ = st.columns(
        [1.5, 1.75, 0.15, 0.5, 0.45, 0.5, 0.15, 0.5])
    # ************COLUMN PLACEHOLDERS *****************************************************
    with dt_place:
        st.write("### __Deployment Type:__",
                 f"{session_state.project.deployment_type}")
    st.markdown("___")

    id_right.write(
        f"### __Training ID:__ {session_state.new_training.id}")

    # <<<< INIT <<<<

# >>>> New Training INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    infocol1.write("## __Training Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):

        context = {'column_name': 'name',
                   'value': session_state.new_training_name}
        log_info(f"New Training: {context}")

        if session_state.new_training_name:
            if session_state.new_training.check_if_exist(context, conn):

                session_state.new_training.name = None
                field_placeholder['new_training_name'].error(
                    f"Training name used. Please enter a new name")
                sleep(1)
                field_placeholder['new_training_name'].empty()
                log_error(f"Training name used. Please enter a new name")

            else:
                session_state.new_training.name = session_state.new_training_name
                log_info(f"Training name fresh and ready to rumble")

        else:
            pass

    with infocol2:

        # **** TRAINING TITLE ****
        st.text_input(
            "Training Title", key="new_training_name",
            help="Enter the name of the training",
            on_change=check_if_name_exist, args=(place, conn,))
        place["new_training_name"] = st.empty()

        # **** TRAINING DESCRIPTION (Optional) ****
        description = st.text_area(
            "Description (Optional)", key="new_training_desc",
            help="Enter the description of the training")

        if description:
            session_state.new_training.desc = remove_newline_trailing_whitespace(
                description)
        else:
            pass

# <<<<<<<< New Training INFO <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>> Choose Dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    info_dataset_divider.write("___")

    datasetcol1.write("## __Dataset :__")

    # ******************************* Right Column to select dataset *******************************
    with datasetcol3:

        # >>>> Store SELECTED DATASET >>>>
        session_state.new_training.dataset_chosen = st.multiselect(
            "Dataset List", key="new_training_dataset_chosen",
            options=session_state.project.dataset_dict, help="Assign dataset to the training")
        place["new_training_dataset_chosen"] = st.empty()

        if len(session_state.new_training.dataset_chosen) > 0:

            # TODO #111 Dataset Partition Config
            # >>>> DATASET PARTITION CONFIG >>>>
            session_state.new_training.partition_ratio = st.number_input(
                "Dataset Partition Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.1, key="partition_ratio")
            with st.expander("Partition info"):
                st.info("Ratio of Training datasets to Evaluation datasets. Example: '0.5' means the dataset are split randomly and equally into training and evaluation datasets.")

            # >>>> DISPLAY DATASET CHOSEN >>>>
            st.write("### Dataset choosen:")
            for idx, data in enumerate(session_state.new_training.dataset_chosen):
                st.write(f"{idx+1}. {data}")

        elif len(session_state.new_training.dataset_chosen) == 0:
            place["new_training_dataset_chosen"].info("No dataset selected")

    # ******************************* Right Column to select dataset *******************************

    # ******************* Left Column to show full list of dataset and selection *******************
    if "dataset_page" not in session_state:
        session_state.new_training_dataset_page = 0

    with datasetcol2:
        start = 10 * session_state.new_training_dataset_page
        end = start + 10

        # >>>>>>>>>>PANDAS DATAFRAME >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        df = create_dataframe(session_state.project.datasets,
                              column_names=session_state.project.column_names,
                              sort=True, sort_by='ID', asc=True, date_time_format=True)

        df_loc = df.loc[:, "ID":"Date/Time"]
        df_slice = df_loc.iloc[start:end]

        # GET color from active theme
        df_row_highlight_color = get_df_row_highlight_color()

        def highlight_row(x, selections):

            if x.Name in selections:

                return [f'background-color: {df_row_highlight_color}'] * len(x)
            else:
                return ['background-color: '] * len(x)

        styler = df_slice.style.apply(
            highlight_row, selections=session_state.new_training.dataset_chosen, axis=1)

        # >>>>DATAFRAME
        st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])]))
    # ******************* Left Column to show full list of dataset and selection *******************

    # **************************************** DATASET PAGINATION ****************************************

    # >>>> PAGINATION CALLBACK >>>>
    def next_page():
        session_state.new_training_dataset_page += 1

    def prev_page():
        session_state.new_training_dataset_page -= 1

    # _, col1, _, col2, _, col3, _ = st.columns(
    #     [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    num_dataset_per_page = 10
    num_dataset_page = len(
        session_state.project.dataset_dict) // num_dataset_per_page

    if num_dataset_page > 1:
        if session_state.new_training_dataset_page < num_dataset_page:
            dataset_button_col3.button(">", on_click=next_page)
        else:
            # this makes the empty column show up on mobile
            dataset_button_col3.write("")

        if session_state.new_training_dataset_page > 0:
            dataset_button_col1.button("<", on_click=prev_page)
        else:
            # this makes the empty column show up on mobile
            dataset_button_col1.write("")

        dataset_button_col2.write(
            f"Page {1+session_state.new_training_dataset_page} of {num_dataset_page}")
    # **************************************** DATASET PAGINATION ****************************************

# <<<<<<<< Choose Dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# >>>>>>>> MODEL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    model_dataset_divider.write("___")
    modelcol1.write("## __Deep Learning Model Selection :__")

    # ***********FRAMEWORK LIST *****************************************************************************
    framework_list = [framework.Name for framework in deepcopy(
        session_state.new_training.get_framework_list())]
    framework_list.insert(0, "")
    framework = modelcol2.selectbox("Select Deep Learning Framework", options=framework_list,
                                    format_func=lambda x: 'Select a framework' if x == "" else x)
    session_state.new_training.framework = framework if framework else None
    # ***********FRAMEWORK LIST *****************************************************************************

    # TODO #112 Add model upload widget

    model_upload_select = modelcol2.radio("",
                                          options=MODEL_TYPE,
                                          key='model_selection',
                                          format_func=lambda x: MODEL_TYPE[x])

    # empty() placeholder to dynamically display file upload if checkbox selected
    place["new_training_model_selection"] = modelcol2.empty()

    # *********** USER CUSTOM DEEP LEARNING MODELS *****************************************************************************
    if model_upload_select == ModelType.UserUpload:
        model = place["new_training_model_selection"].file_uploader("User Custom Model Upload", type=[
            'zip', 'tar.gz', 'tar.bz2', 'tar.xz'], key='user_custom_upload_model')
        if model:
            session_state.new_training.model_selected = deepcopy(
                model)  # store in model attribute
            st.write(model)  # TODO
    # *********** USER CUSTOM DEEP LEARNING MODELS *****************************************************************************

    # *********** PRE-TRAINED MODELS *****************************************************************************

    # TODO #120 Fix selectbox session_state not iterable issue
    elif model_upload_select == ModelType.PreTrained:

        pre_trained_models, pt_column_names = session_state.pt_model.query_PT_table()
        pt_name_list = [
            pt.Name for pt in pre_trained_models if pt.Framework == framework]  # List to get DL model name based on framework

        # **********************************************************************************
        # >>>>RIGHT: Pre-trained models selection >>>>
        pt_name_list.insert(0, "")
        try:
            model_selection = modelcol2.selectbox(
                "", options=pt_name_list, key='pre_trained_models', format_func=lambda x: 'Select a Model' if x == "" else x)
        except ValueError as e:
            pass
        # <<<<RIGHT: Pre-trained models selection <<<<

        session_state.new_training.model_selected = model_selection if model_selection else None

        # >>>>LEFT: Pre-trained models dataframe >>>>
        if "pt_page" not in session_state:
            session_state.pt_page = 0

        def next_pt_page():
            session_state.pt_page += 1

        def prev_pt_page():
            session_state.pt_page -= 1
        with modelcol3:
            start = 10 * session_state.pt_page
            end = start + 10

            df = create_dataframe(pre_trained_models, pt_column_names)
            df_loc = df.loc[(df["Framework"] == session_state.new_training.framework),
                            "ID":"Framework"] if framework else df.loc[:, "ID":"Framework"]
            df_slice = df_loc.iloc[start:end]
            if session_state.new_training.model_selected:
                def highlight_row(x, selections):

                    if x.Name in selections:

                        return ['background-color: #90a4ae'] * len(x)
                    else:
                        return ['background-color: '] * len(x)

                styler = df_slice.style.apply(
                    highlight_row, selections=session_state.new_training.model_selected, axis=1)
            else:
                styler = df_slice.style
            st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
                [dict(selector='th', props=[('text-align', 'center')])]))

        # >>>> Dataset Pagination >>>>

        _, _, col1, _, col2, _, col3, _ = st.columns(
            [1.5, 1.75, 0.15, 0.5, 0.45, 0.5, 0.15, 0.5])
        num_data_per_page = 10
        num_data_page = len(
            pre_trained_models) // num_data_per_page
        # st.write(num_dataset_page)
        if num_data_page > 1:
            if session_state.pt_page < num_data_page:
                col3.button(">", on_click=next_pt_page)
            else:
                col3.write("")  # this makes the empty column show up on mobile

            if session_state.pt_page > 0:
                col1.button("<", on_click=prev_pt_page)
            else:
                col1.write("")  # this makes the empty column show up on mobile

            col2.write(f"Page {1+session_state.pt_page} of {num_data_page}")
        # <<<< Dataset Pagination <<<<

        # <<<<LEFT: Pre-trained models dataframe <<<<

    # *********** PRE-TRAINED MODELS *****************************************************************************

    # *********** PROJECT TRAINED MODELS *****************************************************************************

    else:

        project_models, project_model_column_names = session_state.model.query_model_table()
        if project_models:
            project_model_name_list = [
                m.Name for m in project_models if m.Framework == framework]  # List to get DL model name based on framework
            project_model_name_list.insert(0, "")
        else:
            project_model_name_list = []
        # **********************************************************************************
        # >>>>RIGHT: Project models selection >>>>

        try:
            model_selection = modelcol2.selectbox(
                "", options=project_model_name_list, key='project_models', format_func=lambda x: 'Select a Model' if x == "" else x)
        except ValueError as e:
            pass
        # <<<<RIGHT: Project models selection <<<<

        session_state.new_training.model_selected = model_selection if model_selection else None

        # >>>>LEFT: Pre-trained models dataframe >>>>
        if "model_page" not in session_state:
            session_state.model_page = 0

        def next_model_page():
            session_state.model_page += 1

        def prev_model_page():
            session_state.model_page -= 1
        with modelcol3:
            start = 10 * session_state.model_page
            end = start + 10
            if project_models:
                df = create_dataframe(
                    project_models, project_model_column_names)
                df_loc = df.loc[(df["Framework"] == session_state.new_training.framework),
                                "ID":"Framework"] if framework else df.loc[:, "ID":"Framework"]
                df_slice = df_loc.iloc[start:end]
                if session_state.new_training.model_selected:
                    def highlight_row(x, selections):

                        if x.Name in selections:

                            return ['background-color: #90a4ae'] * len(x)
                        else:
                            return ['background-color: '] * len(x)

                    styler = df_slice.style.apply(
                        highlight_row, selections=session_state.new_training.model_selected, axis=1)
                else:
                    styler = df_slice.style
                st.table(styler.set_properties(**{'text-align': 'center'}).set_table_styles(
                    [dict(selector='th', props=[('text-align', 'center')])]))
            else:
                st.error(
                    "No Project Deep Learning Models available. Please choose from the list of Pre-trained Models or Upload your own Deep Learning Model")

        # >>>> Dataset Pagination >>>>
        if project_models:
            _, _, col1, _, col2, _, col3, _ = st.columns(
                [1.5, 1.75, 0.15, 0.5, 0.45, 0.5, 0.15, 0.5])
            num_data_per_page = 10
            num_data_page = len(
                project_models) // num_data_per_page
            # st.write(num_dataset_page)
            if num_data_page > 1:
                if session_state.model_page < num_data_page:
                    col3.button(">", on_click=next_model_page)
                else:
                    # this makes the empty column show up on mobile
                    col3.write("")

                if session_state.model_page > 0:
                    col1.button("<", on_click=prev_model_page)
                else:
                    # this makes the empty column show up on mobile
                    col1.write("")

                col2.write(
                    f"Page {1+session_state.model_page} of {num_data_page}")
    place["model"] = modelcol2.empty()  # TODO :KIV

    # >>>>>>>>>>>>>>>>>>WARNING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not session_state.new_training.model_selected:

        place["model"].info("No Deep Learning Model selected")

    else:
        def check_if_model_name_exist(field_placeholder, conn):
            context = ['name', session_state.model_name]
            if session_state.model_name:
                if session_state.model.check_if_exist(context, conn):
                    field_placeholder['model_name'].error(
                        f"Model name used. Please enter a new name")
                    sleep(1)
                    log_error(f"Model name used. Please enter a new name")
                else:
                    session_state.model.name = session_state.model_name
                    log_info(f"Model name fresh and ready to rumble")

        place["model"].write(
            f"### **Deep Learning Model selected:** {session_state.new_training.model_selected} ")
        modelcol2.text_input(
            "Exported Model Name", key="model_name", help="Enter the name of the exported model after training", on_change=check_if_model_name_exist, args=(place, conn,))
        place["model_name"] = modelcol2.empty()
    # *********** PROJECT TRAINED MODELS *****************************************************************************

    # <<<<<<<< Model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # TODO #104 Training Configuration and Data Augmentation Setup

    # >>>>>>>> Training Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # TODO !!!
    st.write("___")
    DATASET_LIST = []
    # **** Image Augmentation (Optional) ****
    outercol1, outercol2, outercol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])
    outercol1.write("## __Image Augmentation (Optional) :__")
    augmentation_activate = outercol1.checkbox("Image Augmentation", value=False,
                                               key='augmentation_checkbox', help="Optional")

    if augmentation_activate:
        session_state.new_training.augmentation = outercol2.multiselect(
            "Augmentation List", key="augmentation", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")
        place["augmentation"] = st.empty()
        outercol3.error("# WIP")
    else:
        outercol2.info("Augmentation deactivated")
    # **** Training Parameters (Optional) ****
    # Reference to "training_param" attribute of 'session_state.new_training' object
    training_param = session_state.new_training.training_param

    st.write("___")
    outercol1, outercol2, outercol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])
    outercol1.write("## __Training Parameters :__")
    training_param.training_param_optional = outercol2.multiselect(
        "Training Parameters", key="training_param", options=DATASET_LIST, format_func=lambda x: 'Select an option' if x == '' else x, help="Select the type of deployment of the project")

    with outercol2:
        training_param.num_classes = st.number_input("Number of Class", min_value=1,
                                                     step=1, key="num_class", help="Value must correspond to the number of labels used in the dataset.")
        training_param.batch_size = st.number_input("Batch Size", min_value=1, step=1, key="batch_size",
                                                    help="Number of data passed into the target device at one instance. **Number of Batch = Number of data/Batch size**")
        training_param.num_steps = st.number_input("Number of Training Steps",
                                                   min_value=1, step=1, key="num_steps", help="Number of training steps per training session")
        training_param.learning_rate = st.number_input("Learning Rate", min_value=0.000001,
                                                       step=0.0000001, format='%.7f')
    place["training_param"] = st.empty()
    outercol3.error("# WIP")
    # >>>>>>>> Training Configuration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # **** Submit Button *******************************************************************
    success_place = st.empty()
    field = [session_state.new_training.name,
             session_state.new_training.dataset_chosen, session_state.new_training.model_selected]
    st.write(field)
    col1, col2 = st.columns([3, 0.5])
    submit_button = col2.button("Submit", key="submit")

    if submit_button:
        session_state.new_training.has_submitted = session_state.new_training.check_if_field_empty(
            field, field_placeholder=place)

        if session_state.new_training.has_submitted:
            if session_state.new_training.initialise_training(session_state.model, session_state.project):
                success_place.success(
                    f"Successfully stored **{session_state.new_training.name}** training information in database")

            else:
                success_place.error(
                    f"Failed to stored **{session_state.new_training.name}** training information in database")

    col1, col2 = st.columns(2)
    col1.write("Project Class")
    col1.write(vars(session_state.project))
    col2.write("New Training Class")
    col2.write(vars(session_state.new_training))


def index():
    RELEASE = False

    # ****************** TEST ******************************
    if not RELEASE:
        log_info("At Labelling INDEX")

        # ************************TO REMOVE************************
        with st.sidebar.container():
            st.image("resources/MSF-logo.gif", use_column_width=True)
            st.title("Integrated Vision Inspection System", anchor='title')
            st.header(
                "(Integrated by Malaysian Smart Factory 4.0 Team at SHRDC)", anchor='heading')
            st.markdown("""___""")

        # ************************TO REMOVE************************
        project_id_tmp = 43
        log_info(f"Entering Project {project_id_tmp}")

        # session_state.append_project_flag = ProjectPermission.ViewOnly

        if "project" not in session_state:
            session_state.project = Project(project_id_tmp)
            log_info("Inside")
        if 'user' not in session_state:
            session_state.user = User(1)
        # ****************************** HEADER **********************************************
        st.write(f"# {session_state.project.name}")

        project_description = session_state.project.desc if session_state.project.desc is not None else " "
        st.write(f"{project_description}")

        st.markdown("""___""")
        # ****************************** HEADER **********************************************
        new_training_page()
        # TODO #108 Add Return to Training Dashboard Button with Callback


if __name__ == "__main__":
    if st._is_running_with_streamlit:

        index()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
