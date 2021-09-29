""" 

Title: New Training InfoDataset
Date: 3/9/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)



Copyright (C) 2021 Selangor Human Resource Development Centre

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Copyright (C) 2021 Selangor Human Resource Development Centre
SPDX-License-Identifier: Apache-2.0
========================================================================================
 """
import sys
from pathlib import Path
from time import sleep

import streamlit as st
from streamlit import cli as stcli  # Add CLI so can run Python script directly
from streamlit import session_state as session_state

# DEFINE Web APP page configuration
layout = 'wide'
# st.set_page_config(page_title="Integrated Vision Inspection System",
#                    page_icon="static/media/shrdc_image/shrdc_logo.png", layout=layout)

# >>>>>>>>>>>>>>>>>>>>>>TEMP>>>>>>>>>>>>>>>>>>>>>>>>

SRC = Path(__file__).resolve().parents[5]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"


if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass
# >>>> User-defined Modules >>>>
from core.utils.form_manager import remove_newline_trailing_whitespace
from core.utils.helper import create_dataframe, get_df_row_highlight_color
from core.utils.log import logger  # logger
from data_manager.database_manager import init_connection
from path_desc import chdir_root

from training.training_management import NewTrainingPagination, NewTrainingSubmissionHandlers, TrainingPagination
from training.model_management import ModelsPagination

# <<<<<<<<<<<<<<<<<<<<<<TEMP<<<<<<<<<<<<<<<<<<<<<<<

# >>>> Variable Declaration <<<<
# initialise connection to Database
conn = init_connection(**st.secrets["postgres"])

# <<<< Variable Declaration <<<<

chdir_root()


def infodataset():
    logger.debug(
        "[NAVIGATOR] At `new_training_infodataset.py` `infodataset` function")
    if 'new_training_place' not in session_state:
        session_state.new_training_place = {}
    if 'new_training_pagination' not in session_state:
        session_state.new_training_pagination = NewTrainingPagination.InfoDataset
    # ************COLUMN PLACEHOLDERS *****************************************************
    st.write("___")

    # to display existing Training info for the users
    existing_info_place = st.empty()

    infocol1, infocol2, infocol3 = st.columns([1.5, 3.5, 0.5])

    info_dataset_divider = st.empty()

    # create 2 columns for "New Data Button"
    datasetcol1, datasetcol2, datasetcol3, _ = st.columns(
        [1.5, 1.75, 1.75, 0.5])

    # COLUMNS for Dataset Dataframe buttons
    _, dataset_button_col1, _, dataset_button_col2, _, dataset_button_col3, _ = st.columns(
        [1.5, 0.15, 0.5, 0.45, 0.5, 0.15, 2.25])

    # ************COLUMN PLACEHOLDERS *****************************************************

    # >>>> New Training INFO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if session_state.new_training.has_submitted[NewTrainingPagination.InfoDataset]:
        # display existing information to the users for easier reference when updating
        # existing training info
        with existing_info_place.container():
            st.info(f"""
            **Current Training Title**: {session_state.new_training.name}  \n
            **Current Description**: {session_state.new_training.desc}  \n
            **Current Dataset List**: {session_state.new_training.dataset_chosen}  \n
            **Current Partition Ratio**: {session_state.new_training.partition_ratio}  \n
            """)

    infocol1.write("## __Training Information :__")

    # >>>> CHECK IF NAME EXISTS CALLBACK >>>>
    def check_if_name_exist(field_placeholder, conn):

        context = {'column_name': 'name',
                   'value': session_state.new_training_name}

        logger.debug(f"New Training: {context}")

        if session_state.new_training_name:
            if session_state.new_training.check_if_exists(context, conn):

                session_state.new_training.name = None
                field_placeholder['new_training_name'].error(
                    f"Training name used. Please enter a new name")
                sleep(1)
                field_placeholder['new_training_name'].empty()
                logger.error(f"Training name used. Please enter a new name")

            else:
                session_state.new_training.name = session_state.new_training_name
                logger.info(f"Training name fresh and ready to rumble")

        else:
            pass

    with infocol2.container():

        # **** TRAINING TITLE ****
        st.text_input(
            "Training Title", key="new_training_name",
            help="Enter the name of the training",
            on_change=check_if_name_exist, args=(session_state.new_training_place, conn,))
        session_state.new_training_place["new_training_name"] = st.empty()

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
    with datasetcol3.container():

        # >>>> Store SELECTED DATASET >>>>
        # - JUST include all the dataset selected during the project creation
        # st.multiselect(
        #     "Dataset List", key="new_training_dataset_chosen",
        #     options=session_state.project.dataset_dict, help="Assign dataset to the training")
        # session_state.new_training_place["new_training_dataset_chosen"] = st.empty(
        # )
        # TODO: REMOVE this session state originally used by the multiselect widget
        session_state.new_training_dataset_chosen = session_state.project.dataset_dict.keys()
        # NOTE: This is changed to directly init the self.dataset_chosen from `project.dataset_dict.keys()`

        if len(session_state.new_training_dataset_chosen) > 0:

            # TODO #111 Dataset Partition Config

            def update_dataset_partition_ratio():

                # if session_state.test_partition == True:
                session_state.new_training.partition_ratio['train'] = session_state.partition_slider[0]
                session_state.new_training.partition_ratio['eval'] = round(session_state.partition_slider[1] -
                                                                           session_state.partition_slider[0], 2)
                session_state.new_training.partition_ratio['test'] = round(
                    1.0 - session_state.partition_slider[1], 2)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATASET PARTITION CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            st.slider("Dataset Partition Ratio",
                      min_value=0.5, max_value=1.0,
                      value=(0.8, 1.0), step=0.1,
                      key="partition_slider", on_change=update_dataset_partition_ratio)

            with st.expander("Partition info"):
                st.info("Ratio of Training datasets to Evaluation datasets. Example: '0.5' means the dataset are split randomly and equally into training and evaluation datasets.")

            session_state.new_training.calc_dataset_partition_size(session_state.new_training_dataset_chosen,
                                                                   session_state.project.dataset_dict)

            st.info(f"""
            #### Train Dataset Ratio: {session_state.new_training.partition_ratio['train']} ({session_state.new_training.partition_size['train']} data)
            #### Evaluation Dataset Ratio: {session_state.new_training.partition_ratio['eval']} ({session_state.new_training.partition_size['eval']} data)
            #### Test Dataset Ratio: {session_state.new_training.partition_ratio['test']} ({session_state.new_training.partition_size['test']} data)
            """)

            if session_state.new_training.partition_ratio['eval'] <= 0:
                st.error(
                    f"Evaluation Dataset Partition Ratio should be more than 0.1")

            # >>>> DISPLAY DATASET CHOSEN >>>>
            st.write("### Dataset chosen:")
            for idx, data in enumerate(session_state.new_training_dataset_chosen):
                st.write(f"{idx+1}. {data}")
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATASET PARTITION CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        elif len(session_state.new_training_dataset_chosen) == 0:
            session_state.new_training_place["new_training_dataset_chosen"].info(
                "No dataset selected")
            session_state.new_training.partition_ratio = {
                'train': 0.8,
                'eval': 0.2,
                'test': 0
            }

    # ******************************* Right Column to select dataset *******************************

    # ******************* Left Column to show full list of dataset and selection *******************
    if "dataset_page" not in session_state:
        session_state.new_training_dataset_page = 0

    with datasetcol2.container():
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
            highlight_row, selections=session_state.new_training_dataset_chosen, axis=1)

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


# ************************* NEW TRAINING SECTION PAGINATION BUTTONS **********************
    # Placeholder for Back and Next button for page navigation
    _, _, new_training_section_next_button_place = st.columns([1, 3, 1])

    if session_state.new_training.has_submitted[NewTrainingPagination.InfoDataset]:
        # # session_state.new_training is a Training instance, and will not need to insert anymore
        def insert_function(): return None
    else:
        insert_function = session_state.new_training.insert_training_info_dataset
    # typing.NamedTuple type
    new_training_infodataset_submission_dict = NewTrainingSubmissionHandlers(
        insert=insert_function,
        update=session_state.new_training.update_training_info_dataset,
        context={
            'new_training_name': session_state.new_training_name,
            'new_training_dataset_chosen': session_state.new_training_dataset_chosen
        },
        name_key='new_training_name'
    )

    # >>>> NEXT BUTTON >>>>

    def to_new_training_next_page():

        # Run submission according to current page
        # NEXT page if constraints are met

        # >>>> IF IT IS A NEW SUBMISSION
        if not session_state.new_training.has_submitted[NewTrainingPagination.InfoDataset]:
            if session_state.new_training.check_if_field_empty(
                    context=new_training_infodataset_submission_dict.context,
                    field_placeholder=session_state.new_training_place,
                    name_key=new_training_infodataset_submission_dict.name_key):
                # INSERT Database
                # Training Name,Desc, Dataset chosen, Partition Size
                session_state.new_training.dataset_chosen = session_state.new_training_dataset_chosen
                if new_training_infodataset_submission_dict.insert():
                    session_state.new_training_pagination = NewTrainingPagination.Model
                    # must set this to tell the models_page.py to move to stay in its page
                    session_state.models_pagination = ModelsPagination.ExistingModels
                    session_state.new_training.has_submitted[NewTrainingPagination.InfoDataset] = True
                    logger.info(
                        f"Successfully created new training {session_state.new_training.id}")

        # >>>> UPDATE if Training has already been submitted prior to this
        elif session_state.new_training.has_submitted[NewTrainingPagination.InfoDataset] == True:
            if session_state.new_training.name:

                # UPDATE Database
                # Training Name,Desc, Dataset chosen, Partition Size
                if new_training_infodataset_submission_dict.update(session_state.new_training_dataset_chosen,
                                                                   session_state.project.dataset_dict):
                    session_state.new_training_pagination = NewTrainingPagination.Model
                    # must set this to tell the models_page.py to move to stay in its page
                    session_state.models_pagination = ModelsPagination.ExistingModels
                    logger.info(
                        f"Successfully updated new training {session_state.new_training.id}")
            else:
                session_state.new_training_place['new_training_name'].error(
                    'Training Name already exists, please enter a new name')

    with new_training_section_next_button_place:
        st.button("Next", key="new_training_next_button",
                  on_click=to_new_training_next_page)


if __name__ == "__main__":
    if st._is_running_with_streamlit:
        infodataset()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
