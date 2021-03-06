import json
import streamlit as st
from time import sleep
import streamlit.components.v1 as components
import numpy as np
from threading import Thread
from pathlib import Path

# --------------------------
# Add sys path for modules
import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), 'lib'))  # ./lib
# --------------------------
flag = False
ROOT = Path(__file__).parents[2]
module_path = Path(ROOT, "src", "lib")
sys.path.append(str(module_path))

st.set_page_config(page_title="label studio test",
                   page_icon="random", layout='wide')
st.markdown("""
# SHRDC Image Labelling Web APP 🎨
""")

# st.write(module_path)
# st.write(sys.path)
# placeholder1 = st.empty()
# # Replace the placeholder with some text:
# placeholder1.text("Hello")
# # # Replace the text with a chart:
# # placeholder1.line_chart({"data": [1, 5, 2, 6]})
# # # Replace the chart with several elements:
# column1, column2, column3 = st.beta_columns([1, 1, 1])
# with column1:
#     # col1,col2,col3=st.beta_columns([1,1,1])
#     st.write("Project ID: 1")
#     st.write("Project Name: 🦍")
#     a = 1
#     # st.write(a)

#     placeholder = st.empty()
#     Train1 = placeholder.button(
#         label="Train ▶️", key='Train', help="Start Training")
#     st.button("Delete", key="del1")
#     st.write(f"Train1: {Train1}")

#     if Train1:
#         Stop1 = placeholder.button(
#             label="Stop ⏸️", key='Stop', help="Stop Training")
#         # placeholder_s = st.empty()
#         st.write(f"Stop1: {Stop1}")
#         if Stop1:
#             st.warning('Are you sure you want to __STOP__?')
#         # st.write(f"Train2: {Train}")
#         # st.write(f"Stop2: {Stop}")
# with column2:
#     # col1,col2,col3=st.beta_columns([1,1,1])
#     st.write("Project ID: 2")
#     st.write("Project Name: 🦁")
#     a = 1
#     # st.write(a)

#     placeholder2 = st.empty()
#     Train2 = placeholder2.button(
#         label="Train ▶️", key='Train2', help="Start Training")
#     st.button("Delete", key="del2")
#     st.write(f"Train2: {Train2}")

#     if Train2:
#         Stop2 = placeholder2.button(
#             label="Stop ⏸️", key='Stop2', help="Stop Training")
#         st.write(f"Stop2: {Stop2}")
#         st.warning('Are you sure you want to __STOP__?')
#         Stop2_confirm = placeholder2.button(
#             label="Confirm", key='Stop2_confirm', help="Stop Confirm")
#         st.write(f"Stop2_confirm: {Stop2_confirm}")
#         if Stop2_confirm == True:
#             sleep(1)
#             st.success('Training stopped')

# # ------------------Forms---------------
# with st.form(key='columns_in_form'):
#     cols = st.beta_columns(5)
#     for i, col in enumerate(cols):
#         col.selectbox(f'Make a Selection', ['click', 'or click'], key=i)
#     submitted = st.form_submit_button('Submit')


# PAGE = ['Project', 'Dataset', 'Inference']
# with st.sidebar:
#     st.title("Project")
#     page = st.radio(label="", options=PAGE, key="page")

# from string import ascii_uppercase, digits
# from random import choices
# with st.beta_expander(label="", expanded=True):
#     img_base = "https://www.htmlcsscolor.com/preview/128x128/{0}.png"
#     # st.image("https://media.allure.com/photos/5d601b3e531caa0008cbc17c/3:4/w_1279,h_1705,c_limit/IU%20at%20a%20press%20conference.jpg",width=100)
#     colors = (
#         ''.join(choices(ascii_uppercase[:6] + digits, k=6)) for _ in range(100))
#     # st.markdown("---")
#     with st.beta_container():
#         for col in st.beta_columns(5):
#             col.image(img_base.format(next(colors)), use_column_width=True)

#     with st.beta_container():
#         for col in st.beta_columns(4):
#             col.image(img_base.format(next(colors)), use_column_width=True)

#     with st.beta_container():
#         for col in st.beta_columns(10):
#             col.image(img_base.format(next(colors)), use_column_width=True)

# st.markdown("""
# ---
# """)


# # # bootstrap 4 collapse example
# st.markdown("## Image Annotation:")
# with st.beta_expander(label="", expanded=True):
#     components.html(
#         """
#         <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8" />
#         <meta http-equiv="X-UA-Compatible" content="IE=edge" />
#         <meta name="viewport" content="width=device-width, initial-scale=1.0" />
#         <title>Document</title>
#         <!-- Include Label Studio stylesheet -->
#         <link
#         href="https://unpkg.com/label-studio@1.0.1/build/static/css/main.css"
#         rel="stylesheet"
#         />
#         <!-- Include the Label Studio library -->
#         <script src="https://unpkg.com/label-studio@1.0.1/build/static/js/main.js"></script>
#     </head>
#     <body>
#         <div id="label-studio">
#         <!-- Initialize Label Studio -->
#         <script>
#             var labelStudio = new LabelStudio("label-studio", {
#             config: `
#             <View>
#                 <View style="display:flex;align-items:start;gap:8px;flex-direction:column-reverse">
#                     <Image name="img" value="extra/chair.jpg" zoom="true" zoomControl="true" rotateControl="true"/>
#                     <View>
#                     <Filter toName="tag" minlength="0" name="filter"/>
#                     <RectangleLabels name="tag" toName="img" showInline="true">
#                         <Label value="Moon"/>
#                         <Label value="Comet"/>
#                     </RectangleLabels>
#                     </View>
#                 </View>
#                 </View>

#         `,

#             interfaces: [
#                 "panel",
#                 "update",
#                 "submit",
#                 "controls",
#                 "side-column",
#                 "annotations:menu",
#                 "annotations:add-new",
#                 "annotations:delete",
#                 "predictions:menu",
#             ],

#             user: {
#                 pk: 1,
#                 firstName: "James",
#                 lastName: "Dean",
#             },

#             task: {
#                 annotations: [],
#                 predictions: [],
#                 id: 1,
#                 data: {
#                 image:
#                     "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg",
#                 },
#             },

#             onLabelStudioLoad: function (LS) {
#                 var c = LS.annotationStore.addAnnotation({
#                 userGenerate: true,
#                 });
#                 LS.annotationStore.selectAnnotation(c.id);
#             },
#             });
#         </script>
#         </div>
#     </body>
#     </html>
#         """,
#         height=1000, scrolling=True
#     )
# # tensorboard_link = "http://localhost:6006/"
# # components.iframe(tensorboard_link, scrolling=True, height=900)

# ran = np.random.rand(14)
# st.write(ran)

# cols = {
#     "Airport__Name": "Airport Name",
#     "Aircraft__Make_Model": "Aircraft Make & Model",
#     "Effect__Amount_of_damage": "Effect: Amount of Damage",
#     "Flight_Date": "Flight Date",
#     "Aircraft__Airline_Operator": "Airline Operator",
#     "Origin_State": "Origin State",
#     "When__Phase_of_flight": "When (Phase of Flight)",
#     "Wildlife__Size": "Wildlife Size",
#     "Wildlife__Species": "Wildlife Species",
#     "When__Time_of_day": "When (Time of Day)",
#     "Cost__Other": "Cost (Other)",
#     "Cost__Repair": "Cost (Repair)",
#     "Cost__Total_$": "Cost (Total) ($)",
#     "Speed_IAS_in_knots": "Speed (in Knots)",
# }


# column = st.selectbox("Describe Column", ran, format_func=cols.get)

# st.write(ran[column].describe())
display = ("male", "female")

options = list(range(len(display)))

value = st.selectbox("gender", options, format_func=lambda x: display[x])

st.write(value)

# test password


def is_authenticated(password):
    return password == "shrdc"


password_place = st.sidebar.empty()
pswrd = password_place.text_input(label="Password", type="password")

if is_authenticated(pswrd):

    # st.balloons()
    st.sidebar.header("Welcome In")
    password_place.success("Welcome in")
    sleep(2)
    password_place.empty()

elif pswrd:
    st.sidebar.error(
        "User entered wrong username or password. Please enter again.")

# from annotation.annotation_template import loadAnnotationTemplate

# # tuple of annotation types
# annotationType_list = ("", "Image Classification", "Object Detection with Bounding Boxes",
#                        "Semantic Segmentation with Polygons", "Semantic Segmentation with Masks")
# annotationType_index = list(range(len(annotationType_list)))
# annotationType = st.selectbox("Template", annotationType_list, format_func=lambda x: 'Select an option' if x == '' else x,
#                               key="annotation_type", help="Please select the desired type of annotation")
# st.warning(annotationType)

# if annotationType is not "":
#     annotationConfig_template = loadAnnotationTemplate(
#         annotationType_list.index(annotationType) - 1)

#     st.write(annotationConfig_template)  # annotation config template
# st.write(annotationConfig_template["config"])

#------------Webcam------#
# from webcam import webcam
# with st.beta_container():
#     captured_image = webcam()
#     if captured_image is None:
#         st.write("Waiting for capture...")
#     else:
#         st.write("Got an image from the webcam:")
#         st.image(captured_image)

#-----open JSON file-----#
import json
# import xmltodict
json_file_upload = st.file_uploader(
    label="upload", type=['json'], key='json_files')

if json_file_upload is not None:
    st.write(json_file_upload)
    xml = json_file_upload.read()
    st.write(xml)
    # json_file = json.dumps(xmltodict.parse(xml), indent=4)
    json_file = json.dumps(xml.decode('utf-8'), indent=4)
    st.write(type(json_file))
    json_obj = json.loads(json_file)

    st.json(json_obj)


# *******************************************************
# Session State
# *******************************************************
st.title('Counter Example')

# Streamlit runs from top to bottom on every iteraction so
# we check if `count` has already been initialized in st.session_state.

# If no, then initialize count to 0
# If count is already initialized, don't do anything
if 'count' not in st.session_state:
    st.session_state.count = 2

# Create a button which will increment the counter
increment = st.button('Increment')
if increment:
    st.session_state.count += 1

# A button to decrement the counter
decrement = st.button('Decrement')
if decrement:
    st.session_state.count -= 1

st.write('Count = ', st.session_state.count)

#*****************CALLBACK**********************#


def update_first():
    st.session_state.second = st.session_state.first


def update_second():
    st.session_state.first = st.session_state.second


st.title('🪞 Mirrored Widgets using Session State')

st.text_input(label='Textbox 1', key='first', on_change=update_first)
st.text_input(label='Textbox 2', key='second', on_change=update_second)

# ************************************************
# TIC TAC TOE


def checkRows(board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
        return None


def checkDiagonals(board):
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board) - i - 1] for i in range(len(board))])) == 1:
        return board[0][len(board) - 1]  # return middle
    return None


def checkWin(board):
    for newBoard in [board, np.transpose(board)]:
        result = checkRows(newBoard)
        if result:
            return result

    return checkDiagonals(board)


def show():
    st.write("""
    ## 🕸️ Tic Tac Toe
    """)
    st.write("")

    # Initialise state
    if "board" not in st.session_state:
        st.session_state.board = np.full((3, 3), ".", dtype=str)
        st.session_state.next_player = "X"
        st.session_state.winner = None

    # Define callback function to handle button clicks
    def handle_click(i, j):
        if not st.session_state.winner:
            st.session_state.board[i, j] = st.session_state.next_player
            st.session_state.next_player = (
                "O" if st.session_state.next_player == "X"else "X")
            winner = checkWin(st.session_state.board)
            if winner != ".":
                st.session_state.winner = winner

    for i, row in enumerate(st.session_state.board):
        cols = st.beta_columns([0.1, 0.1, 0.1, 0.7])
        for j, field in enumerate(row):
            cols[j].button(field, key=f"{i}-{j}",
                           on_click=handle_click, args=(i, j))
    if st.session_state.winner:
        st.success(f"Congrats! {st.session_state.winner} won the game! 🎈")


show()
# ********************************
# TEST PLACEHOLDER --> FAILING
# ********************************

# new_placeholder = st.empty()

# # new_placeholder.line_chart({"data":[1,2,3,4]})
# if "button_count" not in st.session_state:
#     st.session_state.button_count = 0
#     st.session_state.run_id = 0

# st.write(st.session_state.button_count, st.session_state.run_id)
# with new_placeholder.beta_container():
#     st.write("Hi")
#     st.write("Bye")


# button = st.button("test")
# st.write(button)
# if button:

#     st.session_state.button_count += 1
#     st.session_state.run_id += 1
#     new_placeholder.empty()
#     st.write(st.session_state.button_count)
