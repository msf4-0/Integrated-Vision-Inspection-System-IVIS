import streamlit as st
from time import sleep
import streamlit.components.v1 as components
import numpy as np

st.set_page_config(page_title="label studio test",
                   page_icon="random", layout='wide')
st.markdown("""
# SHRDC Image Labelling Web APP 🎨
""")
flag = False

placeholder1 = st.empty()
# Replace the placeholder with some text:
placeholder1.text("Hello")
# # Replace the text with a chart:
# placeholder1.line_chart({"data": [1, 5, 2, 6]})
# # Replace the chart with several elements:
column1, column2, column3 = st.beta_columns([1, 1, 1])
with column1:
    # col1,col2,col3=st.beta_columns([1,1,1])
    st.write("Project ID: 1")
    st.write("Project Name: 🦍")
    a = 1
    # st.write(a)

    placeholder = st.empty()
    Train1 = placeholder.button(
        label="Train ▶️", key='Train', help="Start Training")
    st.button("Delete", key="del1")
    st.write(f"Train1: {Train1}")

    if Train1:
        Stop1 = placeholder.button(
            label="Stop ⏸️", key='Stop', help="Stop Training")
        # placeholder_s = st.empty()
        st.write(f"Stop1: {Stop1}")
        if Stop1:
            st.warning('Are you sure you want to __STOP__?')
        # st.write(f"Train2: {Train}")
        # st.write(f"Stop2: {Stop}")
with column2:
    # col1,col2,col3=st.beta_columns([1,1,1])
    st.write("Project ID: 2")
    st.write("Project Name: 🦁")
    a = 1
    # st.write(a)

    placeholder2 = st.empty()
    Train2 = placeholder2.button(
        label="Train ▶️", key='Train2', help="Start Training")
    st.button("Delete", key="del2")
    st.write(f"Train2: {Train2}")

    if Train2:
        Stop2 = placeholder2.button(
            label="Stop ⏸️", key='Stop2', help="Stop Training")
        st.write(f"Stop2: {Stop2}")
        st.warning('Are you sure you want to __STOP__?')
        Stop2_confirm = placeholder2.button(
            label="Confirm", key='Stop2_confirm', help="Stop Confirm")
        st.write(f"Stop2_confirm: {Stop2_confirm}")
        if Stop2_confirm == True:
            sleep(1)
            st.success('Training stopped')

# ------------------Forms---------------
with st.form(key='columns_in_form'):
    cols = st.beta_columns(5)
    for i, col in enumerate(cols):
        col.selectbox(f'Make a Selection', ['click', 'or click'], key=i)
    submitted = st.form_submit_button('Submit')


PAGE = ['Project', 'Dataset', 'Inference']
with st.sidebar:
    st.title("Project")
    page = st.radio(label="", options=PAGE, key="page")

from string import ascii_uppercase, digits
from random import choices
with st.beta_expander(label="", expanded=True):
    img_base = "https://www.htmlcsscolor.com/preview/128x128/{0}.png"
    # st.image("https://media.allure.com/photos/5d601b3e531caa0008cbc17c/3:4/w_1279,h_1705,c_limit/IU%20at%20a%20press%20conference.jpg",width=100)
    colors = (
        ''.join(choices(ascii_uppercase[:6] + digits, k=6)) for _ in range(100))
    # st.markdown("---")
    with st.beta_container():
        for col in st.beta_columns(5):
            col.image(img_base.format(next(colors)), use_column_width=True)

    with st.beta_container():
        for col in st.beta_columns(4):
            col.image(img_base.format(next(colors)), use_column_width=True)

    with st.beta_container():
        for col in st.beta_columns(10):
            col.image(img_base.format(next(colors)), use_column_width=True)

st.markdown("""
---
""")


# # bootstrap 4 collapse example
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
#                     <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
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
# tensorboard_link = "http://localhost:6006/"
# components.iframe(tensorboard_link, scrolling=True, height=900)

ran = np.random.rand(14)
st.write(ran)

cols = {
    "Airport__Name": "Airport Name",
    "Aircraft__Make_Model": "Aircraft Make & Model",
    "Effect__Amount_of_damage": "Effect: Amount of Damage",
    "Flight_Date": "Flight Date",
    "Aircraft__Airline_Operator": "Airline Operator",
    "Origin_State": "Origin State",
    "When__Phase_of_flight": "When (Phase of Flight)",
    "Wildlife__Size": "Wildlife Size",
    "Wildlife__Species": "Wildlife Species",
    "When__Time_of_day": "When (Time of Day)",
    "Cost__Other": "Cost (Other)",
    "Cost__Repair": "Cost (Repair)",
    "Cost__Total_$": "Cost (Total) ($)",
    "Speed_IAS_in_knots": "Speed (in Knots)",
}


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
elif pswrd:
    st.sidebar.error(
        "User entered wrong username or password. Please enter again.")
