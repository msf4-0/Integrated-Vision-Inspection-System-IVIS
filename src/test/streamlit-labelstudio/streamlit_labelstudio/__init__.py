import os
import streamlit.components.v1 as components
import streamlit as st

# st.set_page_config(layout='wide')
# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = False

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if not _RELEASE:
    _component_func = components.declare_component(
        # We give the component a simple, descriptive name ("my_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "streamlit_labelstudio",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3001",
    )
else:
    # When we're distributing a production version of the component, we'll
    # replace the `url` param with `path`, and point it to to the component's
    # build directory:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "streamlit_labelstudio", path=build_dir)


# Define a public function for the package,
# which wraps the caller to the frontend code
def st_labelstudio(config, interfaces, user, task, key=None):
    component_value = _component_func(
        config=config, interfaces=interfaces, user=user, task=task, key=key)
    return component_value


# if not _RELEASE:
#     config = """
#                 <View>
# <Header value="Select label and start to click on image"/>
#   <View style="display:flex;align-items:start;gap:8px;flex-direction:column-reverse">
#     <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false" grid="true" brightnessControl="true" contrastControl="true"/>
#     <View>
#       <Filter toName="tag" minlength="0" name="filter"/>
#       <RectangleLabels name="tag" toName="img" showInline="true" fillOpacity="0.8">
#         <Label value="Hello"/>
#         <Label value="World"/>
#       </RectangleLabels>
#     </View>
#   </View>
# </View>
#     """

#     interfaces = [
#     "annotations:add-new",
#     "annotations:delete",
#     "annotations:menu",
#     "controls",
#     "panel",
#     "predictions:menu",
#     "side-column",
#     "skip",
#     "submit"
#     "update",
# ],

#     user = {
#     'pk': 1,
#     'firstName': "Zhen Hao",
#     'lastName': "Chu"
# },

#     task = {
#     'completions': [],
#     'predictions': [],
#     'id': 1,
#     'data': {
#         'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
#     }
# }

#     results_raw = st_labelstudio(
#         config, interfaces, user, task, key='random')
#     st.write(results_raw)
#     if results_raw is not None:
#         areas = [v for k, v in results_raw['areas'].items()]

#         results = []
#         for a in areas:
#             results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
#                             'height': a['height'], 'label': a['results'][0]['value']['rectanglelabels'][0]})
#         with st.beta_expander('Show Annotation Log'):

#             st.table(results)
