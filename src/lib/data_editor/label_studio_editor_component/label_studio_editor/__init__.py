import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
import streamlit as st
import streamlit.components.v1 as components
from streamlit import session_state as session_state

SRC = Path(__file__).resolve().parents[4]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

# >>>> User-defined modules >>>>>
from core.utils.helper import check_args_kwargs
from core.utils.log import logger

_RELEASE = True


if not _RELEASE:
    _component_func = components.declare_component(
        "label_studio_editor",
        url="http://localhost:3000",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "label_studio_editor", path=build_dir)


def labelstudio_editor(
        config: str,
        interfaces: List[str],
        user: Dict,
        task: Dict,
        key: str = None,
        on_change: Optional[Callable] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None
) -> List:
    """Data annotator component based on Label Studio
        https://labelstud.io/guide/frontend_reference.html
        *Examples are shown below the function definition

    Args:
        config (str): Label Studio template which is an XML DOM
        interfaces (List[str]): List of interfaces to be displayed on the editor
        user (dict): Dictionary of user details
        task (dict): Label Studio Task format
        key ([type], optional): Unique component identifier. Defaults to None.
        on_change (Optional[Callable], optional): Callback function. Defaults to None.
        args (Optional[Tuple], optional): Arguments to Callback functino. Defaults to None.
        kwargs (Optional[Dict], optional): Keyword arguments to callback function. Defaults to None.

    Returns:
        List: Results from the Label Studio Editor ([List of Dictionaries], flag)
    """
    component_value = _component_func(
        config=config, interfaces=interfaces, user=user, task=task, key=key, default=[])
    if component_value:
        if component_value[0]:
            logger.info(f"From LS Component: Flag {component_value[1]}")

        if on_change:
            logger.info(f"Inside LS Editor Callback")
            wildcard = args if args else kwargs
            if args or kwargs:
                check_args_kwargs(wildcards=wildcard, func=on_change)
            if args:
                on_change(*args)
            elif kwargs:
                on_change(**kwargs)
            else:
                on_change()
    # logger.info(f"Label result: {component_value}")
    return component_value


# if not _RELEASE:
#     import streamlit as st

#     st.subheader("Component with constant args")
#     config = """
# <View>
#     <View style="padding: 25px; box-shadow: 2px 2px 8px #AAA">
#         <Header value="Select label and start to click on image"/>
#         <View style="display:flex;align-items:start;gap:8px;flex-direction:column-reverse">
#             <Image name="img" value="$image" zoom="true" zoomControl="true" rotateControl="false" grid="true" brightnessControl="true" contrastControl="true"/>
#             <View>
#                 <Filter toName="tag" minlength="0" name="filter"/>
#                 <RectangleLabels name="tag" toName="img" showInline="true" fillOpacity="0.5" strokeWidth="5">
#                     <Label value="Hello" background="blue"/>
#                     <Label value="World" background="pink"/>
#                 </RectangleLabels>
#             </View>
#         </View>
#     </View>
# </View>
#                         """

#     interfaces = [
#         "panel",
#         "controls",
#         "side-column"

#     ],

#     user = {
#         'pk': 1,
#         'firstName': "Zhen Hao",
#         'lastName': "Chu"
#     },

#     task = {
#         'annotations': [],
#         'predictions': [],
#         'id': 1,
#         'data': {
#             'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
#         }
#     }
