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
from core.utils.log import log_info, log_error

_RELEASE = True


if not _RELEASE:
    _component_func = components.declare_component(
        "label_studio_editor",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "label_studio_editor", path=build_dir)


def labelstudio_editor(
        config: str,
        interfaces: List[str],
        user: dict,
        task: dict,
        key=None,
        on_change: Optional[Callable] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None
) -> List:
    component_value = _component_func(
        config=config, interfaces=interfaces, user=user, task=task, key=key, default=[])
    if component_value:
        if component_value[0]:
            log_info(f"From LS Component: Flag {component_value[1]}")

        if on_change:
            log_info(f"Inside LS Editor Callback")
            wildcard = args if args else kwargs
            if args or kwargs:
                check_args_kwargs(wildcards=wildcard, func=on_change)
            if args:
                on_change(*args)
            elif kwargs:
                on_change(**kwargs)
            else:
                on_change()
         

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

#     results_raw = labelstudio_editor(
#         config, interfaces, user, task, key='random')
#     st.write(results_raw)
    # if results_raw is not None:
    #     areas = [v for k, v in results_raw['areas'].items()]

    #     results = []
    #     for a in areas:
    #         results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
    #                         'height': a['height'], 'label': a['results'][0]['value']['rectanglelabels'][0]})
    #     with st.beta_expander('Show Annotation Log'):

    #         st.table(results)
