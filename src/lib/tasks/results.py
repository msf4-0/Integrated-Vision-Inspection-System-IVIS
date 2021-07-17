"""
To Obtain Annotations Results
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[3]  # ROOT folder -> ./src
LIB_PATH = SRC / "lib"
TEST_MODULE_PATH = SRC / "test" / "test_page" / "module"
LS_PATH = Path(__file__).resolve().parent

if str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))  # ./lib
else:
    pass

if str(TEST_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_MODULE_PATH))
else:
    pass
if str(LS_PATH) not in sys.path:
    sys.path.insert(0, str(LS_PATH))
else:
    pass

import streamlit as st
from frontend.streamlit_labelstudio import st_labelstudio
from annotation.annotation_manager import Results, Annotations

interfaces = [
    "panel",
    "update",
    "submit",
    "controls",
    "side-column",
    "annotations:menu",
    "annotations:add-new",
    "annotations:delete",
    "predictions:menu",
],

#----------Image Classification----------------#


def ImgClassification(config, user, task, interfaces=interfaces, key="ImgClassification"):
    """Obtain annotation results for Image Classification

    Args:
        config (str): Annotation type React JSX DOM
        user (Dict):
                    user = {
                                'pk': 1,
                                'firstName': "James",
                                'lastName': "Dean"
                            },
        task (Dict):
                    task = {
                            'annotations': [],
                            'predictions': [],
                            'id': 1,
                            'data': {
                                # 'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
                                'image': f'{data_url}'
                                }
                            }
        interfaces (List,default): Annotation Tools. Defaults to interfaces.
        key (str, optional): Key for each component. Defaults to "ImgClassification".

    Returns:
        Dict: Annotation ID, Choices
    """

    results_raw = st_labelstudio(config, interfaces, user, task, key)
    st.write(results_raw)
    # flag = results_raw[1]
    if results_raw is not None:
        areas = [v for k, v in results_raw[0]['areas'].items()]

        results = []
        results_display = []
        for a in areas:
            results_display.append(
                {'id': a['id'], 'choices': a['results'][0]['value']['choices'][0]})

            results = a['results'][0]  # store results based on LS format
            st.write(results)

        with st.beta_expander('Show Annotation Log'):

            st.table(results_display)
            st.write(results_raw[0]['areas'])
        # st.write(f"Flag: {flag}")

        # # TODO
        # if flag == 0:  # Submit /INSERT
        #     # INSERT function
        #     x = 1
        # elif flag == 1:
        #     # UPDATE function
        #     y = 1

        flag = results_raw[3]
        return results, flag
    else:
        results, flag = None


#----------Object Detection with Bounding Boxes----------------#


def DetectionBBOX(config, user, task, interfaces=interfaces, key="BBox"):
    """Obtain annotation results for Object Detection with Bounding Boxes

    Args:
        config (str): Annotation type React JSX DOM
        user (Dict):
                    user = {
                                'pk': 1,
                                'firstName': "James",
                                'lastName': "Dean"
                            },
        task (Dict):
                    task = {
                            'annotations': [],
                            'predictions': [],
                            'id': 1,
                            'data': {
                                # 'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
                                'image': f'{data_url}'
                                }
                            }
        interfaces (List,default): Annotation Tools. Defaults to interfaces.
        key (str, optional): Key for each component. Defaults to "BBox".

    Returns:
        Dict: Annotation ID, x, y, width, height, rectanglelabels
    """
    results_raw = st_labelstudio(config, interfaces, user, task, key)
    st.write(results_raw)

    if results_raw is not None:
        areas = [v for k, v in results_raw[0]['areas'].items()]
        canvas_width, canvas_height = [
            canvas_dim for canvas_dim in results_raw[1]]
        st.write(canvas_width, canvas_height)
        original_width, original_height = [
            img_dim for img_dim in results_raw[2]]
        st.write(original_width, original_height)
        results = []  # array to hold dictionary of 'result'
        results_display = []
        for a in areas:
            relativeX = (a['x'] / canvas_width) * 100
            relativeY = (a['y'] / canvas_height) * 100

            results_display.append({'id': a['id'], 'x': relativeX, 'y': relativeY, 'width': a['width'],
                                    'height': a['height'], 'rectanglelabels': a['results'][0]['value']['rectanglelabels'][0]})
            bbox_results = {'x': relativeX, 'y': relativeY, 'width': a['width'],
                            'height': a['height']}  # store current bbox results:x,y,w,h
            results_temp = a['results'][0]  # incomplete results dictionary
            # include bbox results into key:'value'
            results_temp['value'].update(bbox_results)
            results_temp.update(original_width=original_width,
                                original_height=original_height)

            results.append(results_temp)
            st.write("### Results")
            st.write(results)
        with st.beta_expander('Show Annotation Log'):

            st.table(results_display)
            st.write(results_raw[0]['areas'])
        flag = results_raw[3]
        return results, flag
    else:
        results, flag = None

#----------Semantic Segmentation with Polygons----------------#


def SemanticPolygon(config, user, task, original_width, original_height, interfaces=interfaces, key="Polygons"):
    """Obtain annotation results for Semantic Segmentation with Polygons

    Args:
        config (str): Annotation type React JSX DOM
        user (Dict):
                    user = {
                                'pk': 1,
                                'firstName': "James",
                                'lastName': "Dean"
                            },
        task (Dict):
                    task = {
                            'annotations': [],
                            'predictions': [],
                            'id': 1,
                            'data': {
                                # 'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
                                'image': f'{data_url}'
                                }
                            }
        interfaces (List,default): Annotation Tools. Defaults to interfaces.
        key (str, optional): Key for each component. Defaults to "Polygons".

    Returns:
        Dict: Annotation ID, x, y, width, height, polygonlabels
    """

    results_raw = st_labelstudio(config, interfaces, user, task, key)
    st.write(results_raw)

    if results_raw is not None:
        # contains all annotation results
        areas = [v for _, v in results_raw[0]['areas'].items()]

        results = []  # array to hold dictionary of 'result'
        points = []  # List to temp store points for each 'a'
        results_display = []
        for a in areas:

            points = []  # List to temp store points for each 'a'
            results_temp = a['results'][0]  # incomplete results dictionary
            for p in a["points"]:
                results_display.append({'id': p['id'], 'x': p['x'], 'y': p['y'],
                                        'polygonlabels': a['results'][0]['value']['polygonlabels'][0]})
                points.append([p['relativeX'], p['relativeY']])
            results_temp.update(original_width=original_width,
                                original_height=original_height)
            results_temp['value'].update(points=points)
            results.append(results_temp)

            with st.beta_expander("Points"):
                st.write(points)
        col1, col2 = st.beta_columns(2)

        with col1.beta_expander('Show Annotation Log'):

            st.table(results_display)
            st.write(results_raw[0]['areas'])
        with col2.beta_expander('Show Results in LS Format'):

            st.table(results)
            st.write(results)

        flag = results_raw[3]
        return results, flag
    else:
        results, flag = None

#----------Semantic Segmentation with Masks----------------#


def SemanticMask(config, user, task, original_width, original_height, interfaces=interfaces, key="Mask"):
    """Obtain annotation results for Semantic Segmentation with Masks

        Args:
        config (str): Annotation type React JSX DOM
        user (Dict):
                    user = {
                                'pk': 1,
                                'firstName': "James",
                                'lastName': "Dean"
                            },
        task (Dict):
                    task = {
                            'annotations': [],
                            'predictions': [],
                            'id': 1,
                            'data': {
                                # 'image': "https://htx-misc.s3.amazonaws.com/opensource/label-studio/examples/images/nick-owuor-astro-nic-visuals-wDifg5xc9Z4-unsplash.jpg"
                                'image': f'{data_url}'
                                }
                            }
        interfaces (List,default): Annotation Tools. Defaults to interfaces.
        key (str, optional): Key for each component. Defaults to "Mask".

    Returns:
        Dict: Annotation ID, points, brushlabels
    """

    results_raw = st_labelstudio(config, interfaces, user, task, key)
    st.write(results_raw)

    if results_raw is not None:
        areas = [v for k, v in results_raw[0]['areas'].items()]

        results = []
        for a in areas:
            st.write(a["touches"])
            for t in a["touches"]:

                results.append({'id': t['id'], 'points': t["points"],
                               'brushlabels': a['results'][0]['value']['brushlabels'][0]})
        with st.beta_expander('Show Annotation Log'):

            st.table(results)
            st.write(results_raw[0]['areas'])

        flag = results_raw[3]
        return results, flag
    else:
        results, flag = None
