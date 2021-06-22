"""
To Obtain Annotations Results
Date: 21/6/2021
Author: Chu Zhen Hao
Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)

"""
import streamlit as st
from frontend.streamlit_labelstudio import st_labelstudio

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

    if results_raw is not None:
        areas = [v for k, v in results_raw['areas'].items()]

        results = []
        for a in areas:
            results.append(
                {'id': a['id'], 'choices': a['results'][0]['value']['choices'][0]})
        with st.beta_expander('Show Annotation Log'):

            st.table(results)
            st.write(results_raw['areas'])

    else:
        results = None

    return results

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
        areas = [v for k, v in results_raw['areas'].items()]

        results = []
        for a in areas:
            results.append({'id': a['id'], 'x': a['x'], 'y': a['y'], 'width': a['width'],
                            'height': a['height'], 'rectanglelabels': a['results'][0]['value']['rectanglelabels'][0]})
        with st.beta_expander('Show Annotation Log'):

            st.table(results)
            st.write(results_raw['areas'])

        return results
    else:
        results = None

#----------Semantic Segmentation with Polygons----------------#


def SemanticPolygon(config, user, task, interfaces=interfaces, key="Polygons"):
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
        areas = [v for k, v in results_raw['areas'].items()]

        results = []
        for a in areas:
            st.write(a["points"])
            for p in a["points"]:
                results.append({'id': p['id'], 'x': p['x'], 'y': p['y'],
                               'polygonlabels': a['results'][0]['value']['polygonlabels'][0]})
        with st.beta_expander('Show Annotation Log'):

            st.table(results)
            st.write(results_raw['areas'])

        return results
    else:
        results = None

#----------Semantic Segmentation with Masks----------------#


def SemanticMask(config, user, task, interfaces=interfaces, key="Mask"):
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
        areas = [v for k, v in results_raw['areas'].items()]

        results = []
        for a in areas:
            st.write(a["touches"])
            for t in a["touches"]:

                results.append({'id': t['id'], 'points': t["points"],
                               'brushlabels': a['results'][0]['value']['brushlabels'][0]})
        with st.beta_expander('Show Annotation Log'):

            st.table(results)
            st.write(results_raw['areas'])

        return results
    else:
        results = None
