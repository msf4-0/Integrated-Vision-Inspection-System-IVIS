import inspect
import json
import os
from typing import Any, Dict, List, Union
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from pycocotools.coco import COCO
import streamlit as st
from streamlit import session_state

# >>>> User-defined Modules >>>>
from path_desc import (CLASSIF_MODELS_NAME_PATH, SEGMENT_MODELS_TABLE_PATH,
                       TFOD_MODELS_TABLE_PATH)
from core.utils.log import logger


@st.experimental_memo
def get_pretrained_model_details(deployment_type: str, for_display: bool = False) -> pd.DataFrame:
    """Get the model details from the CSV files scraped from their websites,
    based on the `deployment_type`.

    Args:
        deployment_type (str): obtained from `Project.deployment_type` attribute
        for_display (bool, optional): True to drop irrelevant columns for displaying to the users. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with pretrained model details such as Model Name.
    """
    if deployment_type == "Image Classification":
        # this df has columns: Model Name
        models_df = pd.read_csv(CLASSIF_MODELS_NAME_PATH)
    elif deployment_type == "Object Detection with Bounding Boxes":
        # this df has columns: Model Name, Speed (ms), COCO mAP, Outputs, model_links
        models_df = pd.read_csv(TFOD_MODELS_TABLE_PATH)
        if for_display:
            # `model_links` will be required for downloading the pretrained models for TFOD
            models_df.drop(columns='model_links', inplace=True)
    elif deployment_type == "Semantic Segmentation with Polygons":
        # this df has columns: model_func, Model Name, Reference, links
        # The `Reference` contains names of the authors
        # The `links` are just the paper's arXiv links
        models_df = pd.read_csv(SEGMENT_MODELS_TABLE_PATH)
        if for_display:
            # `model_func` are the functions required for using the `keras_unet_collection` library
            models_df.drop(columns='model_func', inplace=True)
    return models_df


def get_training_param_from_session_state(delete: bool = False) -> Dict[str, Any]:
    """Get training_param from session_state with keys starting with `param_`,
    then remove the `param_` part from the name and return the training_param `Dict`.
    Then delete the params from session_state if necessary after submission.

    e.g. `{"param_batch_size": 32} -> {"batch_size": 32}`"""
    to_delete = []  # delete from session_state
    training_param = {}
    for k, v in session_state.items():
        if k.startswith('param_'):
            if delete:
                to_delete.append(k)
            # e.g. param_batch_size -> batch_size
            new_key = k.replace('param_', '')
            training_param[new_key] = v
    if delete:
        for k in to_delete:
            # delete the params as they are not needed anymore after submission
            del session_state[k]

    return training_param


@st.experimental_memo
def get_segmentation_model_func2params() -> Dict[str, List[str]]:
    """Get only the model function names that have simpler parameters
    for our training purpose, with their parameters as Dict values.
    These function names are directly used with keras_unet_collection library."""

    """Model functions with their parameters:

    `att_unet_2d`
    ['input_size', 'filter_num', 'n_labels', 'stack_num_down', 'stack_num_up',
    'activation', 'atten_activation', 'attention', 'output_activation', 'batch_norm',
    'pool', 'unpool', 'backbone', 'weights', 'freeze_backbone',
    'freeze_batch_norm', 'name']

    `r2_unet_2d`
    ['input_size', 'filter_num', 'n_labels', 'stack_num_down', 'stack_num_up',
    'recur_num', 'activation', 'output_activation', 'batch_norm',
    'pool', 'unpool', 'name']
    
    `resunet_a_2d`
    ['input_size', 'filter_num', 'dilation_num', 'n_labels', 'aspp_num_down',
    'aspp_num_up', 'activation', 'output_activation', 'batch_norm', 'pool',
    'unpool', 'name']

    `unet_2d`
    ['input_size', 'filter_num', 'n_labels', 'stack_num_down', 'stack_num_up', 
    'activation', 'output_activation', 'batch_norm', 'pool', 'unpool', 
    'backbone', 'weights', 'freeze_backbone', 'freeze_batch_norm', 'name']

    `unet_plus_2d`
    ['input_size', 'filter_num', 'n_labels', 'stack_num_down', 'stack_num_up', 
    'activation', 'output_activation', 'batch_norm', 'pool', 'unpool', 'deep_supervision',
    'backbone', 'weights', 'freeze_backbone', 'freeze_batch_norm', 'name']
    """
    from keras_unet_collection import models
    model_func2params = {}
    for func_name in dir(models):
        # not using the Transformer model as it doesn't work for new NumPy version,
        # according to the library's repo https://github.com/yingkaisha/keras-unet-collection
        if func_name.endswith('2d') and func_name != 'transunet_2d':
            signature = inspect.signature(getattr(models, func_name))
            parameters = list(signature.parameters.keys())
            # only take the models that have "filter_num" and "stack_num_up" params
            # for simpler use case; also ResUnet-a as it is quite new and good
            if set(('filter_num', 'stack_num_up')).issubset(parameters) \
                    or (func_name == 'resunet_a_2d'):
                model_func2params[func_name] = parameters
    return model_func2params


@st.experimental_memo
def get_segmentation_model_name2func() -> Dict[str, str]:
    """Return a Dict of Model Name -> Model function name to be able to 
    use it with keras_unet_collection.models"""
    df = get_pretrained_model_details(
        "Semantic Segmentation with Polygons")
    df.set_index('Model Name', inplace=True)
    model_name2func = df['model_func'].to_dict()
    return model_name2func
