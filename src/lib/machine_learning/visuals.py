from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import cv2

import streamlit as st

from object_detection.utils import visualization_utils as viz_utils


def pretty_format_param(param_dict: Dict[str, Any], float_format: str = '.5g') -> str:
    """
    Format param_dict to become a nice output to show on Streamlit.
    `float_format` is used for formatting floats.
    The formatting for significant digits `.5g` is based on:
    https://stackoverflow.com/questions/25780022/how-to-make-python-format-floats-with-certain-amount-of-significant-digits
    """
    config_info = []
    for k, v in param_dict.items():
        param_name = ' '.join(k.split('_')).capitalize()
        try:
            param_val = f"{float(v):{float_format}}"
        except ValueError:
            param_val = v
        current_info = f'**{param_name}**: {param_val}'
        config_info.append(current_info)
    config_info = '  \n'.join(config_info)
    return config_info


def pretty_st_metric(
        metrics: Dict[str, Any],
        prev_metrics: Dict[str, Any],
        float_format: str = '.5g'):
    # ! DEPRECATED, use PrettyMetricPrinter class
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        # show green color when loss is reduced;
        # red color when increased
        delta_color = 'inverse'
        # get the previous value before prettifying it
        prev_val = prev_metrics[name]
        # prettifying the metric name for display
        name = ' '.join(name.split('_')).capitalize()
        # calculate the difference with previous metric value
        delta = val - prev_val
        # formatting the float values for display
        val = f"{val:{float_format}}"
        if delta == 0:
            # don't show any indicator if there is no difference, or
            # if it's the initial training metrics
            delta = None
        else:
            delta = f"{delta:{float_format}}"
        col.metric(name, val, delta, delta_color=delta_color)


@dataclass(order=False, eq=False)
class PrettyMetricPrinter:
    """
    Wrapper class for pretty print using [st.metric function](https://docs.streamlit.io/en/stable/api.html#streamlit.metric).

    Args:
        float_format (str | Dict[str, str], optional): the formatting used for floats.
            Can pass in either a `str` to use the same formatting for all metrics, or pass in a `Dict` for different formatting for each metric.
            Defaults to `.5g` for 5 significant figures.
        delta_color (str | Dict[str, str]], optional): Similar to `float_format`, can pass in `str` or `Dict`. 
            Defaults to `inverse` when the metric name contains `loss`, else `normal`.
            Refer to Streamlit docs for the effects on colors.
    """
    float_format: Union[str, Dict[str, str]] = '.5g'
    delta_color: Union[str, Dict[str, str]] = None
    prev_metrics: Dict[str, float] = field(default=None, init=False)

    def write(self, metrics: Dict[str, float]):
        """
        Use this to directly print out the current metrics in a nicely formatted way in columns and st.metric.
        metrics (Dict[str, Any]): The dictionary containing the metrics such as loss or accuracy
        """
        if not self.delta_color:
            self.delta_color = {
                name: 'inverse'
                if 'loss' in name
                else 'normal'
                for name in metrics
            }
        if isinstance(self.float_format, str):
            self.float_format = {name: self.float_format for name in metrics}
        if not self.prev_metrics:
            self.prev_metrics = metrics.copy()

        cols = st.columns(len(metrics))
        for col, (name, val) in zip(cols, metrics.items()):
            # get the current parameters for the metric before updating them
            # and before prettifying the metric name
            delta_color = self.delta_color[name]
            float_format = self.float_format[name]
            prev_val = self.prev_metrics[name]
            # prettifying the metric name for display
            name = ' '.join(name.split('_')).capitalize()
            # calculate the difference with previous metric value
            delta = val - prev_val
            # formatting the float values for display
            val = f"{val:{float_format}}"
            if delta == 0:
                # don't show any indicator if there is no difference, or
                # if it's the initial training metrics
                delta = None
            else:
                delta = f"{delta:{float_format}}"
            # using the st.metric function here
            col.metric(name, val, delta, delta_color=delta_color)

        # updating previous metrics before proceeding
        self.prev_metrics = metrics.copy()


def draw_gt_bbox(
    image_np: np.ndarray,
    box_coordinates: Sequence[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    image_with_gt_box = image_np.copy()
    for xmin, ymin, xmax, ymax in box_coordinates:
        if isinstance(xmin, float):
            xmin, ymin = int(xmin), int(ymin)
            xmax, ymax = int(xmax), int(ymax)
        cv2.rectangle(
            image_with_gt_box,
            (xmin, ymin),
            (xmax, ymax),
            color=color,
            thickness=2)
    return image_with_gt_box


def draw_tfod_bboxes(
        detections: Dict[str, Any],
        image_np: np.ndarray,
        category_index: Dict[int, Any],
        min_score_thresh: float = 0.6) -> np.ndarray:
    """`category_index` is loaded using `load_labelmap` method"""
    label_id_offset = 1  # might need this
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        # detections['detection_classes'] + label_id_offset,
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False
    )
    return image_np_with_detections
