import copy
import shutil
import subprocess
import pprint
from pathlib import Path
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import streamlit as st
from streamlit import session_state
from streamlit_tensorboard import st_tensorboard

from path_desc import TFOD_DIR

from .visuals import pretty_st_metric, PrettyMetricPrinter
from core.utils.log import logger


def run_tensorboard(logdir: Path, port: int = 6007, width: int = 1080):
    """Run and show TensorBoard interface as a Streamlit component"""
    # TODO: test whether this TensorBoard works after deployed the app
    logger.info(f"Running TensorBoard on {logdir}")
    # NOTE: this st_tensorboard does not work if the path passed in
    #  is NOT in POSIX format, thus the `as_posix()` method to convert
    #  from WindowsPath to POSIX format to work in Windows
    st_tensorboard(logdir=logdir.as_posix(),
                   port=port, width=width, scrolling=True)


def find_tfod_metric(name: str, cmd_output: str) -> Tuple[str, Union[int, float]]:
    """
    Find the specific metric name in the command output (`cmd_output`)
    and returns only the digits (i.e. values) of the metric.

    Basically search using regex groups, then take the last match,
    then take the first group for the metric name, and the second group for the digits.
    """
    assert name in ('Step', 'Loss')
    try:
        if name == 'Step':
            value = re.findall(f'({name})\s+(\d+)', cmd_output)[-1][1]
            return name, int(value)
        else:
            loss_name, value = re.findall(
                f'{name}/(\w+).+(\d+\.\d+)', cmd_output)[-1]
            return loss_name, float(value)
    except IndexError:
        logger.debug(f"Value for '{name}' not found from {cmd_output}")


def find_tfod_eval_metrics(cmd_output) -> str:
    """
    Find the specific evaluation metric names in the command output (`cmd_output`)
    and returns the concatenated lines of strings to display to the user on Streamlit.
    """
    all_lines = []
    metric_names = ('DetectionBoxes_Precision/',
                    'DetectionBoxes_Recall/', 'Loss/')
    try:
        for name in metric_names:
            lines = re.findall(f"{name}.+", cmd_output)
            # remove duplicated lines
            lines = list(set(lines))
            for line in lines:
                found_name, val = line.split(":")
                line = f"**{found_name}**: {val}"
                all_lines.append(line)
        all_lines_str = '  \n'.join(all_lines)
        return all_lines_str
    except Exception as e:
        logger.error(
            f"Error with '{name}' from command output: {cmd_output}\n{e}")


def find_traceback(cmd_output: str) -> str:
    """Find traceback texts from command STDOUT output"""
    traceback = re.findall('\nTraceback.+', cmd_output, flags=re.DOTALL)
    traceback = '\n'.join(traceback).strip()
    return traceback


# ! DEPRECATED
def check_process_returncode_old(process: subprocess.Popen,
                                 returncode: int):
    """Check returncode from subprocess. `returncode` of 0 means success.

    On Windows: `returncode` is always 1 if process is terminated or error occurred.
    On POSIX: `returncode` is also 1 but negative when process is terminated due to a signal
      such as through `process.terminate()`.
    """
    if returncode > 0:
        logger.debug(f"{returncode = }")
        cmd_output = process.stderr.read()
        traceback = find_traceback(cmd_output)
        if traceback:
            logger.error("ERROR occurred when running the command, Traceback:\n"
                         f"{traceback}")
            st.error(
                "Error running the command, please try again or contact our admin.")
            st.stop()
        else:
            logger.debug(
                "No traceback found, the process is probably terminated prematurely.")


def check_process_returncode(returncode: int, traceback: List[str]) -> Union[str, None]:
    """Check returncode from subprocess. `returncode` of 0 means success.

    On Windows: `returncode` is always 1 if process is terminated or error occurred.
    On POSIX: `returncode` is also 1 but negative when process is terminated due to a signal
      such as through `process.terminate()`.
    """
    if returncode > 0 and traceback:
        traceback = '\n'.join(traceback)
        logger.error(f"Error occurred with traceback:\n{traceback}")
        return traceback


def run_command(command_line_args: str, st_output: bool = False,
                stdout_output: bool = True,
                filter_by: Optional[List[str]] = None,
                pretty_print: Optional[bool] = False,
                is_cocoeval: Optional[bool] = False
                ) -> Union[str, None]:
    """
    Running commands or scripts, while getting live stdout

    Args:
        command_line_args (str): Command line arguments to run.
        st_output (bool, optional): Set `st_output` to True to
            show the console outputs LIVE on Streamlit. Defaults to False.
        stdout_output (bool, optional): Set `stdout_output` to True to
            show the console outputs LIVE on terminal. Defaults to True.
        filter_by (Optional[List[str]], optional): Provide `filter_by` to
            filter out other strings and show the `filter_by` strings on Streamlit app. Defaults to None.
    Returns:
        str: The entire console output from the command after finish running.
    """
    if pretty_print:
        command_line_args = pprint.pformat(command_line_args)
    logger.info(f"Running command: '{command_line_args}'")
    logger.debug(f"{stdout_output = }")
    # shell=True to work on String instead of list -- not really sure about this...
    # using shell=False for COCO eval to easily stop the process with process.terminate()
    shell = False if is_cocoeval else True
    process = subprocess.Popen(command_line_args, shell=shell,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)
    if filter_by:
        output_str_list = []
    traceback_found = False
    traceback = []
    for line in process.stdout:
        # remove empty trailing spaces, and also string with only spaces
        line = line.strip()
        if line:
            if stdout_output:
                print(line)
            if 'Traceback' in line:
                traceback_found = True
            if traceback_found:
                traceback.append(line)
                continue
            if is_cocoeval and 'Loss/total_loss' in line:
                # stop the process after see this line in COCO eval script
                logger.debug("NOTE: TERMINATING COCO EVAL SCRIPT")
                process.terminate()
            if filter_by:
                for filter_str in filter_by:
                    if filter_str in line:
                        output_str_list.append(line)
                        if st_output:
                            st.markdown(line)
            elif st_output:
                st.markdown(line)

    # wait for the process to terminate and check for any error
    returncode = process.wait()
    traceback = check_process_returncode(returncode, traceback)

    if filter_by and not output_str_list:
        # there is nothing obtained from the filtered stdout
        logger.error("Error getting any filtered text "
                     "from the command outputs!")
        st.error("Some error has occurred during training ..."
                 " Please try again")
        time.sleep(3)
        st.experimental_rerun()
    elif filter_by:
        return '\n'.join(output_str_list)
    return process.stdout.read()


def run_command_update_metrics_old(
    command_line_args: str,
    stdout_output: bool = True,
    step_name: str = 'Step',
    metric_names: Tuple[str] = None,
) -> str:
    # ! DEPRECATED, Using run_command_update_metrics function
    """
    Running commands or scripts, while getting live stdout

    Args:
        command_line_args (str): Command line arguments to run.
        st_output (bool, optional): Set `st_output` to True to
            show the console outputs LIVE on Streamlit. Defaults to False.
        filter_by (Optional[List[str]], optional): Provide `filter_by` to
            filter out other strings and show the `filter_by` strings on Streamlit app. Defaults to None.
    Returns:
        str: The entire console output from the command after finish running.
    """
    assert metric_names is not None, "Please pass in metric_names to use for search and updates"

    logger.info(f"Running command: '{command_line_args}'")
    process = subprocess.Popen(command_line_args, shell=True,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)

    prev_progress = session_state.new_training.progress
    curr_metrics = session_state.new_training.training_model.metrics
    curr_metrics = copy.copy(curr_metrics)
    prev_metrics = copy.copy(curr_metrics)
    for line in process.stdout:
        # remove empty trailing spaces, and also string with only spaces
        line = line.strip()
        if line:
            if step_name in line:
                step_val = find_tfod_metric(step_name, line)
                # only update the `Step` progress if it's different
                if step_val and step_val != prev_progress[step_name]:
                    # update previous step value to be the current
                    prev_progress[step_name] = step_val
                    session_state.new_training.update_progress(
                        prev_progress, verbose=True)
                    st.markdown(f"**{step_name}**: {step_val}")
            for name in metric_names:
                if name in line:
                    metric_val = find_tfod_metric(name, line)
                    if metric_val:
                        curr_metrics[name] = metric_val
                        num_same_values = np.sum(np.isclose(
                            list(curr_metrics.values()),
                            list(prev_metrics.values())))
                        print(f"{num_same_values}")
                        # only update and show the metrics when all values
                        #  are already updated with the latest different metrics
                        if num_same_values == 0:
                            session_state.new_training.update_metrics(
                                curr_metrics, verbose=True)
                            # show the nicely formatted metrics on Streamlit
                            pretty_st_metric(
                                curr_metrics,
                                prev_metrics,
                                float_format='.3g')
                            # update previous metrics to be the current
                            prev_metrics = copy.copy(curr_metrics)
        if line and stdout_output:
            print(line)
    process.wait()
    return process.stdout


def run_command_update_metrics(
    command_line_args: str,
    stdout_output: bool = True,
    step_name: str = 'Step',
    pretty_print: Optional[bool] = False
) -> str:
    """[summary]

    Args:
        command_line_args (str): Command line arguments to run.
        stdout_output (bool, optional): Set `stdout_output` to True to
            show the console outputs LIVE on terminal. Defaults to True.
        step_name (str, optional): The key name used to store our training step progress.
            Should be 'Step' for now. Defaults to 'Step'.

    Returns:
        str: the entire console output generated from the TFOD training script
    """
    # assert metric_names is not None, "Please pass in metric_names to use for search and updates"
    if pretty_print:
        command_line_args = pprint.pformat(command_line_args)
    logger.info(f"Running command: '{command_line_args}'")
    process = subprocess.Popen(command_line_args, shell=True,
                               # stdout to capture all output
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               # text to directly decode the output
                               text=True)

    pretty_metric_printer = PrettyMetricPrinter()
    # list of stored stdout lines
    current_lines = []
    # flag to track the 'Step' text
    step_name_found = False
    # to track any traceback
    traceback_found = False
    traceback = []
    for line in process.stdout:
        # remove empty trailing spaces, and also string with only spaces
        line = line.strip()
        if line:
            if stdout_output:
                # print to console
                print(line)
            if 'Traceback' in line:
                traceback_found = True
            if traceback_found:
                traceback.append(line)
                continue
            # keep storing when stored stdout lines are less than 12,
            #  12 is the exact number of lines output by the script
            #  that will contain all the metrics after the line
            #  with text 'Step', e.g.:
            """
            INFO:tensorflow:Step 900 per-step time 0.561s
            I0813 13:07:28.326545 139687560058752 model_lib_v2.py:700] Step 900 per-step time 0.561s
            INFO:tensorflow:{'Loss/classification_loss': 0.34123567,
            'Loss/localization_loss': 0.0464548,
            'Loss/regularization_loss': 0.24863605,
            'Loss/total_loss': 0.6363265,
            'learning_rate': 0.025333151}
            I0813 13:07:28.326872 139687560058752 model_lib_v2.py:701] {'Loss/classification_loss': 0.34123567,
            'Loss/localization_loss': 0.0464548,
            'Loss/regularization_loss': 0.24863605,
            'Loss/total_loss': 0.6363265,
            'learning_rate': 0.025333151}
            """
            if step_name in line:
                step_name_found = True
            if step_name_found:
                current_lines.append(line)
            if len(current_lines) == 12:
                # take the first line because that's where `step_name` was found and appended first
                _, step_val = find_tfod_metric(step_name, current_lines[0])
                session_state.new_training.progress[step_name] = step_val
                session_state.new_training.update_progress(
                    session_state.new_training.progress, verbose=True)
                st.markdown(f"**{step_name}**: {step_val}")

                metrics = {}
                for line in current_lines[2:]:
                    metric = find_tfod_metric('Loss', line)
                    if metric:
                        metric_name, metric_val = metric
                        metrics[metric_name] = metric_val
                session_state.new_training.update_metrics(
                    metrics, verbose=True)

                # show the nicely formatted metrics on Streamlit
                pretty_metric_printer.write(metrics)
                st.markdown("___")

                # reset the list of outputs to start storing again
                current_lines = []
                step_name_found = False
    # wait for the process to terminate and check for any error
    returncode = process.wait()
    traceback = check_process_returncode(returncode, traceback)
    return process.stdout.read()


def export_tfod_savedmodel(training_paths: Dict[str, Path]) -> bool:
    paths = training_paths
    if paths['export'].exists():
        # remove any existing export directory first
        shutil.rmtree(paths['export'])

    with st.spinner("Exporting TensorFlow Object Detection model ... "
                    "This may take awhile ..."):
        pipeline_conf_path = paths['config_file']
        FREEZE_SCRIPT = TFOD_DIR / 'research' / \
            'object_detection' / 'exporter_main_v2.py '
        command = (f"python {FREEZE_SCRIPT} "
                   "--input_type=image_tensor "
                   f"--pipeline_config_path={pipeline_conf_path} "
                   f"--trained_checkpoint_dir={paths['models']} "
                   f"--output_directory={paths['export']}")
        run_command(command, stdout_output=False)

    if (paths['export'] / 'saved_model').exists():
        logger.info("Successfully exported TensorFlow Object Detection model")
        return True
    else:
        logger.error("Failed to export TensorFlow Object Detection model!")
        return False