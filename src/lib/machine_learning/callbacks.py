from typing import Dict, Any

import streamlit as st
from streamlit import session_state
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow import keras

from .visuals import PrettyMetricPrinter


K = keras.backend


class StreamlitOutputCallback(Callback):
    def __init__(self, pretty_metric_printer: PrettyMetricPrinter,
                 num_epochs, steps_per_epoch, progress_placeholder: Dict[str, Any],
                 refresh_rate=20):
        self.pretty_metric = pretty_metric_printer
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.refresh_rate = refresh_rate
        self.progress_placeholder = progress_placeholder

    def on_epoch_begin(self, epoch, logs=None):
        with self.progress_placeholder['epoch'].container():
            st.markdown(
                f"**Current epoch {epoch + 1} / {self.num_epochs}**:")
            session_state.current_batch_text = st.empty()

        session_state.batch_progress_bar = (
            self.progress_placeholder['batch'].progress(0)
        )

    def on_batch_end(self, batch, logs=None):
        batch += 1
        if (batch % self.refresh_rate) == 0 or batch == self.steps_per_epoch:
            session_state.current_batch_text.markdown(
                f"Step: {batch} / {self.steps_per_epoch}")
            session_state.batch_progress_bar.progress(
                batch / self.steps_per_epoch)

    def on_epoch_end(self, epoch, logs=None):
        # NOTE: MUST INCLUDE at least one line of Streamlit function here to make
        # Keras interacts with Streamlit
        epoch = epoch + 1  # add one to start from 1 instead of 0
        st.markdown(f"**Epoch {epoch} / {self.num_epochs}**:")

        if 'learning_rate' in logs:
            # not storing learning_rate for now
            lr = logs.pop('learning_rate')
            # lr = float(K.get_value(lr))

        self.pretty_metric.write(logs)
        st.markdown('___')

        session_state.new_training.update_progress({'Epoch': epoch})
        session_state.new_training.update_metrics(logs)


class LRTensorBoard(TensorBoard):
    # using this to log learning rate to TensorBoard
    # https://newbedev.com/keras-how-to-output-learning-rate-onto-tensorboard
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'learning_rate': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
