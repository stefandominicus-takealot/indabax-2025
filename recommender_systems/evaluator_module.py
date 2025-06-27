from typing import Any

import keras
import scann  # noqa: F401 # Register ops
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.utils.path_constants import SERVING_MODEL_DIR


class TopKAccuracy(keras.metrics.Mean):
    """
    A watered down version of [tfrs.metrics.FactorizedTopK](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/metrics/FactorizedTopK).
    The TFRS implementation holds the trained embedding state in the metric itself,
    whereas this implementation is stateless, and relies on the model to provide the
    top-k predictions. The former is easy to work with during training, but cannot be
    serialised and re-used in the TFX Evaluator component. [The docs](https://www.tensorflow.org/tfx/model_analysis/faq)
    suggest that this kind of thing should be possible, but our attempts have yet to
    succeed.
    """

    def __init__(
        self, name: str = "top_k_accuracy", dtype: Any | None = None, k: int = 10
    ):
        super().__init__(name=name, dtype=dtype)
        self.k = k

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            tf.cast(
                tf.reduce_any(
                    tf.equal(y_true, y_pred[:, : self.k]), axis=1, keepdims=True
                ),
                dtype=tf.float32,
            ),
            sample_weight=sample_weight,
        )


def custom_eval_shared_model(eval_saved_model_path: str, *args, **kwargs):
    eval_saved_model_path = eval_saved_model_path.removesuffix(SERVING_MODEL_DIR)
    return tfma.default_eval_shared_model(eval_saved_model_path, *args, **kwargs)
