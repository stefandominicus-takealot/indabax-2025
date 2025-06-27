import tensorflow as tf
import tensorflow_transform as tft

from recommender_systems.features import (
    CustomerFeatures,
    MetaFeatures,
    ProductFeatures,
    ReviewFeatures,
    xf,
)


def preprocessing_fn(inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    outputs = {}

    # Transform Categorical Features
    for feature in [
        CustomerFeatures.ID,
        ProductFeatures.ID,
        ProductFeatures.BRAND,
        # ReviewFeatures.ID,  # Do we need to transform this?
        # ReviewFeatures.RATING,  # Should this be categorical or numerical?
    ]:
        outputs[xf(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature], vocab_filename=feature
        )

    # Transform Numerical Features
    for feature in [
        ReviewFeatures.RATING,  # TODO: Should this be standardised or normalized?
    ]:
        outputs[xf(feature)] = tft.scale_to_0_1(inputs[feature])

    # Transform Text Features
    for feature in [
        ProductFeatures.TITLE,
        ReviewFeatures.TEXT,
    ]:
        pass  # TODO

    # Construct Meta Features
    outputs[MetaFeatures.RETRIEVAL_SAMPLE_WEIGHT] = tf.cast(
        tf.greater_equal(inputs[ReviewFeatures.RATING], 4), tf.float32
    )

    return outputs
