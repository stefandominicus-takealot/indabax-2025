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
    # Only those features explicitly added to `outputs` will be available in the
    # training loop.
    outputs = {}

    # Transform Categorical Features
    for feature in [
        CustomerFeatures.ID,
        ProductFeatures.ID,
        # TODO[IndabaX]: Consider transforming other categorical features
        # ProductFeatures.BRAND,
    ]:
        outputs[xf(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature], vocab_filename=feature
        )

    # Transform Numerical Features
    for feature in [
        ReviewFeatures.RATING,
    ]:
        # TODO[IndabaX]: Consider transforming numerical features
        # Have a look at the various `scale_...` functions included in the Tensorflow
        # Transform library:
        #  - https://www.tensorflow.org/tfx/transform/api_docs/python/tft
        pass

    # Transform Text Features
    for feature in [
        ProductFeatures.TITLE,
        ReviewFeatures.TEXT,
    ]:
        # TODO[IndabaX]: Consider transforming text features
        # There are many ways to transform text features, but using a pre-trained
        # language model is a popular choice. For example, you could use a BERT model
        # from Tensorflow Hub to transform the text features into embeddings.
        # If you want to pursue this, look at:
        #  - https://www.kaggle.com/models/tensorflow/bert
        #  - https://www.tensorflow.org/tfx/transform/api_docs/python/tft/make_and_track_object
        pass

    # Construct Meta Features
    # The Tensorflow Recommenders Retrieval task expects only positive samples during
    # training. While one might argue that all reviews are implicitly positive (the
    # customer chose to buy the product initially), reviews with low ratings definitely
    # represent a negative sentiment. Therefore we construct a RETRIEVAL_SAMPLE_WEIGHT
    # feature that is 1.0 for positive reviews (rating >= 4) and 0.0 for negative
    # reviews. This means that the retrieval task will ignore any negative reviews.
    outputs[MetaFeatures.RETRIEVAL_SAMPLE_WEIGHT] = tf.cast(
        tf.greater_equal(inputs[ReviewFeatures.RATING], 4), tf.float32
    )

    return outputs
