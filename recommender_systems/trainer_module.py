from pathlib import Path
from typing import cast

import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs
from absl import logging
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform import TFTransformOutput
from tfx.utils import io_utils
from tfx.v1.components import DataAccessor
from tfx_bsl.tfxio.dataset_options import TensorFlowDatasetOptions

from recommender_systems.features import (
    CustomerFeatures,
    MetaFeatures,
    ProductFeatures,
    xf,
)
from tfx_tfrs.fn_args_utils import FnArgs

BATCH_SIZE = 1024
EPOCHS = 8
EMBEDDING_DIMENSION = 64
FEATURE_EMBEDDING_DIMENSION = 32


def _get_feature_cardinality(
    tf_transform_output: TFTransformOutput, feature_name: str, num_oov_buckets: int = 0
) -> int:
    """Calculate feature cardinality (vocabularies don't include OOV allowances)"""
    assert num_oov_buckets >= 0
    cardinality = (
        tf_transform_output.vocabulary_size_by_name(feature_name) + num_oov_buckets
    )
    logging.info(
        "%s cardinality is %s (including %s OOV buckets)",
        feature_name,
        format(cardinality, ","),
        format(num_oov_buckets, ","),
    )
    return cardinality


def _get_categorical_feature_encoder(
    tf_transform_output: TFTransformOutput, feature_name: str, output_dim: int
) -> keras.Model:
    input_dim = _get_feature_cardinality(
        tf_transform_output=tf_transform_output, feature_name=feature_name
    )
    return keras.Sequential(
        layers=[
            # Strip away the wrapping dimension (TF Record convention)
            keras.layers.Reshape(
                target_shape=(), input_shape=(1,), name=f"{feature_name}_reshape"
            ),
            # TF Transform vocabularies are 0-based, with a default value of -1
            # Since Keras Embedding supports `mask_zero`, we increment the input by 1
            keras.layers.Lambda(
                function=lambda x: tf.add(x, 1), name=f"{feature_name}_add_one"
            ),
            # Convert input indices to embedding vectors
            keras.layers.Embedding(
                input_dim=input_dim + 1,  # Account for 'add one' above
                output_dim=output_dim,
                mask_zero=True,
                name=f"{feature_name}_embedding",
            ),
        ],
        name=f"{feature_name}_encoder",
    )


def _get_real_feature_encoder(feature_name: str, output_dim: int) -> keras.Model:
    return keras.Sequential(
        layers=[
            keras.layers.Dense(
                output_dim, input_shape=(1,), name=f"{feature_name}_dense"
            )
        ],
        name=f"{feature_name}_encoder",
    )


def _get_text_feature_encoder(feature_name: str, output_dim: int) -> keras.Model:
    return keras.Sequential(
        layers=[
            keras.layers.Dense(
                output_dim, input_shape=(128,), name=f"{feature_name}_dense"
            )
        ],
        name=f"{feature_name}_encoder",
    )


class CustomerTower(keras.Model):
    def __init__(
        self,
        tf_transform_output: TFTransformOutput,
        dropout_rate: float = 0.0,
    ):
        super().__init__(name=self.__class__.__name__)
        self._feature_encoders = {
            xf(CustomerFeatures.ID): _get_categorical_feature_encoder(
                tf_transform_output=tf_transform_output,
                feature_name=CustomerFeatures.ID,
                output_dim=FEATURE_EMBEDDING_DIMENSION,
            ),
        }
        self._common = keras.Sequential(
            layers=[keras.layers.Concatenate()], name="common"
        )
        # if dropout_rate:
        #     self._common.add(keras.layers.Dropout(rate=dropout_rate))
        # self._common.add(keras.layers.Dense(units=EMBEDDING_DIMENSION))
        self._common.add(keras.layers.UnitNormalization())

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        # Encode each feature
        encoded_features = [
            feature_encoder(inputs[feature_name])
            for feature_name, feature_encoder in self._feature_encoders.items()
        ]
        # Pass encoded features through the common layers
        return self._common(encoded_features)


class ProductTower(keras.Model):
    def __init__(
        self,
        tf_transform_output: TFTransformOutput,
        dropout_rate: float = 0.0,
    ):
        super().__init__(name=self.__class__.__name__)
        self._feature_encoders = {
            xf(ProductFeatures.ID): _get_categorical_feature_encoder(
                tf_transform_output=tf_transform_output,
                feature_name=ProductFeatures.ID,
                output_dim=FEATURE_EMBEDDING_DIMENSION,
            ),
        }
        self._common = keras.Sequential(
            layers=[keras.layers.Concatenate()], name="common"
        )
        # if dropout_rate:
        #     self._common.add(keras.layers.Dropout(rate=dropout_rate))
        # self._common.add(keras.layers.Dense(units=EMBEDDING_DIMENSION))
        self._common.add(keras.layers.UnitNormalization())

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        # Encode each feature
        encoded_features = [
            feature_encoder(inputs[feature_name])
            for feature_name, feature_encoder in self._feature_encoders.items()
        ]
        # Pass encoded features through the common layers
        return self._common(encoded_features)


class Recommender(tfrs.Model):
    _OUTPUT_KEY_PRODUCT_IDS = "product_ids"

    def __init__(
        self,
        tf_transform_output: TFTransformOutput,
        customer_tower: keras.Model,
        product_tower: keras.Model,
        ranking_task: tfrs.tasks.Ranking | None = None,
        retrieval_task: tfrs.tasks.Retrieval | None = None,
        ranking_loss_weight: float = 1.0,
        retrieval_loss_weight: float = 1.0,
    ):
        super().__init__(name=self.__class__.__name__)
        self._raw_feature_spec = tf_transform_output.raw_feature_spec()
        self._transform_features_layer = tf_transform_output.transform_features_layer()

        self._customer_tower = customer_tower
        self._product_tower = product_tower

        self._ranking_task = ranking_task
        self._retrieval_task = retrieval_task
        assert self._ranking_task or self._retrieval_task, (
            "At least one task must be provided."
        )

        self._ranking_loss_weight = ranking_loss_weight
        self._retrieval_loss_weight = retrieval_loss_weight
        assert self._ranking_loss_weight > 0, "`ranking_loss_weight` must be positive."
        assert self._retrieval_loss_weight > 0, (
            "`retrieval_loss_weight` must be positive."
        )

        self._product_index: tfrs.layers.factorized_top_k.TopK | None = None

    def call(
        self, inputs: dict[str, tf.Tensor], training: bool | None = None
    ) -> tuple[tf.Tensor]:
        return (
            self._customer_tower(inputs, training=training),
            self._product_tower(inputs, training=training),
        )

    def compute_loss(
        self, inputs: dict[str, tf.Tensor], training: bool | None = None
    ) -> tf.Tensor:
        # Compute embeddings
        customer_embedding, product_embedding = self(inputs, training=training)

        # Compute task losses
        ranking_loss = retrieval_loss = 0
        if self._ranking_task:
            pass  # TODO[FUTURE]
        if self._retrieval_task:
            retrieval_loss = self._retrieval_task(
                query_embeddings=customer_embedding,
                candidate_embeddings=product_embedding,
                sample_weight=tf.squeeze(
                    inputs[MetaFeatures.RETRIEVAL_SAMPLE_WEIGHT], axis=-1
                ),
                candidate_ids=tf.squeeze(inputs[xf(ProductFeatures.ID)], axis=-1),
                compute_metrics=not training,
                compute_batch_metrics=not training,
            )

        # Combine task losses
        return (
            ranking_loss * self._ranking_loss_weight
            + retrieval_loss * self._retrieval_loss_weight
        )

    def index_products(self, candidates: tf.data.Dataset) -> None:
        if not self._product_index:
            self._product_index = tfrs.layers.factorized_top_k.ScaNN()
        self._product_index.index_from_dataset(candidates=candidates)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="example")]
    )
    def evaluate_products_for_customer(
        self, example: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        inputs = tf.io.parse_example(example, self._raw_feature_spec)
        return self.predict_products_for_customer(
            {CustomerFeatures.ID: inputs[CustomerFeatures.ID]}
        )

    @tf.function(
        input_signature=[
            {
                CustomerFeatures.ID: tf.TensorSpec(
                    shape=(None, 1), dtype=tf.string, name=CustomerFeatures.ID
                )
            }
        ]
    )
    def predict_products_for_customer(
        self, inputs: dict[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        # Transform raw input features
        transformed_features = self._transform_features_layer(inputs)

        # Compute customer embedding
        customer_embedding = self._customer_tower(transformed_features)

        # Lookup nearest neighbours
        _, identifiers = self._product_index.call(queries=customer_embedding)

        # Return product_line_ids
        return {self._OUTPUT_KEY_PRODUCT_IDS: identifiers}

    def save(self, filepath: str) -> None:
        if not self._product_index:
            raise ValueError(
                "Product index is not built yet. Call `index_products` first."
            )

        # The retrieval task's factorized metrics are not serializable, so we remove it
        # before saving
        self._retrieval_task.factorized_metrics = []

        # Save the model with custom signatures
        cast(keras.Model, super()).save(
            filepath=filepath,
            signatures={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.predict_products_for_customer,
                "evaluate_products_for_customer": self.evaluate_products_for_customer,
            },
        )


def get_tf_dataset_from_examples(
    data_accessor: DataAccessor, file_pattern: list[str], schema: schema_pb2.Schema
) -> tf.data.Dataset:
    return (
        data_accessor.tf_dataset_factory(
            file_pattern=file_pattern,
            options=TensorFlowDatasetOptions(batch_size=1, num_epochs=1, shuffle=False),
            schema=schema,
        )
        .unbatch()
        .cache()
    )


def run_fn(fn_args: FnArgs) -> None:
    # Transform output
    tf_transform_output = TFTransformOutput(fn_args.transform_graph_path)

    # Training dataset
    logging.info("Loading training dataset.")
    train_ds = get_tf_dataset_from_examples(
        data_accessor=fn_args.data_accessor,
        file_pattern=fn_args.train_files,
        schema=tf_transform_output.transformed_metadata.schema,
    ).batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)

    # Validation dataset
    logging.info("Loading validation dataset.")
    validation_ds = get_tf_dataset_from_examples(
        data_accessor=fn_args.data_accessor,
        file_pattern=fn_args.eval_files,
        schema=tf_transform_output.transformed_metadata.schema,
    ).batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)

    # Products dataset
    # NOTE: This dataset is used for indexing, not training
    logging.info("Retrieving unique products.")
    products_ds = get_tf_dataset_from_examples(
        data_accessor=fn_args.item_data_accessor,
        file_pattern=fn_args.item_files,
        schema=io_utils.SchemaReader().read(schema_path=fn_args.item_schema_path),
    ).batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)

    # Define the two towers and the overall model
    logging.info("Building customer tower.")
    customer_tower = CustomerTower(tf_transform_output=tf_transform_output)

    logging.info("Building product tower.")
    product_tower = ProductTower(tf_transform_output=tf_transform_output)

    logging.info("Building overall model.")
    ranking_task = None  # TODO[FUTURE]
    retrieval_task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=(
                products_ds
                # MAP {product_features}
                # -> {transformed_product_features}
                .map(
                    tf_transform_output.transform_features_layer(),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                # Cache static transformations
                .cache()
                # MAP {transformed_product_features}
                # -> (product_id_xf, product_embedding)
                # NOTE: Do not cache afterwards! `product_tower` is updated after each
                # training step.
                .map(
                    lambda transformed_features: (
                        tf.squeeze(
                            transformed_features[xf(ProductFeatures.ID)],
                            axis=-1,
                        ),
                        product_tower(transformed_features),
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .prefetch(buffer_size=tf.data.AUTOTUNE)
            ),
            ks=(10, 100),
        ),
        remove_accidental_hits=True,
        num_hard_negatives=128,
    )
    recommender = Recommender(
        tf_transform_output=tf_transform_output,
        customer_tower=customer_tower,
        product_tower=product_tower,
        ranking_task=ranking_task,
        retrieval_task=retrieval_task,
    )
    recommender = cast(
        Recommender | keras.Model, recommender
    )  # Work around tf.keras lazy loading
    recommender.compile(optimizer=keras.optimizers.Adam())

    # Define callbacks
    callbacks = []
    early_stopping_monitor = "val_factorized_top_k/top_10_categorical_accuracy"
    early_stopping_min_delta = 1e-3
    early_stopping_patience = 3
    early_stopping_start_from_epoch = 3
    logging.info(
        "Creating %s callback (monitoring %s, min_delta %.3f, patience %d, "
        "start_from_epoch %d)",
        keras.callbacks.EarlyStopping.__name__,
        early_stopping_monitor,
        early_stopping_min_delta,
        early_stopping_patience,
        early_stopping_start_from_epoch,
    )
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=early_stopping_monitor,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            verbose=1,  # displays messages when the callback takes an action
            restore_best_weights=True,
            start_from_epoch=early_stopping_start_from_epoch,
        )
    )
    tensorboard_log_dir = fn_args.custom_config.get(
        "tensorboard_log_dir",
        Path(fn_args.model_run_dir)
        / "callbacks"
        / keras.callbacks.TensorBoard.__name__,
    )
    logging.info(
        "Creating %s callback (logging to %s)",
        keras.callbacks.TensorBoard.__name__,
        tensorboard_log_dir,
    )
    callbacks.append(
        keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), histogram_freq=1)
    )

    # Run the training loop
    logging.info("Starting training loop (epochs=%d).", EPOCHS)
    recommender.fit(
        train_ds.prefetch(buffer_size=tf.data.AUTOTUNE),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE),
    )

    # Save the model
    logging.info("Indexing products for retrieval.")
    recommender.index_products(
        candidates=(
            products_ds
            # MAP {product_features}
            # -> (product_id, {transformed_product_features})
            .map(
                lambda features: (
                    features[ProductFeatures.ID],
                    tf_transform_output.transform_features_layer()(features),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            # MAP (product_id, {transformed_product_features})
            # -> (product_id, product_embedding)
            .map(
                lambda product_id, transformed_features: (
                    tf.squeeze(product_id, axis=-1),
                    product_tower(transformed_features),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )
    )

    logging.info("Saving the model to %s", fn_args.serving_model_dir)
    recommender.save(fn_args.serving_model_dir)
