import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Text, cast

import pandas as pd
import streamlit as st
import tensorflow_model_analysis as tfma
from absl import logging
from google.cloud.storage import Client as GCSClient
from google.cloud.storage.blob import Blob
from ml_metadata.proto import metadata_store_pb2
from tfx import v1 as tfx
from tfx.components import CsvExampleGen, Evaluator, SchemaGen, StatisticsGen
from tfx.dsl.components.common.importer import IMPORT_RESULT_KEY, Importer
from tfx.types.standard_artifacts import Model
from tfx.types.standard_component_specs import (
    EXAMPLES_KEY,
    SCHEMA_KEY,
    STATISTICS_KEY,
)

from recommender_systems import evaluator_module
from recommender_systems.features import ProductFeatures
from recommender_systems.splits import Splits

logging.set_verbosity(logging.INFO)

DATA = Path(__file__).parent / "data"

PROJECT = "tal-deep-learning-indabax"
BUCKET = "tal-deep-learning-indabax-models"

PIPELINE_NAME = "evaluate"
PIPELINE_ROOT = str(Path("pipeline-root") / PIPELINE_NAME)
METADATA_PATH = str(Path(PIPELINE_ROOT) / "metadata.db")
ENABLE_CACHE = True


def gcs_blobs(
    client: GCSClient,
    bucket_or_name: str,
    match_glob: str | None = None,
    prefix: str | None = None,
    max_results: int | None = None,
) -> Iterable[Blob]:
    page_token = None
    while response := client.list_blobs(
        bucket_or_name=bucket_or_name,
        match_glob=match_glob,
        page_token=page_token,
        prefix=prefix,
        max_results=max_results,
    ):
        yield from response
        if (page_token := response.next_page_token) is None:
            break


@dataclass
class ModelInfo:
    MODEL_SUFFIX = "saved_model.pb"

    participant: str
    model_id: str
    generation: int
    uri: str

    @classmethod
    def from_blob(cls, blob: Blob) -> "ModelInfo":
        participant, model_id, _ = cast(str, blob.name).split("/")
        return cls(
            participant=participant,
            model_id=model_id,
            generation=blob.generation,
            uri=f"gs://{blob.bucket.name}/{blob.name.removesuffix(f'/{cls.MODEL_SUFFIX}')}",
        )


def get_participant_models(client: GCSClient, bucket: str):
    participant_models = {}

    for blob in gcs_blobs(client, bucket, match_glob=f"**/{ModelInfo.MODEL_SUFFIX}"):
        model_info = ModelInfo.from_blob(blob)
        participant_models.setdefault(model_info.participant, []).append(model_info)

    return {
        participant: sorted(models, key=lambda m: m.generation)
        for participant, models in participant_models.items()
    }


def get_latest_models(client: GCSClient, bucket: str):
    for _, models in get_participant_models(client, bucket).items():
        if models:
            yield models[-1]


class Leaderboard:
    PATH = "leaderboard.json"

    def __init__(self):
        self.results = self.load()

    def __contains__(self, model: ModelInfo):
        return model.model_id in self.results.get(model.participant, {})

    def update(self, model: ModelInfo, result: dict):
        self.results.setdefault(model.participant, {})[model.model_id] = result
        self.save()

    def dataframe(self):
        leaderboard_metric = "Top K Accuracy"
        if not self.results:
            return pd.DataFrame(columns=["Name", "Model ID", leaderboard_metric])
        dataframe = (
            pd.DataFrame(
                [
                    {
                        "Name": participant,
                        "Model ID": model_id,
                        leaderboard_metric: model["top_k_accuracy"],
                    }
                    for participant, models in self.results.items()
                    for model_id, model in models.items()
                ]
            )
            .groupby("Name", group_keys=True)
            .apply(
                lambda df: df[df[leaderboard_metric] == df[leaderboard_metric].max()]
            )
            .sort_values(leaderboard_metric, ascending=False)
        )
        dataframe.index = dataframe[leaderboard_metric].rank(
            method="dense", ascending=False
        )
        return dataframe

    def load(self):
        try:
            with open(self.PATH, "r") as fp:
                return json.load(fp)
        except FileNotFoundError:
            return {}

    def save(self):
        with open(self.PATH, "w") as fp:
            json.dump(self.results, fp, indent=2)


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    enable_cache: bool,
    model_uri: Text,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
):
    components = []
    reviews_example_gen_component = CsvExampleGen(
        input_base=str(DATA / "reviews"),
        input_config=tfx.proto.Input(
            splits=[tfx.proto.Input.Split(name=Splits.TEST, pattern="test.csv")]
        ),
        output_config=tfx.proto.Output(
            split_config=tfx.proto.SplitConfig(
                splits=[tfx.proto.SplitConfig.Split(name=Splits.TEST, hash_buckets=1)]
            )
        ),
    )
    components.append(reviews_example_gen_component)

    reviews_statistics_gen_component = StatisticsGen(
        examples=reviews_example_gen_component.outputs[EXAMPLES_KEY]
    )
    components.append(reviews_statistics_gen_component)

    reviews_schema_gen_component = SchemaGen(
        statistics=reviews_statistics_gen_component.outputs[STATISTICS_KEY]
    )
    components.append(reviews_schema_gen_component)

    model_importer_component = Importer(
        source_uri=model_uri,
        artifact_type=Model,
        reimport=True,
    )
    components.append(model_importer_component)

    evaluator_component = Evaluator(
        examples=reviews_example_gen_component.outputs[EXAMPLES_KEY],
        model=model_importer_component.outputs[IMPORT_RESULT_KEY],
        example_splits=[Splits.TEST],
        eval_config=tfma.EvalConfig(
            metrics_specs=[
                tfma.MetricsSpec(
                    metrics=[
                        tfma.MetricConfig(
                            class_name="TopKAccuracy",
                            module=evaluator_module.__name__,
                        ),
                    ],
                ),
            ],
            model_specs=[
                tfma.ModelSpec(
                    label_key=ProductFeatures.ID,
                    signature_name="evaluate_products_for_customer",
                ),
            ],
        ),
        schema=reviews_schema_gen_component.outputs[SCHEMA_KEY],
        module_file=evaluator_module.__file__,
    )
    components.append(evaluator_component)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


def run_pipeline(model_uri: Text):
    pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        enable_cache=ENABLE_CACHE,
        model_uri=model_uri,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    tfx.orchestration.LocalDagRunner().run(pipeline)


leaderboard = Leaderboard()


def refresh_leaderboard():
    status_slot = st.empty()
    leaderboard_slot = st.empty()

    leaderboard_df = leaderboard.dataframe()
    leaderboard_slot.dataframe(leaderboard_df)

    leaders = set(leaderboard_df.loc[leaderboard_df.index.intersection([1])]["Name"])

    models_to_refresh = [
        model
        for model in get_latest_models(GCSClient(project=PROJECT), BUCKET)
        if model not in leaderboard
    ]
    if models_to_refresh:
        progress_bar = status_slot.progress(0)
        for index, model in enumerate(models_to_refresh):
            progress_bar.progress(
                index / len(models_to_refresh),
                text=f"Evaluating {model.participant}/{model.model_id}...",
            )
            run_pipeline(model.uri)

            eval_output_path = Path(PIPELINE_ROOT) / "Evaluator" / "evaluation"
            eval_run = sorted(map(int, os.listdir(eval_output_path)))[-1]
            eval_run_path = str(eval_output_path / str(eval_run))
            eval_result = tfma.load_eval_result(eval_run_path)
            metrics = eval_result.get_metrics_for_slice()
            metrics = {"top_k_accuracy": metrics["top_k_accuracy"]["doubleValue"]}
            leaderboard.update(model, metrics)
            updated_leaderboard_df = leaderboard.dataframe()
            leaderboard_slot.dataframe(updated_leaderboard_df)

            progress_bar.progress(
                (index + 1) / len(models_to_refresh),
                text=f"Evaluation of {model.participant}/{model.model_id} complete.",
            )

            updated_leaders = set(
                updated_leaderboard_df.loc[
                    updated_leaderboard_df.index.intersection([1])
                ]["Name"]
            )
            if updated_leaders != leaders:
                st.balloons()
                st.toast("New Leader/s: " + ", ".join(updated_leaders))
                time.sleep(3)

        status_slot.empty()


st.fragment(refresh_leaderboard, run_every=60)()
