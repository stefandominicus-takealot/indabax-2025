"""
Extends `tfx.v1.components.Trainer` to support additional "query" and "item" input
artifacts.
"""

from __future__ import annotations

from typing import Any

from tfx.components.trainer import fn_args_utils
from tfx.components.trainer.executor import GenericExecutor as _GenericExecutor
from tfx.components.util import udf_utils
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.dsl.components.base.executor_spec import ExecutorClassSpec
from tfx.orchestration.data_types import RuntimeParameter
from tfx.proto import trainer_pb2
from tfx.types import standard_component_specs
from tfx.types.channel import BaseChannel, Channel
from tfx.types.component_spec import ChannelParameter
from tfx.types.standard_artifacts import Examples, Model, ModelRun, Schema
from tfx.types.standard_component_specs import TrainerSpec as _TrainerSpec
from tfx.utils import json_utils

from tfx_tfrs import (
    ITEM_EXAMPLES_KEY,
    ITEM_SCHEMA_KEY,
    QUERY_EXAMPLES_KEY,
    QUERY_SCHEMA_KEY,
)
from tfx_tfrs.fn_args_utils import get_common_fn_args


class TrainerSpec(_TrainerSpec):
    """
    An extended `TrainerSpec` supporting additional "query" and "item" input channels.
    """

    INPUTS = dict(
        **_TrainerSpec.INPUTS,
        **{
            QUERY_EXAMPLES_KEY: ChannelParameter(type=Examples, optional=True),
            QUERY_SCHEMA_KEY: ChannelParameter(type=Schema, optional=True),
            ITEM_EXAMPLES_KEY: ChannelParameter(type=Examples, optional=True),
            ITEM_SCHEMA_KEY: ChannelParameter(type=Schema, optional=True),
        },
    )


class GenericExecutor(_GenericExecutor):
    """
    A subclassed executor that monkey patches `get_common_fn_args`.

    This executor runs the training task, both locally and on GCP.
    """

    fn_args_utils.get_common_fn_args = get_common_fn_args


class Trainer(BaseComponent):
    """
    A component interface, imitating the standard TFX `Trainer` as closely as possible,
    while supporting the additional "query" and "item" input channels.

    Besides the additional constructor parameters, this class also references the
    modified `TrainerSpec` and `GenericExecutor` classes.
    """

    SPEC_CLASS = TrainerSpec
    EXECUTOR_SPEC = ExecutorClassSpec(GenericExecutor)

    def __init__(
        self,
        examples: BaseChannel | None = None,
        transform_graph: BaseChannel | None = None,
        schema: BaseChannel | None = None,
        base_model: BaseChannel | None = None,
        hyperparameters: BaseChannel | None = None,
        query_examples: BaseChannel | None = None,
        query_schema: BaseChannel | None = None,
        item_examples: BaseChannel | None = None,
        item_schema: BaseChannel | None = None,
        module_file: str | RuntimeParameter | None = None,
        run_fn: str | RuntimeParameter | None = None,
        train_args: trainer_pb2.TrainArgs | RuntimeParameter | None = None,
        eval_args: trainer_pb2.EvalArgs | RuntimeParameter | None = None,
        custom_config: dict[str, Any] | RuntimeParameter | None = None,
    ):  # pragma: no cover
        if bool(module_file) == bool(run_fn):
            raise ValueError(
                "Exactly one of 'module_file' or 'run_fn' must be supplied"
            )

        spec = TrainerSpec(
            examples=examples,
            transform_graph=transform_graph,
            schema=schema,
            base_model=base_model,
            hyperparameters=hyperparameters,
            query_examples=query_examples,
            query_schema=query_schema,
            item_examples=item_examples,
            item_schema=item_schema,
            train_args=train_args or trainer_pb2.TrainArgs(),
            eval_args=eval_args or trainer_pb2.EvalArgs(),
            module_file=module_file,
            run_fn=run_fn,
            custom_config=(
                custom_config
                if isinstance(custom_config, RuntimeParameter)
                else json_utils.dumps(custom_config)
            ),
            model=Channel(type=Model),
            model_run=Channel(type=ModelRun),
        )
        super().__init__(spec=spec)

        if udf_utils.should_package_user_modules():
            # In this case, the `MODULE_PATH_KEY` execution property will be injected
            # as a reference to the given user module file after packaging, at which
            # point the `MODULE_FILE_KEY` execution property will be removed.
            udf_utils.add_user_module_dependency(
                self,
                standard_component_specs.MODULE_FILE_KEY,
                standard_component_specs.MODULE_PATH_KEY,
            )
