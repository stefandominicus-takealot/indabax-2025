from __future__ import annotations

from typing import Any, List

import attr
from tfx.components.trainer.fn_args_utils import _TELEMETRY_DESCRIPTORS, DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs as _FnArgs
from tfx.components.trainer.fn_args_utils import (
    get_common_fn_args as _get_common_fn_args,
)
from tfx.components.util import tfxio_utils
from tfx.types import artifact_utils
from tfx.types.standard_artifacts import Artifact
from tfx.utils import io_utils

from tfx_tfrs import (
    ITEM_EXAMPLES_KEY,
    ITEM_SCHEMA_KEY,
    QUERY_EXAMPLES_KEY,
    QUERY_SCHEMA_KEY,
)


@attr.s
class FnArgs(_FnArgs):
    """
    An extension of the standard `FnArgs`, introducing additional attributes required to
    access to the "query" and "item" artifacts.
    """

    query_files = attr.ib(type=List[str], default=None)
    query_schema_path = attr.ib(type=str, default=None)
    query_data_accessor = attr.ib(type=DataAccessor, default=None)
    item_files = attr.ib(type=List[str], default=None)
    item_schema_path = attr.ib(type=str, default=None)
    item_data_accessor = attr.ib(type=DataAccessor, default=None)


def get_common_fn_args(
    input_dict: dict[str, List[Artifact]],
    exec_properties: dict[str, Any],
    working_dir: str | None = None,
) -> FnArgs:  # pragma: no cover
    """
    A wrapper around the standard `get_common_fn_args` function, responsible for setting
    the additional "query" and "item" attributes on `FnArgs`.
    """
    fn_args = _get_common_fn_args(
        input_dict=input_dict, exec_properties=exec_properties, working_dir=working_dir
    )

    query_files = query_data_accessor = None
    if query_examples := input_dict.get(QUERY_EXAMPLES_KEY):
        query_files = [
            io_utils.all_files_pattern(
                artifact_utils.get_split_uri(query_examples, split)
            )
            for split in artifact_utils.decode_split_names(
                artifact_utils.get_single_instance(query_examples).split_names
            )
        ]
        query_data_accessor = DataAccessor(
            tf_dataset_factory=tfxio_utils.get_tf_dataset_factory_from_artifact(
                query_examples, _TELEMETRY_DESCRIPTORS
            ),
            record_batch_factory=tfxio_utils.get_record_batch_factory_from_artifact(
                query_examples, _TELEMETRY_DESCRIPTORS
            ),
            data_view_decode_fn=tfxio_utils.get_data_view_decode_fn_from_artifact(
                query_examples, _TELEMETRY_DESCRIPTORS
            ),
        )

    query_schema_path = None
    if query_schema := input_dict.get(QUERY_SCHEMA_KEY):
        query_schema_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(query_schema)
        )

    item_files = item_data_accessor = None
    if item_examples := input_dict.get(ITEM_EXAMPLES_KEY):
        item_files = [
            io_utils.all_files_pattern(
                artifact_utils.get_split_uri(item_examples, split)
            )
            for split in artifact_utils.decode_split_names(
                artifact_utils.get_single_instance(item_examples).split_names
            )
        ]
        item_data_accessor = DataAccessor(
            tf_dataset_factory=tfxio_utils.get_tf_dataset_factory_from_artifact(
                item_examples, _TELEMETRY_DESCRIPTORS
            ),
            record_batch_factory=tfxio_utils.get_record_batch_factory_from_artifact(
                item_examples, _TELEMETRY_DESCRIPTORS
            ),
            data_view_decode_fn=tfxio_utils.get_data_view_decode_fn_from_artifact(
                item_examples, _TELEMETRY_DESCRIPTORS
            ),
        )

    item_schema_path = None
    if item_schema := input_dict.get(ITEM_SCHEMA_KEY):
        item_schema_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(item_schema)
        )

    return FnArgs(
        **attr.asdict(fn_args, retain_collection_types=True),
        query_files=query_files,
        query_schema_path=query_schema_path,
        query_data_accessor=query_data_accessor,
        item_files=item_files,
        item_schema_path=item_schema_path,
        item_data_accessor=item_data_accessor,
    )
