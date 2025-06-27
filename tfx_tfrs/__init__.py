"""
TFX Trainer & Tuner components take in a single `examples` artifact, which is sufficient
for many training loops. However, Tensorflow Recommenders (TFRS) models typically
require additional datasets besides the training examples. For example:
  - The TFRS `Retrieval` task requires knowledge of all items (candidates) in the
    problem space.
  - Once trained, it is common to build a `ScaNN` index for serving. This again requires
    a dataset of all items to be indexed into the `ScaNN` layer.

For the sake of generality, we introduce support for both "query" and "item" datasets,
alongside the standard training dataset.
"""

ITEM_EXAMPLES_KEY = "item_examples"
ITEM_SCHEMA_KEY = "item_schema"
QUERY_EXAMPLES_KEY = "query_examples"
QUERY_SCHEMA_KEY = "query_schema"
