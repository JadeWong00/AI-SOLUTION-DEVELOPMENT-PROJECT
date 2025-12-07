from kedro.pipeline import Node, Pipeline

from .nodes import data_processing_node, label_encoding_node, one_hot_encoding_node


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=data_processing_node,
                inputs=["raw_sql_data", "params:number_cols"],
                outputs="processed_data",
                name="preprocess_data_node",
            ),
            Node(
                func=label_encoding_node,
                inputs="processed_data",
                outputs="label_encoded_dataset",
                name="label_encoding_node",
            ),
            Node(
                func=one_hot_encoding_node,
                inputs="processed_data",
                outputs="one_hot_encoded_data",
                name="one_hot_encoding_node",
            ),
        ]
    )
