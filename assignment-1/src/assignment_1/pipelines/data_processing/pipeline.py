from kedro.pipeline import Node, Pipeline

from .nodes import data_processing_node, label_encoding_node, one_hot_encoding_node


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=data_processing_node, # function
                inputs=["raw_sql_data", "params:number_cols"], # Input raw data and columns needed for scaling
                outputs="processed_data", # output for data processing
                name="preprocess_data_node", # Name of this Node
            ),
            Node(
                func=label_encoding_node, # function
                inputs="processed_data", # No need parameters cause the node can get the parameters
                outputs="label_encoded_dataset", # Label encoded data
                name="label_encoding_node", # Name of this Node
            ),
            Node(
                func=one_hot_encoding_node, # function
                inputs="processed_data", # No need parameters cause the node can get the parameters
                outputs="one_hot_encoded_data", #One Hot Encoded Data
                name="one_hot_encoding_node", # Name of this Node
            ),
        ]
    )
