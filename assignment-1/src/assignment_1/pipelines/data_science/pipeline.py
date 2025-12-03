from kedro.pipeline import Node, Pipeline

from .nodes import (evaluate_random_forest_model, split_data, train_random_forest_model,
                    grid_search_random_forest, evaluate_knn_model, 
                    grid_search_knn ,random_search_random_forest, random_search_knn)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=split_data,
                inputs=["one_hot_encoded_data", "params:one_hot_encoded_columns"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            Node(
                func=train_random_forest_model,
                inputs=["X_train", "y_train", "params:random_forest_model_parameters"],
                outputs="forest",
                name="train_model_node",
            ),
            Node(
                func=random_search_knn,
                inputs=["X_train", "y_train"],
                outputs="knn_model",
                name="train_knn_model_node",
            ),
            Node(
                func=evaluate_random_forest_model,
                inputs=["forest", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
            Node(
                func=evaluate_knn_model,
                inputs=["knn_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_knn_model_node",
            ),
        ]
    )


"""
            Node(
                func=train_knn_model,
                inputs=["X_train", "y_train"],
                outputs="knn_model",
                name="train_knn_model_node",
            ),
            Node(
                func=train_knn_model,
                inputs=["X_resampled", "y_resampled"],
                outputs="trained_knn_model",
                name="train_knn_model_node",
            ),
            Node(
                func=evaluate_knn_model,
                inputs=["trained_knn_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_knn_model_node",
            ),
            """