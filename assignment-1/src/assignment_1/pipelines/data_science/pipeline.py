from kedro.pipeline import Node, Pipeline

from .nodes import (evaluate_random_forest_model, split_data, train_random_forest_model, train_xgboost_model,
                    grid_search_random_forest, evaluate_knn_model, random_search_xgboost, train_knn_model,
                    random_search_catboost, evaluate_catboost_model, train_catboost_model, random_search_decision_tree,
                    evaluate_tree_model, drop_cols_data, 
                    grid_search_knn ,random_search_random_forest, random_search_knn, evaluate_xgboost_model)


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
                func=drop_cols_data,
                inputs=["one_hot_encoded_data", "params:kept_columns"],
                outputs=["X2_train", "X2_test", "y2_train", "y2_test"],
                name="dropped_cols_data_node",
            ),
            Node(
                func=train_random_forest_model,
                inputs=["X_train", "y_train", "params:random_forest_model_parameters"],
                outputs="forest",
                name="train_model_node",
            ),
            Node(
                func=train_knn_model,
                inputs=["X_train", "y_train"],
                outputs="knn_model",
                name="train_knn_model_node",
            ),
            Node(
                func=train_xgboost_model,
                inputs=["X_train", "y_train", "params:xgboost_model_parameters"],
                outputs="xgb_model",
                name="train_xgboost_model_node",
            ),
            Node(
                func=train_catboost_model,
                inputs=["X_train", "y_train"],
                outputs="catboost_model",
                name="random_search_catboost_node",
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
            Node(
                func=evaluate_xgboost_model,
                inputs=["xgb_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_xgboost_model_node",
            ),
            Node(
                func=evaluate_catboost_model,
                inputs=["catboost_model", "X_test", "y_test"],
                outputs=None,
                name="evaluate_catboost_model_node",
            ),
            Node(
                func=random_search_decision_tree,
                inputs=["X2_train", "y2_train"],
                outputs="decision_tree_model",
                name="random_search_decision_tree_node",
            ),
            Node(
                func=evaluate_tree_model,
                inputs=["decision_tree_model","X2_test", "y2_test", "params:kept_columns"],
                outputs=None,
                name="evaluate_tree_model_node",
            ),
        ]
    )
