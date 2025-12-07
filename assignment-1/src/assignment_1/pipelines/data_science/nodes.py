import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer, fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    X = data[parameters["features"]]
    y = data["Subscription Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test

def grid_search_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:

    clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    param_grid = {
        'n_estimators': [700, 750, 800],
        'max_depth': [8, 10, 12],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [5, 7, 10],
        'min_samples_leaf': [1, 2, 4]
        }
    
    f2 = make_scorer(fbeta_score, beta=2)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring=f2, verbose=2)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    logger = logging.getLogger(__name__)
    logger.info("Best parameters found: %s", grid_search.best_params_)
    return best_clf

def random_search_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        "n_estimators": randint(300, 1500),
        "max_depth": [None] + list(range(5, 50)),
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 8),
        "bootstrap": [True, False]
    }

    f2_score = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=80,
        scoring=f2_score,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    logger = logging.getLogger(__name__)

    # Log best parameters + score
    logger.info(f"Best RandomForest Params: {search.best_params_}")
    logger.info(f"Best CV F2 Score: {search.best_score_}")

    return search.best_estimator_


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> RandomForestClassifier:
    
    clf = RandomForestClassifier(n_estimators=parameters["n_estimators"], 
                                 random_state=parameters["random_state"],
                                 max_depth=parameters["max_depth"],
                                 n_jobs=parameters["n_jobs"],
                                 class_weight=parameters["class_weight"],
                                 max_features=parameters["max_features"],
                                 min_samples_split=parameters["min_samples_split"],
                                 min_samples_leaf=parameters["min_samples_leaf"]
                                 )
    clf.fit(X_train, y_train)
    return clf

def grid_search_knn(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:

    knn = KNeighborsClassifier(n_jobs=-1)

    param_grid = {
    'n_neighbors': list(range(65, 69, 1)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=1,
    verbose=2
    )

    grid.fit(X_train, y_train)
    best_knn = grid.best_estimator_
    logger = logging.getLogger(__name__)
    logger.info("Best KNN parameters found: %s", grid.best_params_)
    return best_knn

def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("knn", KNeighborsClassifier())
    ])

    param = {
        "pca__n_components": 24,
        "knn__n_neighbors": 32,
        "knn__weights": "distance",
        "knn__p": 2,
    }

    knn = pipe.set_params(**param)
    knn.fit(X_train, y_train)

    return knn

def evaluate_knn_model(
    knn: KNeighborsClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:

    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    
    logger = logging.getLogger(__name__)
    logger.info("KNN Model F2 Score: %.4f", f2)
    logger.info("KNN Model Accuracy: %.3f", accuracy)
    logger.info("KNN Model F1 Score: %.3f", f1)
    logger.info("KNN Confusion Matrix:\n%s", cm)
    
    return {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": cm.tolist()}

def evaluate_random_forest_model(
    clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:

    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    logger = logging.getLogger(__name__)
    logger.info("Forest Model Accuracy: %.3f", accuracy)
    logger.info("Forest Model F1 Score: %.3f", f1)
    logger.info("Forest Model F2 Score: %.4f", f2)
    logger.info("Forest Confusion Matrix:\n%s", cm)
    
    return {"accuracy": accuracy, "f1_score": f1, "confusion_matrix": cm.tolist()}

def random_search_knn(X_train, y_train):

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("knn", KNeighborsClassifier())
    ])

    param_dist = {
        "pca__n_components": randint(5, 40),
        "knn__n_neighbors": randint(5, 150),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    f2 = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=50,
        scoring=f2,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    logger = logging.getLogger(__name__)

    search.fit(X_train, y_train)

    logger.info(f"Best KNN Params: {search.best_params_}")
    logger.info(f"Best CV F2 Score: {search.best_score_}")

    return search.best_estimator_


def random_search_xgboost(X_train, y_train):

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=8 , random_state=42)

    param_dist = {
        "n_estimators": randint(300, 1500),
        "max_depth": randint(3, 15),
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 1.5, 2]
    }

    f2 = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        xgb,
        param_dist,
        n_iter=80,
        scoring=f2,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    logger = logging.getLogger(__name__)

    # Log best parameters + score
    logger.info(f"Best XGBoost Params: {search.best_params_}")
    logger.info(f"Best CV F2 Score: {search.best_score_}")

    return search.best_estimator_

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> XGBClassifier:

    xgb = XGBClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        learning_rate=parameters["learning_rate"],
        subsample=parameters["subsample"],
        colsample_bytree=parameters["colsample_bytree"],
        gamma=parameters["gamma"],
        reg_alpha=parameters["reg_alpha"],
        reg_lambda=parameters["reg_lambda"],
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=parameters["scale_pos_weight"],
        random_state=parameters["random_state"]
    )
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_xgboost_model(
    xgb: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:

    y_pred = xgb.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    
    logger = logging.getLogger(__name__)
    
    logger.info("XGBoost Model Accuracy: %.3f", accuracy)
    logger.info("XGBoost Model F1 Score: %.3f", f1)
    logger.info("XGBoost Model F2 Score: %.4f", f2)
    logger.info("XGBoost Confusion Matrix:\n%s", cm)
    
    return 

def random_search_catboost(X_train, y_train):

    catboost = CatBoostClassifier(
                                    loss_function="Logloss",
                                    eval_metric="F1",
                                    verbose=0,
                                    task_type="CPU",
                                    random_seed=42,
                                    class_weights=[1, 7],
                                    early_stopping_rounds=20
                                )

    param_dist = {
        "iterations": np.arange(300, 1500, 50),
        "depth": np.arange(3, 10),
        "learning_rate": np.linspace(0.01, 0.1, 1),
        "l2_leaf_reg": np.linspace(4, 20, 10),
        "bagging_temperature": np.linspace(0, 1, 10),
        "random_strength": np.linspace(1, 20, 10),
        "border_count": np.arange(100, 256, 32)
    }

    f2 = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        catboost,
        param_dist,
        n_iter=100,
        scoring=f2,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    logger = logging.getLogger(__name__)

    # Log best parameters + score
    logger.info(f"Best CatBoost Params: {search.best_params_}")
    logger.info(f"Best CV F2 Score: {search.best_score_}")

    return search.best_estimator_

def train_catboost_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> CatBoostClassifier:

    catboost = CatBoostClassifier(
        iterations=parameters["iterations"],
        depth=parameters["depth"],
        learning_rate=parameters["learning_rate"],
        l2_leaf_reg=parameters["l2_leaf_reg"],
        bagging_temperature=parameters["bagging_temperature"],
        random_strength=parameters["random_strength"],
        border_count=parameters["border_count"],
        loss_function="Logloss",
        eval_metric="F1",
        verbose=0,
        task_type="CPU",
        random_seed=parameters["random_seed"],
        class_weights=parameters["class_weights"]
    )
    catboost.fit(X_train, y_train)
    return catboost

def evaluate_catboost_model(
    catboost: CatBoostClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:

    y_pred = catboost.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    
    logger = logging.getLogger(__name__)
    
    logger.info("CatBoost Model Accuracy: %.3f", accuracy)
    logger.info("CatBoost Model F1 Score: %.3f", f1)
    logger.info("CatBoost Model F2 Score: %.4f", f2)
    logger.info("CatBoost Confusion Matrix:\n%s", cm)
    
    return

def random_search_decision_tree(X_train : pd.DataFrame, y_train : pd.Series):
    
    tree = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": randint(2, 50),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20),
        "max_features": [None, "sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    f2 = make_scorer(fbeta_score, beta=2)

    random_search = RandomizedSearchCV(
        estimator=tree,
        param_distributions=param_dist,
        n_iter=80,
        scoring=f2,
        cv=5,
        verbose=1,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    best_tree = random_search.best_estimator_
    best_params = random_search.best_params_
    

    logger = logging.getLogger(__name__)
    logger.info(f"Best Decision Tree Params: {best_params}")
    
    return best_tree


def evaluate_tree_model(
    tree: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series, columns: dict
) -> dict[str, float]:

    y_pred = tree.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    tree_struct = tree.tree_
    used_feature_indices = tree_struct.feature[tree_struct.feature >= 0]
    used_feature_indices = np.unique(used_feature_indices)

    feature_array = np.array(columns["features"])

    used_features = feature_array[used_feature_indices].tolist()
    unused_features = list(set(columns["features"]) - set(used_features))
    
    logger = logging.getLogger(__name__)
    
    logger.info("Tree Model Accuracy: %.3f", accuracy)
    logger.info("Tree Model F1 Score: %.3f", f1)
    logger.info("Tree Model F2 Score: %.4f", f2)
    logger.info("Tree Confusion Matrix:\n%s", cm)

    logger.info(f"Used features: {used_features}")
    logger.info(f"Unused features: {unused_features}")
    
    return

def drop_cols_data (data: pd.DataFrame, parameters: dict) -> tuple:

    # data.drop(columns=['Marital Status_unknown','Marital Status_married',"Marital Status_single"], inplace=True)
    data.drop(columns=['Housing Loan_yes', 'Housing Loan_unknown'], inplace=True)
    data.drop(columns=['Credit Default_unknown', "Credit Default_yes"], inplace=True)
    data.drop(columns=['Personal Loan_unknown', 'Personal Loan_yes'], inplace=True)

    X = data[parameters["features"]]
    y = data["Subscription Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return X_train, X_test, y_train, y_test