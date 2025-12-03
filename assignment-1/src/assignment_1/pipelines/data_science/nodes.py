import logging
import pandas as pd
from sklearn.metrics import max_error, mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix, make_scorer, fbeta_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

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
    """Performs grid search to find the best hyperparameters for Random Forest classifier."""
    clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    param_grid = {
        'n_estimators': [700, 750, 800],
        'max_depth': [8, 10, 12],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [5, 7, 10],
        'min_samples_leaf': [1, 2, 4]
        }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1', verbose=2)
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
    """Performs grid search to find the best hyperparameters for KNN classifier."""
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

def evaluate_knn_model(
    knn: KNeighborsClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate KNN classifier."""
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
    """Evaluate Random Forest classifier."""
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
    knn = KNeighborsClassifier()

    param_dist = {
        "n_neighbors": randint(50, 150),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

    f2_score = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        estimator=knn,
        param_distributions=param_dist,
        n_iter=50,
        scoring=f2_score,
        cv=5,
        verbose=1,
        n_jobs=1,
        random_state=42
    )

    search.fit(X_train, y_train)

    logger = logging.getLogger(__name__)

    # Log best parameters + score
    logger.info(f"Best KNN Params: {search.best_params_}")
    logger.info(f"Best CV F2 Score: {search.best_score_}")

    return search.best_estimator_


