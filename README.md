# EGT309 Group Project – Machine Learning Pipeline

## a. Full name (as in NRIC) and email address

- Name:Ng Yik Heng email:ngyikheng04@gmail.com 
- Name:Matthew Christopher Tan Ming Wen email:wehttam36912@gmail.com 
- Name: WONG QUN XIU email:qxiujade48@gmail.com


## b. Overview of the Submitted Folder & Structure

This repository contains the full machine learning pipeline, including:
- Exploratory Data Analysis (EDA),
- Dataset,
- Model training and evaluation,
- Dockerized execution environment,
- Saved trained models,
- Supporting documents.

### Folder Structure

```text
.
├── README.md                              # Project documentation
├── eda.ipynb                              # Task 1 Exploratory Data Analysis (Jupyter)
├── eda.pdf                                # Exported EDA report (PDF)
├── Presentation Slide_Teram GG.pdf        # Final presentation slides
├── bmarket.db                             # Dataset (SQLite database)
├── Dockerfile                             # Docker configuration for pipeline execution
├── requirements.txt                      # Python dependency list
├── run.sh                                 # Script to build & run the pipeline
├── saved_models/                          # Trained and saved machine learning models
└── assignment-1/                          # Source code for pipeline (training + evaluation)


## c. Instructions for executing the pipeline and modifying any parameters

# Option 1: Run Using Docker 

Step 1: Build Docker Image -- docker build -t aisdp-assignment .
Step 2: Run the pipeline -- docker run --rm aisdp-assignment

This will:
- Load the dataset from bmarket.db
- Train the machine learning models
- Evaluate performance
- Save trained models into saved_models/

# Option 2: Run Using run.sh Script

On Windows: bash run.sh
On macOS / Linux: chmod +x run.sh and ./run.sh

The run.sh script automatically:
- Builds the Docker image
- Runs the pipeline container

# Option 3: Run in VsCode terminal (Anaconda Prompt)

Step 1: pip install -r requirements.txt
Step 2: cd assignment-1
Step 3: kedro run

# Modifying Parameters
All key parameters can be adjusted inside the Python files located in: assignment-1/

Examples of parameters include:
- Train-test split ratio
- Feature selection
- Model hyperparameters (number of trees, learning rate)

After parameter changes:
- If using Docker -- Rebuild image
- If using Python -- Just rerun the script


## d. Description of logical steps/flow of the pipeline. If you find it useful, please feel free to include suitable visualisation aids (eg, flow charts) within the README

# Overall Pipeline Flow
1. Data Source (bmarket.db)
- The raw dataset is stored in a SQLite database file named bmarket.db.
- This serves as the single source of truth for all model training and evaluation.

2. Preprocessing Node
- Data is cleaned to handle:
    Missing values (median/mode imputation),
    Inconsistent categorical labels,
    Outliers detected during EDA.
-Irrelevant or highly correlated features (as identified in Task 1 EDA) are removed.

3. Hot Encoding Node
- All cleaned categorical features are transformed using one-hot encoding.
- This converts categorical attributes into numerical format suitable for machine learning models.

4. Split Data Node
- The processed dataset is split into:
    Training set (used to learn model parameters),
    Test set (used strictly for unbiased evaluation).

5. Training Data → Model
- The selected machine learning model (Random Forest Classifier) is trained using the training dataset.
- A baseline Logistic Regression model is also trained for performance comparison.

6. Model Evaluation Using Test Data
- The trained models are evaluated using the test dataset.
- Performance is measured using:
    Accuracy,
    Precision,
    Recall,
    F1-Score,
    Confusion Matrix.
- The best-performing model is saved into the saved_models/ directory.

# Flowchart 
    A[bmarket.db] --> B[Preprocessing Node]
    B --> C[Hot Encoding Node]
    C --> D[Split Data Node]
    D --> E[Training Data]
    E --> F[Model Training]
    F --> G[Model Evaluation with Test Data]


## e. Overview of key findings from the EDA conducted in Task 1 and the choices made in the pipeline based on these findings, particularly any feature engineering. Please keep the details of the EDA in the `.ipynb`. The information in the `README.md` should be a quick summary of the details from `.ipynb`

# High Feature Dimensionality After Encoding
The dataset contains a large number of categorical features. When these categorical variables were transformed using one-hot encoding, the total number of features increased significantly. This resulted in a high-dimensional feature space, which has the following implications:

- Models that rely heavily on distance measurements (such as K-Nearest Neighbours) become computationally expensive and less stable.
- Linear models such as Logistic Regression may struggle to capture complex relationships when feature interactions are nonlinear.

As a result, tree-based models were prioritised because:
- They are not affected by feature scaling,
- They handle high-dimensional data efficiently,
- They can automatically model feature interactions and nonlinear relationships.

# Dataset Complexity & Nonlinear Relationships
EDA revealed that the target variable does not exhibit a simple linear relationship with individual features. Instead, multiple attributes interact in complex ways. This makes tree-based models (e.g., Random Forest) more suitable than simpler linear models.

Tree-based models were therefore selected because they:
- Capture nonlinear dependencies naturally,
- Are robust to noisy features and outliers,
- Maintain strong performance on mixed numerical and categorical datasets.

# Overall Impact of EDA on Pipeline Design
The EDA directly influenced:
- The use of one-hot encoding for categorical variables,
- The exclusion of distance-based models,
- The selection of tree-structured ensemble models for final deployment,
- The need for robust evaluation metrics due to data complexity.

## f. Describe how the features in the dataset are processed (summarised in a table)

| Feature Type |                       Features                               |    Processing Applied                  |
| ------------ | -----------------------------------------------------------  | ---------------------------------      |
| Numerical    | Age, Campaign Calls(CC), Previous Contact Days(PCD)          | Median imputation (Age), Convert nulls to 'unknown' (CC, PCD)  |
| Categorical  | Occupation, Marital Status, Education Level, Credit Default  | Cleaning + One-hot encoding            |
| Boolean      | Subscription Status, Contact Method                          | Converted to 0/1                       |
| -            | Client ID                                                   | Dropped                                |


## g. Explanation of your choice of models for each machine learning task

## Random Forest
- Maintains original data structure
- Reliable as a baseline model

## CatBoost
- Can handle imbalanced datasets well
- Can also adjust weights of the classes

## XGBoost
- Follows random forest, except each tree tries to fix its previous mistakes
- Highly tunable

## h. Evaluation of the models developed. Any metrics used in the evaluation should also be explained

Three machine learning models were trained and evaluated on the test dataset:
- Random Forest
- CatBoost
- XGBoost

The evaluation was conducted using the following metrics:
- Accuracy – overall correctness of predictions
- F1-Score – balance between precision and recall
- F2-Score – prioritises recall over precision (important for imbalanced datasets)
- Confusion Matrix – detailed breakdown of correct and incorrect predictions

| Model         | Accuracy  | F1-Score  | F2-Score |
| ------------- | --------- | --------- | -------- |
| Random Forest | 0.668     | 0.305     | 0.4478   |
| CatBoost      | 0.685     | 0.304     | 0.4348   |
| XGBoost       | **0.698** | **0.307** | 0.4335   |

Confusion Matrices:
[[ TN   FP ]
 [ FN   TP ]]

- Random Forest
[[4904 2409]
 [ 324  601]]

- CatBoost
[[5081 2232]
 [ 360  565]]

- XGBoost
[[5198 2115]
 [ 373  552]]


## i. Other considerations for deploying the models developed

1. Data Imbalance & Need for More Positive Samples
The dataset exhibits class imbalance, where the number of positive target samples is significantly lower than the negative class. This limits the model’s ability to learn sufficient patterns from the minority class and may result in:
- Lower recall for positive predictions,
- Increased false negatives,
- Biased decision boundaries

To improve real-world performance:
- More positive and relevant samples should be collected, and
- Techniques such as resampling, SMOTE, or cost-sensitive learning may be considered during retraining.

2. Data Quality Issues & Human Validation
During EDA, it was observed that:
- There are invalid and unrealistic values, such as records indicating individuals aged 150 years old.

Before deployment:
- A manual data audit should be performed,
- Automated validation rules should be implemented (e.g. age must be between 0–100),
- Suspicious records should be corrected or removed to prevent misleading model predictions.

3. Model Performance Monitoring
Once deployed:
- The model should be continuously monitored for:
    Performance degradation,
    Changes in data distribution (data drift),
    Changes in user behaviour (concept drift).
- Periodic retraining may be required when new clean data becomes available.

4. Ethical & Reliability Considerations
- Predictions should not be used as the sole decision-making authority.
- Human review should remain part of high-risk decision processes.
- Bias and fairness across different age groups and demographic segments should be regularly reviewed.

## Conclusion
For successful deployment, the model requires cleaner validated data, more balanced target classes, and continuous monitoring after release to ensure long-term reliability and ethical performance.

