# Overview of insights from EDA
- The data seems to be not that correlated to the subscription status, the lack of a balanced dataset would lead to one side being more accurate than others. Overall, the data has too little positive response for subscription and will greatly affect the performance of all models that use this dataset

# Explanation of choice of models

## Random Forest
- Maintains original data structure
- Reliable as a baseline model

## CatBoost
- Can handle imbalanced datasets well
- Can also adjust weights of the classes

## XGBoost
- Follows random forest, except each tree tries to fix its previous mistakes
- Highly tunable

# How to run pipeline(s)
- conda activate <kedro-environment>
- cd to file location
- kedro run

or

- cd to where run.sh is
- bash run.sh


# Contributions

## Ng Yik Heng ()
- EDA for Marital Status, Personal Loan, Campaign Calls
- Pipeline structure
- Nodes

## Matthew Christopher Tan Ming Wen (230649F)
- EDA for Occupation, Housing Loan, Previous Contact Days
- EDA & Github structure
- Logistic Regression

## Wong Qun Xiu ()
- EDA for Age, Credit Default, Contact Method
- Pipeline structure
- XGBoost