import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def data_processing_node(df: pd.DataFrame, number_cols: dict) -> pd.DataFrame:

    #drop duplicates
    df_drop_dups = df.drop_duplicates()

    # Convert Age column to integer
    df_drop_dups["Age"] = df_drop_dups["Age"].apply(lambda x: int(x.split()[0]))

    # Transform Contact Method column values into 2 distinct values
    df_drop_dups['Contact Method'] = df_drop_dups['Contact Method'].str.lower().str.strip()

    df_drop_dups['Contact Method'] = df_drop_dups['Contact Method'].replace({
        'cell': 'cellular', 'Telephone': 'telephone'})

    # Convert Housing and Personal Loan null values to 'unknown'
    df_drop_dups['Housing Loan'] = df_drop_dups['Housing Loan'].replace({ None: "unknown" })
    df_drop_dups['Personal Loan'] = df_drop_dups['Personal Loan'].replace({ None: "unknown" })

    df_drop_dups["Campaign Calls"] = df_drop_dups["Campaign Calls"].abs()

    #drop null values that may have slip through from other columns, for when they expand the dataset
    df_drop_dups = df_drop_dups.dropna()

    df_copy = df_drop_dups

    df_copy['Subscription Status'] = df_copy['Subscription Status'].replace({ "no" : 0, "yes": 1 })

    # Calculate median of acceptable ages
    median_age = df_copy.loc[df_copy["Age"] < 123, "Age"].median()

    # Impute impossible ages with median age
    df_copy.loc[df_copy["Age"] > 123, "Age"] = median_age

    # Scaling features that do not go from 0 to 1
    minMax_Scaler = MinMaxScaler()
    df_copy[number_cols['features']] = minMax_Scaler.fit_transform(df_copy[number_cols['features']])

    processed_data = df_copy

    return processed_data

def label_encoding_node(processed_data: pd.DataFrame) -> pd.DataFrame:

    #Label Encoder
    labelencoder = LabelEncoder()

    df_label_encoded = processed_data.copy()

    # Drop Client ID cause each client ID is unique, hard to find relevance using it
    df_label_encoded.drop(columns=["Client ID"], inplace=True)

    # All string columns put into a list
    str_columns = processed_data.select_dtypes(include=['object']).columns.tolist()

    # For loop to change every single string column into label encoded data
    for col in str_columns:
        df_label_encoded[col] = labelencoder.fit_transform(df_label_encoded[col])

    # Return Label Data
    return df_label_encoded


def one_hot_encoding_node(processed_data: pd.DataFrame) -> pd.DataFrame:

    # Copy data
    df_copy_noID = processed_data.copy()

    # Drop Client ID cause each client ID is unique, harder to find relevance
    df_copy_noID.drop(columns=["Client ID"], inplace=True)

    # All string columns put into a list
    str_columns = processed_data.select_dtypes(include=['object']).columns.tolist()

    # Transform relevant columns into one hot encoded data
    df_one_hot_encoded = pd.get_dummies(df_copy_noID, columns=str_columns, drop_first=True)

    return df_one_hot_encoded
