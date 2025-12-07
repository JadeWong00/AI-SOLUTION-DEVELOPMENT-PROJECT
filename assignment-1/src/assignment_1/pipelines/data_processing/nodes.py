import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def data_processing_node(df: pd.DataFrame, number_cols: dict) -> pd.DataFrame:

    df_drop_dups = df.drop_duplicates()

    df_drop_dups["Age"] = df_drop_dups["Age"].apply(lambda x: int(x.split()[0]))

    df_drop_dups['Contact Method'] = df_drop_dups['Contact Method'].str.lower().str.strip()

    df_drop_dups['Contact Method'] = df_drop_dups['Contact Method'].replace({
        'cell': 'cellular', 'Telephone': 'telephone'})

    df_drop_dups['Housing Loan'] = df_drop_dups['Housing Loan'].replace({ None: "unknown" })

    df_drop_dups['Personal Loan'] = df_drop_dups['Personal Loan'].replace({ None: "unknown" })

    df_drop_dups["Campaign Calls"] = df_drop_dups["Campaign Calls"].abs()

    df_drop_dups = df_drop_dups.dropna()

    df_copy = df_drop_dups

    df_copy['Subscription Status'] = df_copy['Subscription Status'].replace({ "no" : 0, "yes": 1 })

    df_norm_age = df_copy[df_copy['Age'] < 123]
    df_copy["Age"].loc[df_copy["Age"] > 123] = df_norm_age["Age"].median()

    median_age = df_copy.loc[df_copy["Age"] < 123, "Age"].median()
    df_copy.loc[df_copy["Age"] > 123, "Age"] = median_age

    minMax_Scaler = MinMaxScaler()
    df_copy[number_cols['features']] = minMax_Scaler.fit_transform(df_copy[number_cols['features']])

    processed_data = df_copy

    return processed_data

def label_encoding_node(processed_data: pd.DataFrame) -> pd.DataFrame:

    labelencoder = LabelEncoder()

    df_label_encoded = processed_data.copy()
    df_label_encoded.drop(columns=["Client ID"], inplace=True)
    str_columns = processed_data.select_dtypes(include=['object']).columns.tolist()

    for col in str_columns:
        df_label_encoded[col] = labelencoder.fit_transform(df_label_encoded[col])

    return df_label_encoded


def one_hot_encoding_node(processed_data: pd.DataFrame) -> pd.DataFrame:

    df_copy_noID = processed_data.copy()
    df_copy_noID.drop(columns=["Client ID"], inplace=True)
    str_columns = processed_data.select_dtypes(include=['object']).columns.tolist()
    df_one_hot_encoded = pd.get_dummies(df_copy_noID, columns=str_columns, drop_first=True)

    return df_one_hot_encoded
