from time import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import RunTypes, load_prices


def get_features(data_df, run_type=RunTypes.NON_GEO.value):
    all_features = data_df.columns.to_list()
    exclude = ["id", "price"]
    geo_features = [
        "latitude",
        "longitude",
        "poiCount",
        *[col for col in all_features if "Distance" in col],
    ]
    if run_type in [RunTypes.GEO.value, RunTypes.GEO_OSM.value]:
        features = [col for col in all_features if col not in exclude]
        return features
    else:
        features = [col for col in all_features if col not in exclude + geo_features]
        return features


def clean_data(data_df, feature_columns):
    to_remove = _check_nan(data_df, feature_columns)
    to_remove += ["id"]
    print(f"Removing {len(to_remove)} features {to_remove}")
    feature_columns = [col for col in feature_columns if col not in to_remove]
    data_df = data_df.drop(to_remove, axis=1)
    data_df = _fill_nan_with_median(data_df, feature_columns)
    data_df = _no_yes_to_num(data_df, feature_columns)
    return data_df, feature_columns


def _check_nan(data, features):
    size = data.shape[0]
    to_remove = []
    for feature in features:
        nan_sum = pd.isnull(data[feature]).sum()
        # print(f"Feature {feature} has {nan_sum} NaNs")
        if int(nan_sum) > int(0.35 * size):
            to_remove.append(feature)

    return to_remove


def _fill_nan_with_median(data_df, feature_columns):
    for feature in feature_columns:
        if data_df[feature].dtype == "O":
            data_df[feature].fillna(data_df[feature].mode()[0], inplace=True)
        else:
            data_df[feature].fillna(data_df[feature].median(), inplace=True)
    return data_df


def _no_yes_to_num(data_df, feature_columns):
    for feature in feature_columns:
        if data_df[feature].dtype == "O" and set(data_df[feature].unique()) == {
            "no",
            "yes",
        }:
            data_df[feature] = data_df[feature].map({"no": 0, "yes": 1})
    return data_df


def preprocess(city, run_type=RunTypes.NON_GEO.value):
    print("Preprocessing...")
    start_time = time()
    prices = load_prices(city)
    feature_columns = get_features(prices, run_type=run_type)
    prices, feature_columns = clean_data(prices, feature_columns)

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder())])

    numeric_features = [col for col in feature_columns if prices[col].dtype != "O"]
    categorical_features = [col for col in feature_columns if prices[col].dtype == "O"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print(f"Preprocessing took {time() - start_time:.2f} seconds")
    return preprocessor, prices
