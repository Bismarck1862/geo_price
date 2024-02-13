import warnings
from time import time

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from .geo_emb import get_embeddings
from .utils import RunTypes, load_prices, GEO_TYPES, OSM_TYPES

warnings.filterwarnings("ignore")

def get_features(data_df, run_type=RunTypes.NON_GEO.value):
    all_features = data_df.columns.to_list()
    exclude = ["id", "price"]
    non_osm_features = [
        "poiCount",
        *[col for col in all_features if "Distance" in col],
    ]
    geo_features = [
        "latitude",
        "longitude",
        *non_osm_features,
    ]

    if run_type in GEO_TYPES:
        print("Using geo features")
        features = [col for col in all_features if col not in exclude]
        return features
    elif run_type in OSM_TYPES:
        print("Using osm features")
        features = [col for col in all_features if col not in exclude + non_osm_features]
        return features
    else:
        print("Not using geo features")
        features = [col for col in all_features if col not in exclude + geo_features]
        return features


def clean_data(data_df, feature_columns):
    to_remove = _check_nan(data_df, feature_columns)
    to_remove += ["id"]
    to_remove += [col for col in data_df.columns if col not in [*feature_columns, "price"]]
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
    print(f"Number of features: {len(feature_columns)}")
    print(f"Features: {feature_columns}")
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
    if run_type in OSM_TYPES:
        geometries = [Point(longitude, latitude) \
                    for longitude, latitude in zip(prices["longitude"],
                                                    prices["latitude"])]
        cites = [f"{city.title()}, Poland" for city in list(prices["city"].unique())]
        hex_embeddings, highway_embeddings, region_gdfs = gpd.GeoDataFrame(), \
            gpd.GeoDataFrame(), gpd.GeoDataFrame()

        for city in tqdm(cites):
            print(f"Getting embeddings for {city}")
            hex_embedding, highway_embedding, region_gdf = get_embeddings(city)
            hex_embeddings = pd.concat([hex_embeddings, hex_embedding])
            highway_embeddings = pd.concat([highway_embeddings, highway_embedding])
            region_gdfs = pd.concat([region_gdfs, region_gdf])
    
        regions_list = []
        for point in tqdm(geometries):
            region = region_gdfs[region_gdfs["geometry"].contains(point)]
            if region.shape[0] == 0:
                regions_list.append(-1)
            else:
                regions_list.append(list(region.to_dict()["geometry"].keys())[0])
        
        prices["region_id"] = regions_list
        if run_type == RunTypes.GEO_MEAN_OSM.value or run_type == RunTypes.MEAN_OSM.value:
            hex_columns = [col for col in hex_embeddings.columns if col != "region_id"]
            hex_embeddings["hex_mean"] = hex_embeddings[hex_columns].mean(axis=1)
            hex_embeddings.drop(columns=hex_columns, inplace=True)
            highway_columns = [col for col in highway_embeddings.columns if col != "region_id"]
            highway_embeddings["highway_mean"] = highway_embeddings[highway_columns].mean(axis=1)
            highway_embeddings.drop(columns=highway_columns, inplace=True)
        else:
            hex_embeddings.columns = [f"hex_{col}" for col in hex_embeddings.columns]
            highway_embeddings.columns = [f"highway_{col}" for col in highway_embeddings.columns]
        prices = prices.merge(hex_embeddings, on="region_id")
        prices = prices.merge(highway_embeddings, on="region_id")


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("geo", "passthrough", [col for col in prices.columns if "hex_" in col] + \
                [col for col in prices.columns if "highway_" in col]),
        ]
    )
    print(f"Preprocessing took {time() - start_time:.2f} seconds")
    return preprocessor, prices
