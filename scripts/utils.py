import os
import warnings
from enum import Enum

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DEMOG = "./data/demographic/"
DATA_PRICES = "./data/prices/"


class RunTypes(Enum):
    NON_GEO = "non_geo"
    GEO = "geo"
    OSM = "osm"
    MEAN_OSM = "mean_osm"
    GEO_OSM = "geo_osm"
    GEO_MEAN_OSM = "geo_mean_osm"


GEO_TYPES = [RunTypes.GEO.value, RunTypes.GEO_OSM.value, RunTypes.GEO_MEAN_OSM.value]
OSM_TYPES = [RunTypes.OSM.value, RunTypes.MEAN_OSM.value, RunTypes.GEO_OSM.value, RunTypes.GEO_MEAN_OSM.value]


TYPES_MAP = {
    RunTypes.NON_GEO.value: 31,
    RunTypes.GEO.value: 42,
    RunTypes.OSM.value: 73,
    RunTypes.MEAN_OSM.value: 35,
    RunTypes.GEO_OSM.value: 82,
    RunTypes.GEO_MEAN_OSM.value: 44,
}


def _file_generator(dir_path: str) -> str:
    """Yields all file and subdir paths from directory.

    :param dir_path: path to directory
    :type dir_path: str
    :rtype: str
    """
    for root, _, files in os.walk(dir_path):
        for name in files:
            root_path = os.path.join(root, name)
            yield root_path


def load_prices(city, rent=False) -> pd.DataFrame:
    dfs = []
    for file in _file_generator(DATA_PRICES):
        if rent:
            if "apartments_rent_pl_2023" in file:
                dfs.append(pd.read_csv(file))
        else:
            if "apartments_pl_2023" in file:
                dfs.append(pd.read_csv(file))
    print(f"Loaded {len(dfs)} files")
    df = pd.concat(dfs, ignore_index=True)
    if city:
        df = df[df["city"] == city]
        df.reset_index(drop=True, inplace=True)
    print(f"Number of rows: {df.shape[0]}")
    return df


def load_demographic() -> gpd.GeoDataFrame:
    dfs = []
    for file in _file_generator(DATA_DEMOG):
        if file.endswith(".shp"):
            dfs.append(gpd.read_file(file, encoding="utf-8"))

    geodf = pd.concat(dfs, ignore_index=True)
    return geodf
