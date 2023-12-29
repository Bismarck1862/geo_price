import json
import os
import warnings
from statistics import mean
from typing import Any, Dict, List

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
import yaml
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV,
                                     HalvingRandomSearchCV, RandomizedSearchCV,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from utils import write_to_markdown

warnings.filterwarnings('ignore')


def get_data(dataset_name: str) -> pd.DataFrame:
    data_dir = os.path.join("data", "prepared", dataset_name)
    data_path = os.path.join(data_dir, "data.json")
    return pd.read_json(data_path)