from scripts.encode import run_encoding
from scripts.train import run_training
from scripts.utils import RunTypes


if __name__ == "__main__":
    run_type=RunTypes.GEO_OSM.value
    run_encoding(run_type=run_type)
    run_training(run_type=run_type)
