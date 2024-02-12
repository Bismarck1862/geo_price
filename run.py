from scripts.encode import run_encoding
from scripts.train import run_training, run_test
from scripts.utils import RunTypes


if __name__ == "__main__":
    run_type=RunTypes.NON_GEO.value

    run_encoding(run_type=run_type)
    run_training(run_type=run_type)
    # run_test(run_type=run_type)
