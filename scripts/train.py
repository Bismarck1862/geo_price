import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

from .autoencoder import Autoencoder
from .dataset import get_dataloader
from .utils import TYPES_MAP, RunTypes


def encode_data(autoencoder, data, device):
    encoded_data = []
    ys = []
    for x, y in tqdm(data):
        x = x.to(device)
        encoded_x = autoencoder.encoder_forward(x)
        encoded_data.append(encoded_x.detach().cpu().numpy())
        ys.append(y.numpy())
    return np.vstack(encoded_data), np.hstack(ys)


def train_knn(train_dataloader, autoencoder, device, n_neighbors=5):
    print("Encoding data...")
    X_train, y_train = encode_data(autoencoder, train_dataloader, device)
    print("Training KNN...")
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    print("Evaluating...")
    return knn


def get_results(test_data, autoencoder, knn, device,
                run_type=RunTypes.NON_GEO.value):
    X_test, y_test = encode_data(autoencoder, test_data, device)
    y_pred = knn.predict(X_test)
    # print("TEST", y_test.tolist()[:10])
    # print("PRED", y_pred.tolist()[:10])
    # print("X_TEST", X_test.tolist()[:10])
    results = {"y_test": y_test.tolist(), "y_pred": y_pred.tolist(), "run_type": "non_geo", "x_test": X_test.tolist()}
    with open(f"results_{run_type}.json", "w") as f:
        json.dump(results, f)



def visualize_results(test_data, autoencoder, knn, device,
                      run_type=RunTypes.NON_GEO.value):
    X_test, y_test = encode_data(autoencoder, test_data, device)
    y_pred = knn.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_test, bins=50, alpha=0.5, label="True prices")
    ax.hist(y_pred, bins=50, alpha=0.5, label="Predicted prices")
    ax.set_title("Distribution of predicted and true prices")
    ax.legend()
    plt.savefig(f"results_{run_type}.png")

    mse = mean_squared_error(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    huber_loss = np.mean(
        np.where(
            np.abs(y_test - y_pred) < 1,
            0.5 * (y_test - y_pred) ** 2,
            np.abs(y_test - y_pred) - 0.5,
        )
    )

    report = {
        "Mean squared error": format(mse, ".2f"),
        "Mean absolute error": format(mae, ".2f"),
        "Mean absolute percentage error": format(mape, ".2f"),
        "Huber loss": format(huber_loss, ".2f"),
    }

    with open(f"report.jsonl", "a") as f:
        json.dump({"run type": run_type, "metrics": report}, f)
        f.write("\n")
    return y_pred


def run_training(run_type=RunTypes.NON_GEO.value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    autoencoder = Autoencoder(
        n_data_features=TYPES_MAP[run_type],
        n_encoder_hidden_features=128,
        n_decoder_hidden_features=128,
        n_latent_features=16,
    ).to(device)
    autoencoder.load_state_dict(torch.load(f"autoencoder_{run_type}.pth"))
    train_data, test_data, X_test, y_test = get_dataloader(batch_size=32, run_type=run_type)
    knn = train_knn(train_data, autoencoder, device)
    y_pred = visualize_results(test_data, autoencoder, knn, device, run_type=run_type)
    # save dataframe
    df = pd.DataFrame(X_test)
    df['price'] = y_test
    df['predicted_price'] = y_pred
    df.to_pickle(f"results_{run_type}.pkl")
    print(f"Saved results to results_{run_type}.pkl")
    print("Done!")


def run_test(run_type=RunTypes.NON_GEO.value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    autoencoder = Autoencoder(
        n_data_features=TYPES_MAP[run_type],
        n_encoder_hidden_features=128,
        n_decoder_hidden_features=128,
        n_latent_features=16,
    ).to(device)
    autoencoder.load_state_dict(torch.load(f"autoencoder_{run_type}.pth"))
    train_data, test_data = get_dataloader(batch_size=32, run_type=run_type)
    knn = train_knn(train_data, autoencoder, device)
    get_results(test_data, autoencoder, knn, device, run_type=run_type)
    print("Done!")

# if __name__ == "__main__":
#     run_training(run_type=run_type)