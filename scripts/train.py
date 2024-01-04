import json
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts.autoencoder import Autoencoder
from scripts.dataset import get_dataloader


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


def visualize_results(test_data, autoencoder, knn, device):
    X_test, y_test = encode_data(autoencoder, test_data, device)
    y_pred = knn.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_test, bins=50, alpha=0.5, label="True prices")
    ax.hist(y_pred, bins=50, alpha=0.5, label="Predicted prices")
    ax.set_title("Distribution of predicted and true prices")
    ax.legend()
    plt.savefig("results.png")

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

    with open("report.json", "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    autoencoder = Autoencoder(
        n_data_features=31,
        n_encoder_hidden_features=128,
        n_decoder_hidden_features=128,
        n_latent_features=16,
    ).to(device)
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    train_data, test_data = get_dataloader(batch_size=32)
    knn = train_knn(train_data, autoencoder, device)
    visualize_results(test_data, autoencoder, knn, device)
    print("Done!")
