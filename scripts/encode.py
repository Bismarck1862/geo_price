import numpy as np
import torch
from tqdm import tqdm

from .autoencoder import Autoencoder
from .dataset import get_dataloader
from .utils import TYPES_MAP, RunTypes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)


def train_encoder(autoencoder, data, epochs=20, lr=0.001):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    train_metrics = {"loss": [], "epoch": []}
    for epoch in tqdm(range(epochs)):
        loss_list = np.array([])
        print(data.__iter__())
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum()
            loss.backward()
            opt.step()
            loss_list = np.append(loss_list, loss.item())
        mean_loss = np.mean(loss_list)
        train_metrics["loss"].append(mean_loss)
        train_metrics["epoch"].append(epoch)
        print(f"Epoch [{epoch+1}/{epochs}], avg_loss: {mean_loss:.4f}")
    return autoencoder, train_metrics


def run_encoding(run_type=RunTypes.NON_GEO.value):
    autoencoder = Autoencoder(
        n_data_features=TYPES_MAP[run_type],
        n_encoder_hidden_features=128,
        n_decoder_hidden_features=128,
        n_latent_features=16,
    ).to(device)
    train_data, _, _, _ = get_dataloader(batch_size=32, run_type=run_type)
    print("Training...")
    autoencoder, train_metrics = train_encoder(
        autoencoder, train_data, epochs=100, lr=0.0001
    )
    save_path = f"autoencoder_{run_type}.pth"
    torch.save(autoencoder.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    print("Training metrics:", train_metrics)
    print("Done!")


# if __name__ == "__main__":
#     run_encoding(geo=True)
