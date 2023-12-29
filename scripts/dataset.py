import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .prepare import preprocess

RANDOM_STATE = 42


def get_dataloader(city=None, batch_size=32):
    """Load data into DataLoader."""
    preprocessor, data_df = preprocess(city)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        data_df.drop("price", axis=1),
        data_df["price"],
        test_size=0.3,
        random_state=RANDOM_STATE,
    )
    print("Vectorizing data...")
    preprocessor.fit(X_train)
    X_train_vectorized = preprocessor.transform(X_train)
    X_test_vectorized = preprocessor.transform(X_test)

    train_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_train_vectorized), torch.Tensor(y_train.values)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_test_vectorized), torch.Tensor(y_test.values)
    )

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_data, test_data
