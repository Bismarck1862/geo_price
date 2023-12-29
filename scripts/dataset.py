from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .prepare import preprocess

RANDOM_STATE = 42


def get_dataloader(batch_size=32):
    """Load data into DataLoader."""
    preprocessor, data_df = preprocess()
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(data_df.drop('price', axis=1),
                                                        data_df['price'], test_size=0.3,
                                                        random_state=RANDOM_STATE)
    print("Vectorizing data...")
    preprocessor.fit(X_train)
    X_train_vectorized = preprocessor.transform(X_train)
    X_test_vectorized = preprocessor.transform(X_test)

    train_data = DataLoader(
        X_train_vectorized,
        batch_size=batch_size,
        shuffle=True)
    test_data = DataLoader(
        X_test_vectorized,
        batch_size=batch_size,
        shuffle=True)
    return train_data, test_data, y_train, y_test
