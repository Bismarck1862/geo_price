from scripts.dataset import get_dataloader


def test_get_dataloader():
    print("Testing get_dataloader...")
    train_data, test_data = get_dataloader(batch_size=32)
    print(train_data)
    print(test_data)


if __name__ == "__main__":
    test_get_dataloader()
