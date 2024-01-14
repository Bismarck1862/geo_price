from scripts.dataset import get_dataloader
from scripts.geo_emb import get_embeddings


def test_get_dataloader(geo=False):
    print("Testing get_dataloader...")
    train_data, test_data = get_dataloader(batch_size=32, geo=geo)
    print(train_data)
    print(test_data)

def test_geo_emb():
    print("Testing geo_emb...")
    hex_embeddings, highway_embeddings = get_embeddings()
    print(hex_embeddings)
    print(highway_embeddings)


if __name__ == "__main__":
    test_get_dataloader(geo=True)
    # test_geo_emb()
