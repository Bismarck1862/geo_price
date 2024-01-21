from scripts.dataset import get_dataloader
from scripts.geo_emb import get_embeddings
from scripts.utils import RunTypes


def test_get_dataloader(run_type=RunTypes.NON_GEO.value):
    print("Testing get_dataloader...")
    train_data, test_data = get_dataloader(city="wroclaw",
                                           batch_size=32,
                                           run_type=run_type)
    print(train_data)
    print(test_data)

def test_geo_emb(city_name):
    print("Testing geo_emb...")
    hex_embeddings, highway_embeddings = get_embeddings(city_name)
    print(hex_embeddings)
    print(highway_embeddings)


if __name__ == "__main__":
    test_get_dataloader(run_type=RunTypes.OSM.value)
    # test_geo_emb(city_name="Gdynia, Poland")