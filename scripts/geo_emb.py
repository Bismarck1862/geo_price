import warnings

from pytorch_lightning import seed_everything
from srai.embedders import Hex2VecEmbedder, Highway2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMNetworkType, OSMOnlineLoader, OSMWayLoader
from srai.neighbourhoods import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

HEX_TAGS = {
        "leisure": "park",
        "landuse": "forest",
        "amenity": ["bar", "restaurant", "cafe"],
        "water": "river",
        "sport": "soccer",
    }


def load_area_and_regions(city_name="Wrocław, Poland"):
    area_gdf = geocode_to_region_gdf(city_name)

    regionalizer = H3Regionalizer(resolution=9)
    regions_gdf = regionalizer.transform(area_gdf)

    return area_gdf, regions_gdf


def load_features(area_gdf, tags=HEX_TAGS):
    loader = OSMOnlineLoader()
    features_gdf = loader.load(area_gdf, tags)

    return features_gdf


def load_edges(area_gdf):
    loader = OSMWayLoader(OSMNetworkType.DRIVE)
    _, edges_gdf = loader.load(area_gdf)

    return edges_gdf


def hex2vec(regions_gdf, features_gdf):
    print("Running hex2vec...")
    neighbourhood = H3Neighbourhood(regions_gdf)
    embedder = Hex2VecEmbedder([15, 10])
    joiner = IntersectionJoiner()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embeddings = embedder.fit_transform(
            regions_gdf,
            features_gdf,
            joiner.transform(regions_gdf, features_gdf),
            neighbourhood,
            trainer_kwargs={"max_epochs": 5, "accelerator": "cpu"},
            batch_size=100,
        )
    return embeddings


def highway2vec(regions_gdf, edges_gdf):
    print("Running highway2vec...")
    embedder = Highway2VecEmbedder()
    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, edges_gdf)
    embeddings = embedder.fit_transform(regions_gdf,
                                        edges_gdf,
                                        joint_gdf)
    return embeddings


def get_embeddings(city_name="Wrocław, Poland"):
    area_gdf, regions_gdf = load_area_and_regions(city_name)
    regionalizer = H3Regionalizer(9)
    regions_gdf = regionalizer.transform(area_gdf)
    features_gdf = load_features(area_gdf)
    edges_gdf = load_edges(area_gdf)
    hex_embeddings = hex2vec(regions_gdf, features_gdf)
    highway_embeddings = highway2vec(regions_gdf, edges_gdf)
    return hex_embeddings, highway_embeddings
