import concurrent.futures
import itertools
import time
from pathlib import Path

import geopandas as gpd
import jsonlines
import pandas as pd
import requests
from pandas import Timedelta
from searchtweets import ResultStream, gen_request_parameters, load_credentials
from tqdm import tqdm

SEARCH_ARGS = load_credentials(
    "~/.twitter_keys.yaml", yaml_key="search_tweets_v2", env_overwrite=False
)
FLOOD_DIR = Path("data/floods")


def get_flood_areas(url, path, floods):
    r = requests.get(url)
    areas = pd.DataFrame(r.json()["items"])

    if not path.exists():
        tqdm.pandas()
        areas["geometry"] = areas["polygon"].progress_apply(
            lambda x: gpd.read_file(x)["geometry"]
        )
        areas = gpd.GeoDataFrame(areas, geometry="geometry")
        areas.to_file(path, driver="GPKG")
    else:
        areas = gpd.read_file(path)

    areas = areas[["fwdCode", "lat", "long", "geometry"]]
    merged = floods.merge(areas, left_on="CODE", right_on="fwdCode")
    merged = merged[["DATE", "TYPE", "lat", "long", "geometry"]]

    return gpd.GeoDataFrame(merged, geometry="geometry")


def get_tweets(row, flood_related: bool):
    idx, row = row

    start_time = str(row["DATE"] - pd.Timedelta("7 day"))[:-9] + "T00:00"
    end_time = str(row["DATE"] + pd.Timedelta("8 day"))[:-9] + "T00:00"
    bounds = row.geometry.bounds

    bounding_box = f"[{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}]"
    keywords = [
        "flood",
        "floods",
        "flooding",
        "flooded",
        "rain",
        "raining",
        "rains",
        "rained",
        "storm",
        "stormy",
        "thunder",
        "lightning",
    ]

    if flood_related:
        keywords = "(" + " OR ".join(keywords) + ")"
    else:
        keywords = "-" + " -".join(keywords)

    query = gen_request_parameters(
        keywords
        + f" has:geo bounding_box:{bounding_box} -is:retweet -is:reply -is:quote",
        granularity=None,
        user_fields="location",
        tweet_fields="created_at",
        place_fields="contained_within,country,country_code,full_name,geo,id,name,place_type",
        start_time=start_time,
        end_time=end_time,
    )
    rs = ResultStream(
        **SEARCH_ARGS,
        request_parameters=query,
        max_results=10_000,
        max_tweets=10_000,
        max_pages=10_000,
    )

    # when rate limit hits, should stop for 15 mins
    tweets = []
    for collection in rs.stream():
        collection = collection["data"]
        for tweet in collection:
            tweet["warning_time"] = row["DATE"].strftime("%Y-%m-%dT%H:%M:%S")
            tweet["bounding_box"] = bounds
            tweet["idx"] = idx
            tweet["label"] = "FLOOD" if flood_related else "NOT_FLOOD"
            tweets.append(tweet)
    return tweets


def label_day(row):
    if (row["diff_date"] == Timedelta("-1 days 00:00:00")) | (
        row["diff_date"] == Timedelta("0 days 00:00:00")
    ):
        return "FLOOD"
    else:
        return "NOT_FLOOD"


if __name__ == "__main__":
    # https://environment.data.gov.uk/portalstg/home/item.html?id=da003769a0c3490981bfd426d3253540
    floods = pd.read_csv(FLOOD_DIR / "202104_flood_warnings.csv", parse_dates=[0])
    flood_areas = get_flood_areas(
        url="http://environment.data.gov.uk/flood-monitoring/id/floodAreas?_limit=10000",
        path=FLOOD_DIR / "flood_areas.gpkg",
        floods=floods,
    )

    flood_warnings = flood_areas[
        (flood_areas["TYPE"] == "Severe Flood Warning")
        & (flood_areas["DATE"] >= "2010-01-01")
    ]

    t1 = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(get_tweets)
        flood_tweets = list(
            tqdm(
                executor.map(
                    get_tweets, flood_warnings.iterrows(), [True] * len(flood_warnings)
                ),
                total=len(flood_warnings),
            ),
        )
    flood_tweets = list(itertools.chain.from_iterable(flood_tweets))

    with jsonlines.open(FLOOD_DIR / "flood_tweets.jsonl", mode="w") as writer:
        for tweet in flood_tweets:
            writer.write(tweet)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(get_tweets)
        non_flood_tweets = list(
            tqdm(
                executor.map(
                    get_tweets, flood_warnings.iterrows(), [False] * len(flood_warnings)
                ),
                total=len(flood_warnings),
            ),
        )
    t2 = time.time()
    tdiff = t2 - t1
    f = open(FLOOD_DIR / "time_taken.txt", "w")
    f.write(str(tdiff))

    non_flood_tweets = list(itertools.chain.from_iterable(non_flood_tweets))

    with jsonlines.open(FLOOD_DIR / "flood_tweets.jsonl", mode="a") as writer:
        for tweet in non_flood_tweets:
            writer.write(tweet)

    with jsonlines.open(FLOOD_DIR / "flood_tweets.jsonl", mode="r") as reader:
        tweets = pd.DataFrame(list(reader))

    tweets["created_at"] = pd.to_datetime(tweets["created_at"].str[:-14])
    tweets["warning_time"] = pd.to_datetime(tweets["warning_time"].str[:-9])

    tweets = tweets.sort_values("created_at")
    tweets["diff_date"] = tweets["created_at"] - tweets["warning_time"]
    tweets["day_label"] = tweets.apply(lambda x: label_day(x), axis=1)

    tweets.to_csv(FLOOD_DIR / "flood_tweets.csv", index=False)
