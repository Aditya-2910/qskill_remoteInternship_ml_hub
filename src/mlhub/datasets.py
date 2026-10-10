"""Dataset loaders.

* Iris ships with scikit-learn, so it always works offline.
* SMS Spam Collection is downloaded from the UCI repository and cached under ``data/``.
* California Housing is fetched through scikit-learn (cached under ``data/``).

If a download fails (no internet, corporate proxy, ...) or ``MLHUB_OFFLINE=1`` is set, small
synthetic stand-ins are generated so that the whole pipeline, the tests and the CI still run.
Every loader returns ``(dataframe, source)`` and ``source`` is stored in the model metadata,
so a model trained on stand-in data is never mistaken for one trained on the real data.
"""

from __future__ import annotations

import io
import logging
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris

from mlhub.config import Settings

log = logging.getLogger(__name__)

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_SPECIES = ["setosa", "versicolor", "virginica"]
HOUSE_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
HOUSE_TARGET = "MedHouseVal"  # in units of $100,000

# --------------------------------------------------------------------------- iris


def load_iris_df() -> tuple[pd.DataFrame, str]:
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df.columns = IRIS_FEATURES + ["species"]
    df["species"] = df["species"].map(dict(enumerate(IRIS_SPECIES)))
    return df, "sklearn.datasets.load_iris"


# --------------------------------------------------------------------------- spam


def parse_sms_zip(payload: bytes) -> pd.DataFrame:
    """Parse the UCI ``smsspamcollection.zip`` archive into a ``label,text`` dataframe."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("SMSSpamCollection"))
        raw = zf.read(name).decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in {"ham", "spam"} and text.strip():
            rows.append((label, text.strip()))
    return pd.DataFrame(rows, columns=["label", "text"])


def _download_sms(cache: Path) -> pd.DataFrame:
    import httpx

    resp = httpx.get(SMS_URL, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    df = parse_sms_zip(resp.content)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


_SPAM_TEMPLATES = [
    "WINNER!! You have been selected to receive a {prize}. Call {phone} now to claim your reward",
    "Congratulations! Your number won a {prize}. Text CLAIM to {short} to collect. T&Cs apply",
    "URGENT! Your account has been suspended. Verify your details at {url} within 24 hours",
    "FREE entry in our weekly competition to win a {prize}. Reply WIN to {short} now",
    "You have {n} unclaimed cash prize. Click {url} to redeem before it expires",
    "Get cheap loans approved today, no credit check! Call {phone} or visit {url}",
    "Hot singles in your area want to chat. Text JOIN to {short}. 18+ only, {n}p per msg",
    "Limited offer: buy one get one FREE on all {prize}s. Order now at {url}",
    "Your mobile has won a {prize}! To claim call {phone}. Offer ends tonight, hurry",
    "Final notice: you are eligible for a {n} bonus. Reply STOP to opt out or click {url}",
]
_HAM_TEMPLATES = [
    "Hey, are we still meeting at {time} for {meal}? Let me know if you are running late",
    "Can you send me the notes from {subject} class? I missed the lecture today",
    "I'll call you when I get home, around {time}. Need to talk about the weekend plan",
    "Thanks for the birthday wishes yesterday, dinner was lovely. See you at {time}",
    "Don't forget to pick up milk and bread on your way back, we are out of both",
    "The {subject} assignment is due on Friday, do you want to work on it together tonight?",
    "Running a bit late, the bus is stuck in traffic. Start {meal} without me",
    "Mom says come over on Sunday for {meal}, she is cooking your favourite",
    "Good luck with the {subject} exam tomorrow! You will do great, call me after",
    "Sorry I missed your call, I was in a meeting. Free after {time} if you want to chat",
]


def make_sample_sms(n_spam: int = 150, n_ham: int = 900, seed: int = 0) -> pd.DataFrame:
    """Small synthetic SMS dataset used only as an offline stand-in."""
    rng = random.Random(seed)
    prizes = ["holiday", "new phone", "gift voucher", "cash prize", "laptop", "cruise"]
    subjects = ["maths", "physics", "chemistry", "history", "biology", "economics"]
    meals = ["lunch", "dinner", "breakfast", "coffee"]

    def fill(tpl: str) -> str:
        return tpl.format(
            prize=rng.choice(prizes),
            phone=f"0{rng.randint(7000000000, 7999999999)}",
            short=str(rng.randint(80000, 89999)),
            url=f"http://{rng.choice(['bit.ly', 'tinyurl.com', 'claim-now.biz'])}/{rng.randint(100, 999)}",
            n=rng.choice([100, 250, 500, 1000, 5000]),
            time=f"{rng.randint(1, 11)}{rng.choice(['am', 'pm'])}",
            meal=rng.choice(meals),
            subject=rng.choice(subjects),
        )

    rows = [("spam", fill(rng.choice(_SPAM_TEMPLATES))) for _ in range(n_spam)]
    rows += [("ham", fill(rng.choice(_HAM_TEMPLATES))) for _ in range(n_ham)]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["label", "text"])


def load_spam_df(settings: Settings) -> tuple[pd.DataFrame, str]:
    cache = settings.data_dir / "sms_spam.csv"
    if cache.exists():
        return pd.read_csv(cache), "UCI SMS Spam Collection (cached)"
    if not settings.offline:
        try:
            return _download_sms(cache), "UCI SMS Spam Collection"
        except Exception as exc:  # network errors, bad archive, ...
            log.warning("Could not download the SMS Spam Collection (%s). Using synthetic sample.", exc)
    return make_sample_sms(), "synthetic sample (offline stand-in)"


# --------------------------------------------------------------------------- house


def make_synthetic_housing(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the California Housing columns (offline stand-in)."""
    rng = np.random.default_rng(seed)
    med_inc = rng.gamma(shape=4.0, scale=1.0, size=n).clip(0.5, 15)
    house_age = rng.uniform(1, 52, size=n)
    ave_rooms = rng.normal(5.4, 1.2, size=n).clip(2, 12)
    ave_bedrms = (ave_rooms * rng.uniform(0.15, 0.25, size=n)).clip(0.8, 4)
    population = rng.lognormal(mean=7.0, sigma=0.6, size=n).clip(50, 12000)
    ave_occup = rng.normal(3.0, 0.7, size=n).clip(1, 8)
    lat = rng.uniform(32.6, 41.9, size=n)
    lon = rng.uniform(-124.3, -114.4, size=n)
    coastal = np.exp(-((lon + 122.0) ** 2) / 8.0)
    value = (
        0.45 * med_inc
        + 0.01 * house_age
        + 0.12 * ave_rooms
        + 0.9 * coastal
        - 0.08 * ave_occup
        + rng.normal(0, 0.35, size=n)
    ).clip(0.15, 5.0)
    return pd.DataFrame(
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": lat,
            "Longitude": lon,
            HOUSE_TARGET: value,
        }
    )


def load_house_df(settings: Settings) -> tuple[pd.DataFrame, str]:
    if not settings.offline:
        try:
            bunch = fetch_california_housing(data_home=str(settings.data_dir / "sklearn"), as_frame=True)
            df = bunch.frame[HOUSE_FEATURES + [HOUSE_TARGET]].copy()
            return df, "sklearn.datasets.fetch_california_housing"
        except Exception as exc:
            log.warning("Could not fetch California Housing (%s). Using synthetic sample.", exc)
    return make_synthetic_housing(), "synthetic sample (offline stand-in)"
