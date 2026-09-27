"""Synthetic data with the schema of the Kaggle ``ks-projects-201801.csv`` file."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAIN_CATEGORIES = {
    "Games": ["Tabletop Games", "Video Games", "Playing Cards"],
    "Music": ["Indie Rock", "Hip-Hop", "Classical Music"],
    "Technology": ["Gadgets", "Apps", "Hardware"],
    "Art": ["Painting", "Illustration"],
    "Food": ["Restaurants", "Drinks"],
}
COUNTRIES = {"US": "USD", "GB": "GBP", "DE": "EUR", "CA": "CAD"}
STATES_OTHER = ["canceled", "live", "suspended", "undefined"]


def make_raw(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Projects launched between 2016 and early 2018 with a learnable success signal."""
    rng = np.random.default_rng(seed)
    main = rng.choice(list(MAIN_CATEGORIES), n)
    category = np.array([rng.choice(MAIN_CATEGORIES[m]) for m in main])
    country = rng.choice(list(COUNTRIES), n, p=[0.7, 0.12, 0.1, 0.08])
    start = pd.Timestamp("2016-01-01").value // 10**9
    end = pd.Timestamp("2018-01-02").value // 10**9
    launched = pd.to_datetime(rng.integers(start, end, n), unit="s")
    duration = rng.choice([15, 30, 30, 30, 45, 60], n)
    deadline = (launched + pd.to_timedelta(duration, unit="D")).normalize()
    goal = np.round(np.exp(rng.normal(8.5, 1.5, n)), 2)

    category_effect = pd.Series(main).map(
        {"Games": 0.8, "Music": 0.3, "Technology": -0.4, "Art": 0.1, "Food": -0.8}).to_numpy()
    logit = 3.0 - 0.4 * np.log1p(goal) + category_effect - 0.02 * (duration - 30)
    success = rng.random(n) < 1 / (1 + np.exp(-logit))
    state = np.where(success, "successful", "failed").astype(object)
    other = rng.random(n) < 0.1
    state[other] = rng.choice(STATES_OTHER, other.sum())

    pledged = np.where(success, goal * rng.uniform(1.0, 3.0, n), goal * rng.uniform(0, 0.9, n))
    backers = (pledged / rng.uniform(20, 80, n)).astype(int)
    df = pd.DataFrame({
        "ID": np.arange(n) + 1_000_000,
        "name": [f"Project {i}" for i in range(n)],
        "category": category,
        "main_category": main,
        "currency": [COUNTRIES[c] for c in country],
        "deadline": deadline.strftime("%Y-%m-%d"),
        "goal": goal,
        "launched": launched.strftime("%Y-%m-%d %H:%M:%S"),
        "pledged": pledged.round(2),
        "state": state,
        "backers": backers,
        "country": country,
        "usd pledged": pledged.round(2),
        "usd_pledged_real": pledged.round(2),
        "usd_goal_real": goal,
    })
    # A placeholder project as found in the real data.
    df.loc[0, "launched"] = "1970-01-01 01:00:00"
    return df
