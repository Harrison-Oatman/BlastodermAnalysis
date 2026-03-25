"""
Determines division times for all nuclei in a dataset
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def _determine_prev_tracklets(df: pd.DataFrame) -> pd.DataFrame:
    df["parent_tracklet_id"] = df["parent_id"].map(df["tracklet_id"])
    df["parent_tracklet_id"] = df["parent_tracklet_id"].fillna(-1)
    df["prev_tracklet_id"] = -1

    for frame, group in df.groupby("frame"):
        parent_tracklet = group["parent_id"].map(df["tracklet_id"])
        parent_tracklet.fillna(-1, inplace=True)

        parent_prev_id = group["parent_id"].map(df["prev_tracklet_id"])
        parent_prev_id.fillna(-1, inplace=True)

        is_child = group["is_child"]

        prev_tracklet_id = parent_tracklet * (is_child) + parent_prev_id * (1 - is_child)
        df.loc[group.index, "prev_tracklet_id"] = prev_tracklet_id.astype(int)

    df["time_since_division"] = df.groupby("tracklet_id")["time_since_nc11"].transform(lambda x: x - x.min())

    return df

def _match_quadratic(x, y, z, range_min=-1.0, range_max=1.0, num_points=250):
    x_test = np.linspace(range_min, range_max, num_points)
    x_adj = np.expand_dims(x, 1) - np.expand_dims(x_test, 0)

    y_eval = z[0] * x_adj ** 2 + z[1] * x_adj + z[2]
    y_error = y_eval - np.expand_dims(y, 1)
    y_corrected = y_error - np.mean(y_error, axis=0)
    best_idx = np.argmin(np.sum(y_corrected ** 2, axis=0))

    x0 = x_test[best_idx]
    offset = np.mean(z[0] * (x - x0) ** 2 + z[1] * (x - x0) + z[2] - y)

    mse = np.mean((y - (z[0] * (x - x0) ** 2 + z[1] * (x - x0) + z[2] - offset)) ** 2)

    return x0, offset, mse

def _match_quadratic2(x, y, z, range_min=-1.0, range_max=1.0, num_points=250):
    x_test = np.linspace(range_min, range_max, num_points)
    x_adj = np.expand_dims(x, 1) - np.expand_dims(x_test, 0)

    y_eval = z[0] * x_adj ** 2 + z[1] * x_adj + z[2]
    y_error = y_eval / np.expand_dims(y, 1)
    y_corrected = np.expand_dims(y, 1) * y_error
    best_idx = np.argmin(np.sum(y_corrected ** 2, axis=0))

    x0 = x_test[best_idx]
    offset = np.mean(z[0] * (x - x0) ** 2 + z[1] * (x - x0) + z[2] - y)

    mse = np.mean((y - (z[0] * (x - x0) ** 2 + z[1] * (x - x0) + z[2] - offset)) ** 2)

    return x0, offset, mse

def get_division_times(df: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    df

    Returns
    -------
    tracklet_division_times: dataframe with the following columns
    """
    df = _determine_prev_tracklets(df)

    tgb = df.groupby(["frame", "prev_tracklet_id"])

    distance = (tgb[["z", "y", "x"]].max() - tgb[["z", "y", "x"]].min()).sum(axis=1).values
    time_since = np.round(tgb["time_since_division"].max().values, 2)
    cycle = tgb["cycle"].first().values
    prev_tracklet = tgb["prev_tracklet_id"].first().values

    prev_tracklet_df = pd.DataFrame({
        "prev_tracklet_id": prev_tracklet,
        "time_since_division": time_since,
        "distance": distance,
        "cycle": cycle,
        "n_points": tgb["z"].count().values,
        "nc11_time": tgb["time_since_nc11"].min().values,
        "z": tgb["z"].mean().values,
        "y": tgb["y"].mean().values,
        "x": tgb["x"].mean().values,
        "AP": tgb["AP"].mean().values,
        "theta": tgb["theta"].mean().values,
    })

    prev_tracklet_df = prev_tracklet_df[prev_tracklet_df["n_points"] == 2]
    prev_tracklet_df = prev_tracklet_df[prev_tracklet_df["time_since_division"] < 2.1]
    prev_tracklet_df = prev_tracklet_df[prev_tracklet_df["prev_tracklet_id"] != -1]

    print(prev_tracklet_df.head())

    cycle_fits = {}

    for c in [11, 12, 13, 14]:
        cycle_df = prev_tracklet_df[prev_tracklet_df["cycle"] == c]
        z = np.polyfit(cycle_df["time_since_division"], cycle_df["distance"], deg=2)
        print(f"Cycle {c}: {z}")

        cycle_fits[c] = z

        print(f"x intercepts for cycle {c}: {np.roots(z)}")

    tracklet_times = defaultdict(list)
    tracklet_tracks = df.groupby("tracklet_id")["track_id"].first()

    for tid, group in prev_tracklet_df.groupby("prev_tracklet_id"):
        distances = group["distance"].values
        time_sinces = group["time_since_division"].values
        this_cycle = group["cycle"].values[0]

        if this_cycle == 10:
            continue

        x0, offset, mse = _match_quadratic(
            time_sinces, distances, cycle_fits[this_cycle]
        )

        corrected_division_time = group["nc11_time"].min() + x0

        tracklet_times["tracklet_id"].append(tid)
        tracklet_times["track_id"].append(tracklet_tracks[tid])
        tracklet_times["division_time"].append(group["nc11_time"].min())
        tracklet_times["cycle"].append(this_cycle - 1)
        tracklet_times["corrected_division_time"].append(corrected_division_time)
        tracklet_times["z"].append(group["z"].mean())
        tracklet_times["y"].append(group["y"].mean())
        tracklet_times["x"].append(group["x"].mean())
        tracklet_times["AP"].append(group["AP"].mean())
        tracklet_times["theta"].append(group["theta"].mean())
        tracklet_times["error"].append(mse)

    tracklet_division_times = pd.DataFrame(tracklet_times)

    return tracklet_division_times

