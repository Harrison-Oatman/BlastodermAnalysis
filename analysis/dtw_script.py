import numpy as np
from dtaidistance import dtw_ndim
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def get_mean_traj(df, axes, n=50):
    """
    Get mean trajectory for a given DataFrame `ss` and columns `cols`.
    Interpolates the data to `n` points.
    """
    pseudo_time = np.linspace(0, 1, n)

    interpolated_axis_values = {col: list() for col in axes}

    mean_length = df.groupby("tracklet_id")[axes[0]].count().mean()

    for i, (tid, group) in enumerate(df.groupby("tracklet_id")):

        if np.abs(len(group) - mean_length) > 25:
            continue

        for col in axes:
            vals = group[col].rolling(5, 1, center=True).mean()
            vals = (vals - vals.mean()) / vals.std()

            st = np.arange(len(vals)) / len(vals)

            if np.sum(vals.isna()) > 0:
                print(f"skipping {tid} due to NaNs")
                continue

            x_interp = np.interp(pseudo_time, st, vals)
            interpolated_axis_values[col].append(x_interp)

    for col in axes:
        interpolated_axis_values[col] = np.array(interpolated_axis_values[col])
        interpolated_axis_values[col] = np.nanmean(interpolated_axis_values[col], axis=0)

    return pseudo_time, interpolated_axis_values


def map_pseudotime(warping_path, pseudotime_full, len_this_traj):
    # Collect all j's for each i
    mapping = defaultdict(list)
    for i, j in warping_path:
        mapping[i].append(j)
    # For each i, compute the mean pseudotime of mapped j's
    pseudotime_mapped = np.full(len_this_traj, np.nan)
    for i, js in mapping.items():
        pseudotime_mapped[i] = np.mean(pseudotime_full[js])
    return pseudotime_mapped

def warp_df(df, n=100):
    pseudotime_mapper = {}
    distance_mapper = {}

    for cycle in [11, 12, 13]:
        print(f"Processing cycle {cycle}")

        ss = df[df["cycle"] == cycle].copy()

        mean_length = ss.groupby("tracklet_id")["area"].count().mean()

        st, mean_trajectory = get_mean_traj(ss, ["radius", "intensity_mean"], n=n)

        y_traj = mean_trajectory["radius"]
        x_traj = mean_trajectory["intensity_mean"]

        full_traj = np.stack([y_traj, x_traj], axis=1)

        pseudotime_sequences = []

        grouped = ss.groupby('tracklet_id')
        group_keys = list(grouped.groups.keys())

        shuffle(group_keys)

        for tid in tqdm(group_keys):
            group = grouped.get_group(tid)

            if np.abs(len(group) - mean_length) > 25:
                continue

            y = group["radius"].rolling(5, 1, center=True).mean()
            x = group["intensity_mean"].rolling(5, 1, center=True).mean()


            y = (y - y.mean()) / y.std()
            x = (x - x.mean()) / x.std()

            this_traj = np.stack([y, x], axis=1)

            if np.sum(y.isna()) > 0 or np.sum(x.isna()) > 0:
                continue

            seq, dis = dtw_ndim.warping_path(this_traj.astype(np.double), full_traj.astype(np.double),
                                             include_distance=True)

            pseudotime_seq = map_pseudotime(seq, st, len(this_traj))
            pseudotime_sequences.append(pseudotime_seq)

            pseudotime_mapper.update({spot_id: pt for spot_id, pt in zip(group.index, pseudotime_seq)})
            distance_mapper.update({spot_id: dis for spot_id in group.index})

    df["pseudotime"] = df.index.map(pseudotime_mapper)
    df["distance"] = df.index.map(distance_mapper)

    return df

def main():
    base_path = Path(r"D:\Tracking\DrosophilaNucleusTracking\data\spots")
    save_path = base_path / "dtw"
    save_path.mkdir(exist_ok=True)

    for spots_path in base_path.glob("*_spots.h5"):
        spots_df = pd.read_hdf(spots_path, key="df")

        print(f"Processing {spots_path.name} with {len(spots_df)} spots")
        warp_df(spots_df, n=100)

        subsetted_df = spots_df[["pseudotime", "distance"]].copy()
        subsetted_df.to_hdf(save_path / f"{spots_path.stem}_dtw.h5", key="df", mode="w")

if __name__ == "__main__":
    main()
