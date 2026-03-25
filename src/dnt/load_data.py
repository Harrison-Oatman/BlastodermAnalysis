from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from h5py import File
from typing import Optional



def load_spots_data(spots_directory: Path, included: Optional[list] = None):
    spots_dfs = []
    stems = []

    directories = list(spots_directory.glob("*_spots.h5"))
    print([f"{d.stem}" for d in directories])

    for i, spots_path in tqdm(enumerate(spots_directory.glob("*_spots.h5")), desc="reading spots dfs"):
        stem = spots_path.stem
        embryo_name = stem.split("_spots")[0]

        if included is not None and embryo_name not in included:
            continue

        print(spots_path.stem)

        spots_df: pd.DataFrame = pd.read_hdf(spots_path, key="lineages")
        stems.append(spots_path.stem)
        spots_df["source"] = i

        print(spots_df["frame"].min(), spots_df["frame"].max())
        # print(spots_df["id_prev"].value_counts())
        # spots_df = spots_df.set_index("id_prev")


        # """
        # Get metadata
        # """
        # metadata = {}
        # with File(spots_path, "r") as f:
        #     m = f["metadata"]
        #     metadata.update(m.attrs)

        # metadatas.append(metadata)

        # """
        # Calculate cycle and tracklet_id
        # """
        #
        spots_df["n_children"] = spots_df.index.map(spots_df["parent_id"].value_counts())
        spots_df["is_parent"] = spots_df["n_children"] > 1
        spots_df["is_child"] = (spots_df["parent_id"].map(spots_df["is_parent"])).fillna(0)
        #
        # spots_df["tracklet_id"] = spots_df.index.to_series()
        #
        # for frame, group in spots_df.groupby("frame"):
        #     group = group[~group["is_child"]]
        #     spots_df.loc[group.index, "tracklet_id"] = group["parent_id"].map(spots_df["tracklet_id"])
        #
        # track_id_remap = {old_id: new_id for new_id, old_id in enumerate(spots_df["tracklet_id"].unique(), start=1)}
        # spots_df["tracklet_id"] = spots_df["tracklet_id"].map(track_id_remap)
        #
        # cycle_starts = metadata["cycle_starts"]
        # print(cycle_starts, spots_df["frame"].min(), spots_df["frame"].max())
        #
        # t_cycle = (
        #     spots_df.groupby("tracklet_id")["frame"]
        #     .min()
        #     .apply(lambda time: np.argmax(cycle_starts > time))
        # )
        #
        # spots_df["cycle"] = spots_df["tracklet_id"].map(t_cycle) + 9
        #
        # if "time_since_nc11" not in spots_df.columns:
        #     nc11_start = spots_df[spots_df["cycle"] == 11].groupby("tracklet_id")["time"].min().mean()
        #     spots_df["time_since_nc11"] = spots_df["time"] - nc11_start
        #
        # """
        # Include dynamic time warp data
        # """
        # dtw_path = spots_directory / "dtw" / f"{spots_path.stem}_dtw.h5"
        # if dtw_path.exists():
        #     dtw_update = pd.read_hdf(dtw_path, key="df")
        #     spots_df = pd.merge(
        #         spots_df, dtw_update, left_index=True, right_index=True, how="left"
        #     )
        #     spots_df["cycle_pseudotime"] = spots_df["pseudotime"] + spots_df["cycle"] - 11
        # else:
        #     spots_df["distance"] = np.nan  # rename this later
        #     spots_df["pseudotime"] = np.nan

        """
        Calculate displacement
        """

        spots_df["dt"] = spots_df["time"] - spots_df["parent_id"].map(spots_df["time"])

        cols = ["x", "y", "z", "AP", "theta"]
        for col in cols:
            spots_df[f"d{col}"] = (spots_df[col] - spots_df["parent_id"].map(spots_df[col])) / spots_df["dt"]

        spots_df["dtot"] = np.sqrt(spots_df["dx"] ** 2 + spots_df["dy"] ** 2 + spots_df["dz"] ** 2)
        spots_df["dAP_abs"] = spots_df["dAP"].abs()

        spots_dfs.append(spots_df)

    return spots_dfs, stems