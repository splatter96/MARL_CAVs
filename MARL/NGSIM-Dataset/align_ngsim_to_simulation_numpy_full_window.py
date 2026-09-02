#!/usr/bin/env python3
"""
Align NGSIM US-101 trajectories to a three-lane weaving simulation.

Target simulation geometry
--------------------------
- Through lane 1 center: y = 0 m
- Through lane 2 center: y = 4 m
- Weaving/ramp lane center: y = 8 m in the parallel section
- Main road: x = 0 ... 450 m
- Parallel weaving section: x = 230 ... 310 m

NGSIM lane interpretation used here
-----------------------------------
- Lane 4 -> first simulated through lane
- Lane 5 -> second simulated through lane
- Lane 6 -> parallel weaving/auxiliary lane
- Lane 7 -> on-ramp approach
- Lane 8 -> off-ramp departure

The transformation is data-driven:
1. Keep every vehicle that is observed at least once on NGSIM lanes 4-8.
   Its complete trajectory is retained by default.
2. Convert NGSIM feet-based quantities to SI units.
3. Map the observed longitudinal extent of lane 6 exactly to x=230...310 m.
4. Estimate the lane-4, lane-5 and lane-6 centerlines from the dataset.
5. Straighten the two through lanes to y=0 and y=4 m.
6. In the weaving section, map lane 6 to y=8 m.
7. Before/after the weaving section, preserve the real lateral shape of
   lanes 7 and 8, so the on-ramp/off-ramp remain diagonal.

Output
------
One NumPy ``.npy`` file is written per retained vehicle. Each file contains an
``N x 4`` floating-point array with columns:

    [time_s, x_m, y_m, speed_kph]

``time_s`` is measured from the start of the complete input dataset, so the
time axes of all vehicle files are directly comparable. ``Global_Time`` is used
when available; otherwise NGSIM's 10 Hz ``Frame_ID`` is used.

Example
-------
Export trajectories that are present for the complete interval from 15 s to
30 s of the dataset:

python align_ngsim_to_simulation_numpy_full_window.py trajectories-0820am-0835am.csv \
    --output-dir trajectories_numpy \
    --start-time 15 \
    --end-time 30 \
    --preview aligned_preview.png

A vehicle is exported only if its trajectory starts at or before ``start-time``
and ends at or after ``end-time``. The retained trajectory is then cropped to
the inclusive requested interval. For example, a vehicle whose trajectory is
only available from 20 s to 40 s is excluded for a requested interval of
15 s to 30 s. If either bound is omitted, only the specified side is checked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEET_TO_METERS = 0.3048

DEFAULT_VALID_LANES = (4, 5, 6, 7, 8)
DEFAULT_THROUGH_LANES = (4, 5)
DEFAULT_WEAVING_LANE = 6

DEFAULT_SIM_LANE_WIDTH = 4.0
DEFAULT_SIM_WEAVE_START_X = 230.0
DEFAULT_SIM_WEAVE_END_X = 310.0
DEFAULT_SIM_ROAD_START_X = 0.0
DEFAULT_SIM_ROAD_END_X = 450.0

REQUIRED_COLUMNS = {
    "Vehicle_ID",
    "Frame_ID",
    "Local_X",
    "Local_Y",
    "Lane_ID",
    "v_Vel",
}


def convert_ngsim_to_si(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NGSIM feet-based quantities to SI units in-place on a copy."""
    df = df.copy()

    # Distances / dimensions stored in feet.
    length_columns = (
        "Local_X",
        "Local_Y",
        "Global_X",
        "Global_Y",
        "v_Length",
        "v_Width",
        "Space_Hdwy",
    )

    for column in length_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce") * FEET_TO_METERS

    # NGSIM velocities are ft/s and accelerations are ft/s^2.
    if "v_Vel" in df.columns:
        df["v_Vel"] = pd.to_numeric(df["v_Vel"], errors="coerce") * FEET_TO_METERS

    if "v_Acc" in df.columns:
        df["v_Acc"] = pd.to_numeric(df["v_Acc"], errors="coerce") * FEET_TO_METERS

    return df


def filter_relevant_vehicles(
    df: pd.DataFrame,
    valid_lanes=DEFAULT_VALID_LANES,
    drop_samples_outside_valid_lanes: bool = False,
) -> pd.DataFrame:
    """
    Keep vehicles that are observed at least once on one of valid_lanes.

    By default their complete trajectories are retained. This means that if,
    for example, a vehicle later changes from lane 4 to lane 3, those lane-3
    samples remain part of its trajectory.

    Set drop_samples_outside_valid_lanes=True if the output should contain
    only samples whose Lane_ID is itself in 4-8.
    """
    valid_lanes = set(valid_lanes)

    relevant_vehicle_ids = df.loc[
        df["Lane_ID"].isin(valid_lanes), "Vehicle_ID"
    ].unique()

    filtered = df[df["Vehicle_ID"].isin(relevant_vehicle_ids)].copy()

    if drop_samples_outside_valid_lanes:
        filtered = filtered[filtered["Lane_ID"].isin(valid_lanes)].copy()

    return filtered


def estimate_lane_centerline(
    df: pd.DataFrame,
    lane_id: int,
    bin_width_m: float = 10.0,
    smoothing_bins: int = 5,
):
    """
    Estimate lateral lane-center position Local_X(Local_Y).

    Dense NGSIM trajectory samples are grouped into longitudinal bins. The
    median Local_X in each bin gives a robust estimate of the lane center.
    A rolling median suppresses local trajectory noise.
    """
    lane = df.loc[df["Lane_ID"] == lane_id, ["Local_Y", "Local_X"]].dropna().copy()

    if lane.empty:
        raise ValueError(f"No samples found for Lane_ID={lane_id}.")

    y0 = lane["Local_Y"].min()

    lane["_bin"] = np.floor((lane["Local_Y"] - y0) / bin_width_m).astype(int)

    center = (
        lane.groupby("_bin", sort=True)
        .agg(
            longitudinal=("Local_Y", "median"),
            lateral=("Local_X", "median"),
            count=("Local_X", "size"),
        )
        .reset_index(drop=True)
    )

    if smoothing_bins > 1:
        center["lateral"] = (
            center["lateral"]
            .rolling(
                window=smoothing_bins,
                center=True,
                min_periods=1,
            )
            .median()
        )

    return (
        center["longitudinal"].to_numpy(dtype=float),
        center["lateral"].to_numpy(dtype=float),
    )


def interpolate_centerline(query_y, center_y, center_x):
    """Interpolate a lane centerline, using endpoint values outside support."""
    return np.interp(
        np.asarray(query_y, dtype=float),
        center_y,
        center_x,
        left=center_x[0],
        right=center_x[-1],
    )


def build_alignment(
    full_df_si: pd.DataFrame,
    through_lane_1: int = 4,
    through_lane_2: int = 5,
    weaving_lane: int = 6,
    sim_lane_width: float = DEFAULT_SIM_LANE_WIDTH,
    sim_weave_start_x: float = DEFAULT_SIM_WEAVE_START_X,
    sim_weave_end_x: float = DEFAULT_SIM_WEAVE_END_X,
    centerline_bin_width_m: float = 10.0,
):
    """
    Derive the longitudinal and lateral transformation from the dataset.

    Returns a dictionary containing the lane-centerline estimates and the
    longitudinal affine transformation parameters.
    """
    lane6 = full_df_si.loc[
        full_df_si["Lane_ID"] == weaving_lane, ["Local_Y", "Local_X"]
    ].dropna()

    if lane6.empty:
        raise ValueError(
            f"Cannot align the weaving section because Lane_ID={weaving_lane} "
            "does not exist in the dataset."
        )

    ngsim_weave_start_y = float(lane6["Local_Y"].min())
    ngsim_weave_end_y = float(lane6["Local_Y"].max())

    if ngsim_weave_end_y <= ngsim_weave_start_y:
        raise ValueError("Invalid longitudinal extent for the NGSIM weaving lane.")

    longitudinal_scale = (
        (sim_weave_end_x - sim_weave_start_x)
        / (ngsim_weave_end_y - ngsim_weave_start_y)
    )

    c4_y, c4_x = estimate_lane_centerline(
        full_df_si,
        through_lane_1,
        bin_width_m=centerline_bin_width_m,
    )
    c5_y, c5_x = estimate_lane_centerline(
        full_df_si,
        through_lane_2,
        bin_width_m=centerline_bin_width_m,
    )
    c6_y, c6_x = estimate_lane_centerline(
        full_df_si,
        weaving_lane,
        bin_width_m=centerline_bin_width_m,
    )

    return {
        "through_lane_1": through_lane_1,
        "through_lane_2": through_lane_2,
        "weaving_lane": weaving_lane,
        "sim_lane_width": sim_lane_width,
        "sim_weave_start_x": sim_weave_start_x,
        "sim_weave_end_x": sim_weave_end_x,
        "ngsim_weave_start_y": ngsim_weave_start_y,
        "ngsim_weave_end_y": ngsim_weave_end_y,
        "longitudinal_scale": longitudinal_scale,
        "c4_y": c4_y,
        "c4_x": c4_x,
        "c5_y": c5_y,
        "c5_x": c5_x,
        "c6_y": c6_y,
        "c6_x": c6_x,
    }


def transform_positions(df: pd.DataFrame, alignment: dict) -> pd.DataFrame:
    """
    Add Sim_X and Sim_Y coordinates to a metric NGSIM dataframe.

    Sim_X:
        affine longitudinal mapping chosen so the NGSIM lane-6 interval maps
        exactly onto x=230...310 m.

    Sim_Y:
        a continuous, position-based lateral transformation. It does not use
        the row's Lane_ID, so lane changes do not create artificial jumps.
    """
    out = df.copy()

    local_y = out["Local_Y"].to_numpy(dtype=float)
    local_x = out["Local_X"].to_numpy(dtype=float)

    # ----- Longitudinal transform -----
    out["Sim_X"] = (
        alignment["sim_weave_start_x"]
        + (local_y - alignment["ngsim_weave_start_y"])
        #* alignment["longitudinal_scale"]
    )

    # ----- Lateral transform -----
    # Straighten the mainline using the local centers of lanes 4 and 5.
    c4 = interpolate_centerline(
        local_y, alignment["c4_y"], alignment["c4_x"]
    )
    c5 = interpolate_centerline(
        local_y, alignment["c5_y"], alignment["c5_x"]
    )

    width = alignment["sim_lane_width"]

    spacing_45 = c5 - c4
    if np.any(spacing_45 <= 0.25):
        raise ValueError("Invalid estimated spacing between NGSIM lanes 4 and 5.")

    # Default mapping:
    # lane 4 center -> 0 m
    # lane 5 center -> 4 m
    # positions farther outside lane 5 are extrapolated using the same spacing.
    sim_y = width * (local_x - c4) / spacing_45

    # Inside the actual NGSIM lane-6 interval, use lane 6 as an additional
    # anchor. This makes the three lane centers 0, 4 and 8 m.
    in_parallel_section = (
        (local_y >= alignment["ngsim_weave_start_y"])
        & (local_y <= alignment["ngsim_weave_end_y"])
    )

    c6 = interpolate_centerline(
        local_y, alignment["c6_y"], alignment["c6_x"]
    )
    spacing_56 = c6 - c5

    # Apply the second interval only on the outboard side of lane 5.
    # This keeps the transformation continuous and preserves lane changes.
    use_56 = (
        in_parallel_section
        & (local_x > c5)
        & (spacing_56 > 0.25)
    )

    sim_y[use_56] = (
        width
        + width
        * (local_x[use_56] - c5[use_56])
        / spacing_56[use_56]
    )

    out["Sim_Y"] = sim_y

    return out


def add_sim_lane_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simulation-lane label.

    0 -> NGSIM lane 4
    1 -> NGSIM lane 5
    2 -> NGSIM lanes 6, 7, 8
    -1 -> any retained sample from another lane
    """
    out = df.copy()

    out["Sim_Lane_ID"] = np.select(
        [
            out["Lane_ID"] == 4,
            out["Lane_ID"] == 5,
            out["Lane_ID"].isin([6, 7, 8]),
        ],
        [0, 1, 2],
        default=-1,
    ).astype(int)

    return out


def create_preview(
    df: pd.DataFrame,
    output_path: Path,
    sim_road_start_x: float,
    sim_road_end_x: float,
    sim_weave_start_x: float,
    sim_weave_end_x: float,
    lane_width: float,
    max_vehicles: int = 250,
    max_points_per_vehicle: int = 200,
):
    """Create a lightweight plot for checking the transformed geometry."""
    import matplotlib.pyplot as plt

    vehicle_ids = df["Vehicle_ID"].drop_duplicates().to_numpy()

    if len(vehicle_ids) > max_vehicles:
        # Evenly sample vehicles across the dataset for a representative preview.
        indices = np.linspace(
            0, len(vehicle_ids) - 1, max_vehicles
        ).astype(int)
        vehicle_ids = vehicle_ids[indices]

    preview = df[df["Vehicle_ID"].isin(vehicle_ids)]

    fig, ax = plt.subplots(figsize=(14, 7))

    for _, vehicle in preview.groupby("Vehicle_ID", sort=False):
        if len(vehicle) > max_points_per_vehicle:
            indices = np.linspace(
                0, len(vehicle) - 1, max_points_per_vehicle
            ).astype(int)
            vehicle = vehicle.iloc[np.unique(indices)]

        ax.plot(
            vehicle["Sim_X"],
            vehicle["Sim_Y"],
            linewidth=0.7,
            alpha=0.3,
        )

    # Target simulation lane-center geometry.
    ax.plot(
        [sim_road_start_x, sim_road_end_x],
        [0.0, 0.0],
        linewidth=2.0,
        label="Through lane 1 center",
    )
    ax.plot(
        [sim_road_start_x, sim_road_end_x],
        [lane_width, lane_width],
        linewidth=2.0,
        label="Through lane 2 center",
    )
    ax.plot(
        [sim_weave_start_x, sim_weave_end_x],
        [2.0 * lane_width, 2.0 * lane_width],
        linewidth=2.0,
        label="Parallel weaving lane center",
    )

    ax.set_xlim(sim_road_start_x, sim_road_end_x)
    ax.set_xlabel("Simulation x [m]")
    ax.set_ylabel("Simulation y [m]")
    ax.set_title("NGSIM US-101 trajectories aligned to simulation geometry")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def add_dataset_time(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Add time in seconds relative to the start of the complete NGSIM dataset."""
    out = df.copy()

    if "Global_Time" in raw_df.columns and "Global_Time" in out.columns:
        global_time_all = pd.to_numeric(raw_df["Global_Time"], errors="coerce")
        dataset_start_ms = global_time_all.min()
        if pd.notna(dataset_start_ms):
            out["Time_s"] = (
                pd.to_numeric(out["Global_Time"], errors="coerce")
                - dataset_start_ms
            ) / 1000.0
            return out

    # NGSIM trajectory data is sampled at 10 Hz. Frame_ID is global within the
    # recording, so subtracting the minimum frame gives time since dataset start.
    # frame_all = pd.to_numeric(raw_df["Frame_ID"], errors="coerce")
    # dataset_start_frame = frame_all.min()
    # if pd.isna(dataset_start_frame):
    #     raise ValueError("Cannot determine dataset start time from Frame_ID.")

    # out["Time_s"] = (
    #     pd.to_numeric(out["Frame_ID"], errors="coerce") - dataset_start_frame
    # ) * 0.1
    # return out


def filter_vehicles_covering_time_window(
    df: pd.DataFrame,
    start_time: float | None = None,
    end_time: float | None = None,
    tolerance_s: float = 1e-6,
) -> pd.DataFrame:
    """Keep only vehicles that cover the complete requested time window.

    The filtering is performed in two steps:

    1. Determine the first and last available timestamp of every vehicle.
       A vehicle is retained only if it is already present at ``start_time``
       and still present at ``end_time``.
    2. After this vehicle-level filtering, crop the retained trajectories to
       the inclusive interval ``[start_time, end_time]``.

    Thus, for a requested interval of 15...30 s, a trajectory spanning
    20...40 s is rejected, while one spanning 10...35 s is retained and its
    saved samples are cropped to 15...30 s.

    ``tolerance_s`` only protects comparisons against floating-point rounding.
    """
    if start_time is not None and start_time < 0:
        raise ValueError("--start-time must be >= 0 seconds.")

    if end_time is not None and end_time < 0:
        raise ValueError("--end-time must be >= 0 seconds.")

    if (
        start_time is not None
        and end_time is not None
        and end_time < start_time
    ):
        raise ValueError("--end-time must be greater than or equal to --start-time.")

    if df.empty or (start_time is None and end_time is None):
        return df.copy()

    # First/last dataset-relative timestamp for each candidate vehicle.
    coverage = (
        df.groupby("Vehicle_ID", sort=False)["Time_s"]
        .agg(first_time="min", last_time="max")
    )

    keep = pd.Series(True, index=coverage.index)

    # The vehicle must already exist when the requested window starts.
    if start_time is not None:
        keep &= coverage["first_time"] <= start_time + tolerance_s

    # The vehicle must still exist when the requested window ends.
    if end_time is not None:
        keep &= coverage["last_time"] >= end_time - tolerance_s

    valid_vehicle_ids = coverage.index[keep]
    out = df[df["Vehicle_ID"].isin(valid_vehicle_ids)].copy()

    # Only after selecting complete-window vehicles do we crop their samples
    # to the interval that will actually be replayed in the simulation.
    # if start_time is not None:
    #     out = out[out["Time_s"] >= start_time - tolerance_s]

    # if end_time is not None:
    #     out = out[out["Time_s"] <= end_time + tolerance_s]

    return out.copy()


def save_vehicle_numpy_files(df: pd.DataFrame, output_dir: Path) -> int:
    """
    Save one N x 4 NumPy array per vehicle.

    Column order in every file:
        0: time_s
        1: x_m
        2: y_m
        3: speed_kph
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove trajectory files from a previous run so a narrower time filter
    # cannot leave stale vehicles in the output directory. Other files are
    # left untouched.
    for old_file in output_dir.glob("vehicle_*.npy"):
        old_file.unlink()

    saved = 0

    for vehicle_id, vehicle in df.groupby("Vehicle_ID", sort=True):
        vehicle = vehicle.sort_values(["Time_s", "Frame_ID"]).copy()

        trajectory = np.column_stack(
            (
                pd.to_numeric(vehicle["Time_s"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(vehicle["Sim_X"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(vehicle["Sim_Y"], errors="coerce").to_numpy(dtype=float),
                (
                    pd.to_numeric(vehicle["v_Vel"], errors="coerce").to_numpy(dtype=float)
                    * 3.6
                ),
            )
        )

        # Remove malformed samples so np.interp can be used directly later.
        trajectory = trajectory[np.all(np.isfinite(trajectory), axis=1)]
        if trajectory.size == 0:
            continue

        # Guard against accidental duplicate timestamps for a vehicle. Keep the
        # first sample at each time because np.interp expects increasing x values.
        _, unique_indices = np.unique(trajectory[:, 0], return_index=True)
        trajectory = trajectory[np.sort(unique_indices)]

        if float(vehicle_id).is_integer():
            vehicle_label = str(int(vehicle_id))
        else:
            vehicle_label = str(vehicle_id).replace(".", "_")

        np.save(output_dir / f"vehicle_{vehicle_label}.npy", trajectory)
        saved += 1

    return saved

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter and align NGSIM US-101 trajectories to a three-lane "
            "weaving simulation and save one NumPy trajectory per vehicle."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input NGSIM trajectory CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for vehicle_*.npy files. Default: "
            "<input-stem>-numpy next to the input CSV."
        ),
    )

    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Optional PNG path for a transformed-trajectory preview.",
    )

    parser.add_argument(
        "--drop-samples-outside-4-8",
        action="store_true",
        help=(
            "After the vehicle-level filter, also remove samples whose "
            "Lane_ID is outside 4-8. By default complete trajectories are kept."
        ),
    )

    parser.add_argument(
        "--clip-to-road",
        action="store_true",
        help="Drop transformed samples outside simulation x=0...450 m.",
    )

    parser.add_argument(
        "--centerline-bin-width",
        type=float,
        default=10.0,
        help="Longitudinal bin width in meters used to estimate lane centers.",
    )

    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help=(
            "Require each exported vehicle to be present at this dataset-relative "
            "time, then crop its saved trajectory to start here. If omitted, no "
            "lower time-coverage requirement is applied."
        ),
    )

    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help=(
            "Require each exported vehicle to remain present until this "
            "dataset-relative time, then crop its saved trajectory to end here. "
            "If omitted, no upper time-coverage requirement is applied."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.input_csv.with_name(f"{args.input_csv.stem}-numpy")

    print(f"Reading: {args.input_csv}")
    raw = pd.read_csv(args.input_csv)

    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(
            "The CSV does not contain all required NGSIM columns. "
            f"Missing: {sorted(missing)}"
        )

    original_vehicle_count = raw["Vehicle_ID"].nunique()

    # Convert the complete dataset first because it is also used to estimate
    # the road/lane geometry. v_Vel becomes m/s here.
    full_si = convert_ngsim_to_si(raw)

    alignment = build_alignment(
        full_si,
        through_lane_1=4,
        through_lane_2=5,
        weaving_lane=6,
        sim_lane_width=DEFAULT_SIM_LANE_WIDTH,
        sim_weave_start_x=DEFAULT_SIM_WEAVE_START_X,
        sim_weave_end_x=DEFAULT_SIM_WEAVE_END_X,
        centerline_bin_width_m=args.centerline_bin_width,
    )

    filtered = filter_relevant_vehicles(
        full_si,
        valid_lanes=DEFAULT_VALID_LANES,
        drop_samples_outside_valid_lanes=args.drop_samples_outside_4_8,
    )

    aligned = transform_positions(filtered, alignment)
    aligned = add_dataset_time(aligned, raw)

    # Keep only vehicles whose trajectories cover the COMPLETE requested
    # time interval, then crop those retained trajectories to that interval.
    # Time_s remains relative to the beginning of the complete recording.
    vehicles_before_time_filter = aligned["Vehicle_ID"].nunique()
    aligned = filter_vehicles_covering_time_window(
        aligned,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    vehicles_after_time_filter = aligned["Vehicle_ID"].nunique()

    if args.clip_to_road:
        aligned = aligned[
            aligned["Sim_X"].between(
                DEFAULT_SIM_ROAD_START_X,
                DEFAULT_SIM_ROAD_END_X,
            )
        ].copy()

    aligned = aligned.sort_values(
        ["Vehicle_ID", "Time_s", "Frame_ID"]
    ).reset_index(drop=True)

    retained_vehicle_count = aligned["Vehicle_ID"].nunique()
    saved_vehicle_count = save_vehicle_numpy_files(aligned, output_dir)

    print()
    print("Alignment summary")
    print("-----------------")
    print(f"Original vehicles:          {original_vehicle_count}")
    print(f"Retained vehicles:          {retained_vehicle_count}")
    print(f"Saved vehicle files:        {saved_vehicle_count}")
    print(f"Output trajectory samples:  {len(aligned):,}")
    if args.start_time is not None or args.end_time is not None:
        start_label = "start" if args.start_time is None else f"{args.start_time:.3f} s"
        end_label = "end" if args.end_time is None else f"{args.end_time:.3f} s"
        print(f"Requested time window:      {start_label} ... {end_label}")
        print(
            f"Vehicles covering window:    {vehicles_after_time_filter} / "
            f"{vehicles_before_time_filter}"
        )
    print(
        "NGSIM lane-6 interval:     "
        f"{alignment['ngsim_weave_start_y']:.3f} ... "
        f"{alignment['ngsim_weave_end_y']:.3f} m"
    )
    print(
        "Mapped weaving interval:   "
        f"{DEFAULT_SIM_WEAVE_START_X:.1f} ... "
        f"{DEFAULT_SIM_WEAVE_END_X:.1f} m"
    )
    print(
        "Longitudinal scale:        "
        f"{alignment['longitudinal_scale']:.6f}"
    )
    if not aligned.empty:
        print(
            "Aligned x range:           "
            f"{aligned['Sim_X'].min():.3f} ... "
            f"{aligned['Sim_X'].max():.3f} m"
        )
        print(
            "Dataset-relative time:     "
            f"{aligned['Time_s'].min():.3f} ... "
            f"{aligned['Time_s'].max():.3f} s"
        )

    print(f"Saved NumPy trajectories:   {output_dir}")
    print("Array columns:              [time_s, x_m, y_m, speed_kph]")

    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        create_preview(
            aligned,
            args.preview,
            sim_road_start_x=DEFAULT_SIM_ROAD_START_X,
            sim_road_end_x=DEFAULT_SIM_ROAD_END_X,
            sim_weave_start_x=DEFAULT_SIM_WEAVE_START_X,
            sim_weave_end_x=DEFAULT_SIM_WEAVE_END_X,
            lane_width=DEFAULT_SIM_LANE_WIDTH,
        )
        print(f"Saved preview:              {args.preview}")


if __name__ == "__main__":
    main()
