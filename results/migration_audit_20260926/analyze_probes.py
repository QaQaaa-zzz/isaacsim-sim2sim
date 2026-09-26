"""Reproducible scientific figures from bounded, frozen PhysX probe records."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="label=directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    colors = ["#1767a6", "#c57b17", "#ae4975", "#597541", "#616161"]
    summary, traces = [], {}
    for index, item in enumerate(args.case):
        label, directory = item.split("=", 1)
        directory = Path(directory)
        record = np.load(directory / "traces.npz")
        target = record["target"]
        assert np.isfinite(target).all() and np.all(np.diff(target[:, 0]) > 0)
        names = record["joint_names"].tolist()
        rear = names.index("rearwheel_joint")
        rotation = Rotation.from_quat(target[:, [5, 6, 7, 4]])
        angles = rotation.as_euler("xyz")
        gyro = rotation.inv().apply(target[:, 21:24])
        reward = record["rewards"]
        assert len(reward) == len(target)
        cumulative = np.cumsum(reward, dtype=np.float64)
        info = json.loads((directory / "steps.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        time = target[:, 0]
        style = "--" if "repeat" in label else "-"
        color = colors[index % len(colors)]
        pairs = [(target[:, 1], target[:, 3]), (time, angles[:, 0]),
                 (time, target[:, 13 + rear]), (time, gyro[:, 1]),
                 (np.arange(1, len(reward) + 1), reward),
                 (np.arange(1, len(reward) + 1), cumulative)]
        for axis, (x, y) in zip(axes.ravel(), pairs):
            axis.plot(x, y, style, color=color, linewidth=1.6, label=label)
            axis.plot(x[-1], y[-1], "x", color=color, markersize=7)
        row = dict(case=label, source=str(directory.resolve()), observed_steps=len(time),
                   observed_s=float(time[-1]), end_code=int(info[-1]["end_code"]),
                   endpoint_reason=status["endpoint_reason"],
                   max_x_m=float(target[:, 1].max()), max_z_m=float(target[:, 3].max()),
                   final_x_m=float(target[-1, 1]), final_z_m=float(target[-1, 3]),
                   max_abs_roll_rad=float(np.abs(angles[:, 0]).max()),
                   return_=float(cumulative[-1]),
                   physics_substeps=int(status["physics_substeps"]))
        summary.append(row)
        traces[label] = target
        columns = np.column_stack([time, target[:, 1:4], angles, target[:, 18:21],
                                   target[:, 21:24], gyro, target[:, 13 + rear],
                                   target[:, 28], reward, cumulative, record["done"]])
        np.savetxt(args.output / f"{label}.csv", columns, delimiter=",", comments="",
                   header="time_s,x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad,vx_mps,vy_mps,vz_mps,"
                   "omega_world_x_radps,omega_world_y_radps,omega_world_z_radps,"
                   "omega_body_x_radps,omega_body_y_radps,omega_body_z_radps,"
                   "rear_velocity_radps,rear_target_radps,reward,cumulative_reward,done")
    definitions = [("Root trajectory", "x (m)", "z (m)"),
                   ("Body roll", "Time (s)", "Roll (rad)"),
                   ("Rear joint velocity", "Time (s)", "Angular velocity (rad/s)"),
                   ("Body-frame pitch gyro", "Time (s)", "Angular velocity (rad/s)"),
                   ("Reward at each actual control step", "Control step", "Reward"),
                   ("Cumulative reward including failure", "Control step", "Cumulative reward")]
    for axis, (title, xlabel, ylabel) in zip(axes.ravel(), definitions):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("Frozen MJX Actor transition_4988928 | PhysX 1 ms / policy 20 ms\n"
                   "One ground initial state per variant; x marks actual endpoint; 2 s diagnostic cap", fontsize=13)
    figure.savefig(args.output / "drive_comparison.png", dpi=180)
    figure.savefig(args.output / "drive_comparison.pdf")
    plt.close(figure)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if "calibrated" in traces and "calibrated_repeat" in traces:
        one, two = traces["calibrated"], traces["calibrated_repeat"]
        repeat = dict(same_shape=one.shape == two.shape, identical=np.array_equal(one, two),
                      max_abs_difference=float(np.abs(one - two).max()) if one.shape == two.shape else None)
        (args.output / "repeatability.json").write_text(json.dumps(repeat, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
