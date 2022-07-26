import json
import os
import pathlib
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd


def run_pytest_benchmarks(force: bool) -> None:
    td = os.environ["OPENSLIDE_TESTDATA_DIR"]
    assert td, f"must set OPENSLIDE_TESTDATA_DIR. Got: {td!r}"
    root = pathlib.Path(__file__).parent.parent
    assert root.joinpath("tiffslide", "tests").is_dir()

    fn_json = ".tiffslide_docs_benchmark.json"

    if not root.joinpath(fn_json).is_file() or force:
        subprocess.run(
            ["pytest", "-x", f"--benchmark-json={fn_json}", "--benchmark-only"],
            env=os.environ,
            cwd=root,
            check=True,
        )

    with root.joinpath(fn_json).open() as f:
        data = json.load(f)

    records = []
    for record in data["benchmarks"]:
        test_name, _, _ = record["name"].partition("[")
        test_name = test_name[5:]
        for rtime in record["stats"]["data"]:
            ps = record["params"].copy()
            file_type, modname = ps.pop("slide_with_tile_size")
            records.append(
                {
                    "test_name": test_name,
                    "file_type": file_type,
                    "modname": modname,
                    "label": "-".join(ps.values()),
                    "time": rtime,
                }
            )

    df = pd.DataFrame.from_records(records)
    for test_name, ptdf in df.groupby("test_name"):

        ft_ax_order = [
            "svs",
            "generic",
            "hamamatsu",
            "leica",
            "ventana",
        ]
        assert set(ptdf["file_type"].unique()) == set(ft_ax_order)
        fig, axes = plt.subplots(1, len(ft_ax_order), figsize=(10, 4))
        fig.suptitle(test_name, x=0.1, y=0.99)
        ft_ax_map = {ft: ax for ft, ax in zip(ft_ax_order, axes)}

        for file_type, tdf in ptdf.groupby("file_type"):
            ldf = tdf.groupby(["modname", "label"]).mean()
            ldf["time"] *= 1000
            ldf = ldf.reset_index()
            pdf = ldf.pivot(index="label", columns="modname", values="time")
            idx = sorted(pdf.index.tolist())
            pdf = pdf.loc[idx, :]
            pdf.plot.barh(title=f"{file_type}", ax=ft_ax_map[file_type], legend=False)

        axes[0].set_ylabel("access pattern")
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        for ax in axes:
            ax.set_xlabel("time (ms)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="upper right")

        fig.savefig(root.joinpath("docs", "images", f"benchmark_{test_name}.png"))


if __name__ == "__main__":
    force = sys.argv[-1] == "--force"
    run_pytest_benchmarks(force)
