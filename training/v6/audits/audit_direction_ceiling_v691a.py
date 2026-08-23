from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.models.train_execution_precision_brain_v671 as v671
import training.v6.models.train_daily_opportunity_brain_v672 as v672
import training.v6.models.train_multiscale_execution_brain_v680 as v680
import training.v6.models.train_direction_recovery_v691 as v691


OUT = Path(
    "training/artifacts/v6/"
    "direction_ceiling_v691a"
)

EXPECTED_V680_2024_WIN = 0.4227642
EXPECTED_V680_2024_TRADES = 246


def oracle_side_for_rows(rows, arrays, execution):
    source = arrays["source"][rows]
    valid = execution["valid"][source]
    gross = execution["gross"][source].astype(np.float32)

    actual = np.where(
        valid,
        gross - 0.5,
        -np.inf,
    )

    best_long = np.max(actual[:, :9], axis=1)
    best_short = np.max(actual[:, 9:], axis=1)

    finite = np.isfinite(best_long) & np.isfinite(best_short)

    if not finite.all():
        raise RuntimeError(
            f"Non-finite oracle side rows: {(~finite).sum()}"
        )

    return (best_short > best_long).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        choices=(2023, 2024),
        default=2024,
    )
    args = parser.parse_args()

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9.1A EXACT DIRECTION CEILING AUDIT")
    print("=" * 130)
    print("Year:", args.year)
    print("No training. V6.8 is frozen. Only side is ablated.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    ) = brain.load_data()

    execution = v671.load_execution_targets()
    daily = v672.load_daily_targets()
    features = arrays["features"]

    years = execution["year"][arrays["source"]]
    val = split["val"]

    rows = v680.filter_rows(
        val[years[val] == args.year],
        arrays,
        execution,
    )

    print("Rows:", f"{len(rows):,}")
    print("2025: LOCKED / NOT EVALUATED")
    print("2026: LOCKED / NOT EVALUATED")

    mean_t = torch.from_numpy(
        np.asarray(mean, dtype=np.float32)
    ).view(1, 1, -1).to(device)

    std_np = np.asarray(std, dtype=np.float32).copy()
    std_np[std_np < 1e-6] = 1.0

    std_t = torch.from_numpy(std_np).view(
        1, 1, -1
    ).to(device)

    base_model, checkpoint = v691.load_v680(
        features.shape[1],
        device,
    )

    print("Frozen V6.8 epoch:", checkpoint["epoch"])
    print("Frozen V6.8 policy:", checkpoint["policy"])

    print()
    print("Computing frozen V6.8 outputs ...")

    base_pred = v680.predict(
        base_model,
        rows,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )

    oracle_side = oracle_side_for_rows(
        rows,
        arrays,
        execution,
    )

    print()
    print("EXACT SAME V6.8 SYSTEM — SIDE ABLATION")
    print("=" * 130)

    baseline = v691.system_eval(
        rows,
        base_pred["side"],
        base_pred,
        arrays,
        execution,
        daily,
        checkpoint["policy"],
        "V6.8 BASELINE",
    )

    oracle = v691.system_eval(
        rows,
        oracle_side,
        base_pred,
        arrays,
        execution,
        daily,
        checkpoint["policy"],
        "ORACLE SIDE CEILING",
    )

    recoverable = float(
        oracle["win"] - baseline["win"]
    )

    ceiling_over_80 = bool(
        oracle["win"] >= 0.80
        and oracle["mean"] > 0.0
        and oracle["pf"] >= 1.20
    )

    baseline_reproduced = None

    if args.year == 2024:
        baseline_reproduced = bool(
            baseline["trades"] == EXPECTED_V680_2024_TRADES
            and abs(
                baseline["win"] - EXPECTED_V680_2024_WIN
            ) < 1e-4
        )

        print()
        print("REPRODUCIBILITY CHECK")
        print("-" * 130)
        print(
            "Expected 2024 baseline:",
            f"N={EXPECTED_V680_2024_TRADES}",
            f"WIN={EXPECTED_V680_2024_WIN:.2%}",
        )
        print(
            "Reproduced:",
            baseline_reproduced,
        )

    print()
    print("DIRECTION RECOVERY TARGET")
    print("-" * 130)
    print(
        "Recoverable WR gap:",
        f"{recoverable:+.2%}"
    )
    print(
        "Oracle ceiling >= 80%:",
        ceiling_over_80,
    )

    if ceiling_over_80:
        print(
            "GO: direction-only recovery can theoretically reach the target "
            "under this frozen V6.8 policy."
        )
    else:
        print(
            "STOP: direction alone cannot reach 80% under this exact frozen "
            "V6.8 policy; do not spend GPU credits on the heavy run yet."
        )

    result = {
        "year": int(args.year),
        "baseline": baseline,
        "oracle_side_ceiling": oracle,
        "recoverable_win_gap": recoverable,
        "oracle_ceiling_over_80": ceiling_over_80,
        "baseline_reproduced": baseline_reproduced,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    pd.DataFrame(
        [baseline, oracle]
    ).to_csv(
        OUT / f"system_comparison_{args.year}.csv",
        index=False,
    )

    with open(
        OUT / f"summary_{args.year}.json",
        "w",
    ) as f:
        json.dump(result, f, indent=2)

    print()
    print(json.dumps(result, indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
