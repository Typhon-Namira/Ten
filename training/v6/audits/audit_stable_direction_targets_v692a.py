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
    "stable_direction_targets_v692a"
)

COST = 0.5


def build_pair_targets(execution):
    valid = execution["valid"]
    gross = execution["gross"].astype(np.float32)
    net = gross - COST

    n = len(gross)
    pair_side = np.zeros((n, 9), dtype=np.int8)
    pair_gap = np.full((n, 9), np.nan, dtype=np.float32)
    pair_valid = np.zeros((n, 9), dtype=bool)

    for j in range(9):
        sj = j + 9
        ok = (
            valid[:, j]
            & valid[:, sj]
            & np.isfinite(net[:, j])
            & np.isfinite(net[:, sj])
        )

        gap = net[:, sj] - net[:, j]
        ok &= np.abs(gap) > 1e-6

        pair_valid[:, j] = ok
        pair_gap[ok, j] = gap[ok]
        pair_side[ok, j] = (gap[ok] > 0.0).astype(np.int8)

    safe = np.where(valid, net, -np.inf)
    best_long = np.max(safe[:, :9], axis=1)
    best_short = np.max(safe[:, 9:], axis=1)

    master_valid = np.isfinite(best_long) & np.isfinite(best_short)
    master_side = np.zeros(n, dtype=np.int8)
    master_side[master_valid] = (
        best_short[master_valid] > best_long[master_valid]
    ).astype(np.int8)

    return {
        "pair_side": pair_side,
        "pair_gap": pair_gap,
        "pair_valid": pair_valid,
        "master_side": master_side,
        "master_valid": master_valid,
    }


def majority_target(pair_side, pair_valid, ids):
    ids = np.asarray(ids, dtype=np.int64)
    side = pair_side[:, ids]
    valid = pair_valid[:, ids]

    votes = (side * valid.astype(np.int8)).sum(axis=1)
    count = valid.sum(axis=1)

    usable = (count > 0) & (votes * 2 != count)
    out = np.zeros(len(pair_side), dtype=np.int8)
    out[usable] = (votes[usable] * 2 > count[usable]).astype(np.int8)

    return out, usable


def resolve_with_fallback(candidate, usable, fallback):
    out = np.asarray(fallback, dtype=np.float32).copy()
    out[usable] = candidate[usable].astype(np.float32)
    return out


def evaluate_candidate(
    name,
    candidate_side,
    usable,
    rows,
    base_pred,
    arrays,
    execution,
    daily,
    policy,
    master_side,
    master_valid,
):
    source = arrays["source"][rows]

    fallback = (base_pred["side"] >= 0.5).astype(np.int8)
    chosen = resolve_with_fallback(
        candidate_side[source],
        usable[source],
        fallback,
    )

    valid_compare = usable[source] & master_valid[source]
    agreement = (
        float(
            (
                candidate_side[source][valid_compare]
                == master_side[source][valid_compare]
            ).mean()
        )
        if valid_compare.any()
        else np.nan
    )

    result = v691.system_eval(
        rows,
        chosen,
        base_pred,
        arrays,
        execution,
        daily,
        policy,
        name,
    )

    result["target_availability"] = float(usable[source].mean())
    result["oracle_agreement"] = agreement
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        choices=(2023, 2024),
        default=2024,
    )
    args = parser.parse_args()
    year = int(args.year)

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9.2A STABLE DIRECTION TARGET AUDIT")
    print("=" * 136)
    print("Purpose: find the most useful stable direction target before another GPU run.")
    print("Year:", year)
    print("No training. Frozen V6.8 action/WHEN policy.")

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
        val[years[val] == year],
        arrays,
        execution,
    )

    mean_t = torch.from_numpy(
        np.asarray(mean, dtype=np.float32)
    ).view(1, 1, -1).to(device)

    std_np = np.asarray(std, dtype=np.float32).copy()
    std_np[std_np < 1e-6] = 1.0
    std_t = torch.from_numpy(std_np).view(1, 1, -1).to(device)

    base_model, checkpoint = v691.load_v680(
        features.shape[1],
        device,
    )

    base_pred = v680.predict(
        base_model,
        rows,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )

    targets = build_pair_targets(execution)
    source = arrays["source"][rows]

    print()
    print("REFERENCE")
    print("=" * 136)

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

    oracle_side = targets["master_side"][source].astype(np.float32)
    oracle = v691.system_eval(
        rows,
        oracle_side,
        base_pred,
        arrays,
        execution,
        daily,
        checkpoint["policy"],
        "MAX-OF-MAX ORACLE",
    )

    results = []

    print()
    print("FIXED PAIR TARGETS")
    print("=" * 136)

    for j in range(9):
        meta = brain.TASKS[j]
        name = (
            f"H{meta['horizon']}_"
            f"TP{meta['tp']}_SL{meta['sl']}"
        )

        result = evaluate_candidate(
            name,
            targets["pair_side"][:, j],
            targets["pair_valid"][:, j],
            rows,
            base_pred,
            arrays,
            execution,
            daily,
            checkpoint["policy"],
            targets["master_side"],
            targets["master_valid"],
        )
        results.append(result)

        print(
            f"  availability={result['target_availability']:.2%} "
            f"oracle_agreement={result['oracle_agreement']:.2%}"
        )

    print()
    print("HORIZON CONSENSUS TARGETS")
    print("=" * 136)

    for hi, horizon in enumerate((30, 60, 120)):
        ids = list(range(hi * 3, hi * 3 + 3))
        side, usable = majority_target(
            targets["pair_side"],
            targets["pair_valid"],
            ids,
        )

        result = evaluate_candidate(
            f"H{horizon}_MAJORITY",
            side,
            usable,
            rows,
            base_pred,
            arrays,
            execution,
            daily,
            checkpoint["policy"],
            targets["master_side"],
            targets["master_valid"],
        )
        results.append(result)

        print(
            f"  availability={result['target_availability']:.2%} "
            f"oracle_agreement={result['oracle_agreement']:.2%}"
        )

    print()
    print("ALL-PAIR CONSENSUS")
    print("=" * 136)

    side, usable = majority_target(
        targets["pair_side"],
        targets["pair_valid"],
        list(range(9)),
    )

    consensus = evaluate_candidate(
        "ALL_9_MAJORITY",
        side,
        usable,
        rows,
        base_pred,
        arrays,
        execution,
        daily,
        checkpoint["policy"],
        targets["master_side"],
        targets["master_valid"],
    )
    results.append(consensus)

    print(
        f"  availability={consensus['target_availability']:.2%} "
        f"oracle_agreement={consensus['oracle_agreement']:.2%}"
    )

    table = pd.DataFrame(results).sort_values(
        ["win", "mean", "pf"],
        ascending=False,
    )

    print()
    print("RANKING BY DOWNSTREAM V6.8 WIN RATE")
    print("=" * 136)
    print(
        table[
            [
                "label",
                "trades",
                "coverage",
                "win",
                "mean",
                "pf",
                "target_availability",
                "oracle_agreement",
            ]
        ].to_string(
            index=False,
            formatters={
                "coverage": lambda x: f"{x:.2%}",
                "win": lambda x: f"{x:.2%}",
                "mean": lambda x: f"{x:+.3f}",
                "pf": lambda x: f"{x:.3f}",
                "target_availability": lambda x: f"{x:.2%}",
                "oracle_agreement": lambda x: f"{x:.2%}",
            },
        )
    )

    table.to_csv(
        OUT / f"stable_target_ranking_{year}.csv",
        index=False,
    )

    summary = {
        "year": year,
        "baseline": baseline,
        "max_of_max_oracle": oracle,
        "best_stable_target": table.iloc[0].to_dict(),
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    with open(OUT / f"summary_v692a_{year}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("BEST STABLE TARGET:")
    print(json.dumps(summary["best_stable_target"], indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
