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
import training.v6.models.train_stable_direction_brain_v692 as v692
import training.v6.audits.audit_raw_m5_direction_learnability_v693a as rawprobe
import training.v6.audits.audit_actionable_direction_learnability_v693b as actionable


OUT = Path(
    "training/artifacts/v6/"
    "actionable_direction_head_selection_v693c"
)

DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 512
DEFAULT_MAX_TRAIN = 120000
SEED = 20260824


def candidate_side_probs(pred):
    h30 = pred["horizon"][:, 0]
    h60 = pred["horizon"][:, 1]
    h120 = pred["horizon"][:, 2]
    all9 = pred["all9"]

    return {
        "H30": h30,
        "H60": h60,
        "H120": h120,
        "ALL9": all9,
        "ENS_FAST": (
            0.50 * all9
            + 0.35 * h30
            + 0.15 * h60
        ).astype(np.float32),
        "ENS_BALANCED": (
            0.40 * all9
            + 0.30 * h30
            + 0.20 * h60
            + 0.10 * h120
        ).astype(np.float32),
        "ENS_LONG": (
            0.40 * all9
            + 0.20 * h30
            + 0.20 * h60
            + 0.20 * h120
        ).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    rawprobe.seed_all(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9.3C ACTIONABLE DIRECTION HEAD-SELECTION AUDIT")
    print("=" * 132)
    print("Question: H30/ALL9 look more learnable on selected rows than H120.")
    print("Compare all learned heads and fixed ensembles inside frozen V6.8.")
    print("2016-2022 train -> 2023 dev only. 2024/2025/2026 remain closed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    arrays, split, groups, names, feature_mean, feature_std = brain.load_data()
    execution = v671.load_execution_targets()
    daily = v672.load_daily_targets()
    stable = v692.build_stable_targets(execution)
    raw, raw_ts = rawprobe.load_raw_channels()

    train_rows_all = v680.filter_rows(split["train"], arrays, execution)
    val = split["val"]
    years = execution["year"][arrays["source"]]
    rows23 = v680.filter_rows(val[years[val] == 2023], arrays, execution)

    train_rows_all = train_rows_all[
        arrays["source"][train_rows_all] >= rawprobe.SEQ - 1
    ]
    rows23 = rows23[arrays["source"][rows23] >= rawprobe.SEQ - 1]

    mean_t = torch.from_numpy(np.asarray(feature_mean, np.float32)).view(1, 1, -1).to(device)
    std_t = torch.from_numpy(np.asarray(feature_std, np.float32)).view(1, 1, -1).to(device)

    base_model, base_checkpoint = v691.load_v680(len(names), device)
    policy = base_checkpoint["policy"]
    print("Frozen V6.8 epoch:", base_checkpoint["epoch"])
    print("Frozen V6.8 policy:", policy)

    print()
    print("Computing frozen V6.8 TRAIN readiness ...")
    base_train = v680.predict(
        base_model,
        train_rows_all,
        arrays,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )
    train_masks = actionable.policy_masks(
        train_rows_all,
        base_train,
        arrays,
        daily,
        policy,
    )

    train_rows = train_rows_all[train_masks["candidate"]]
    h120_valid = stable["consensus_valid"][arrays["source"][train_rows], 2]
    train_rows = train_rows[h120_valid]

    rng = np.random.default_rng(args.seed)
    if args.max_train > 0 and len(train_rows) > args.max_train:
        train_rows = np.sort(
            rng.choice(train_rows, size=args.max_train, replace=False)
        )

    print("All train rows:", f"{len(train_rows_all):,}")
    print("Actionable train candidates:", f"{train_masks['candidate'].sum():,}")
    print("Actual first-per-day train selections:", f"{train_masks['selected'].sum():,}")
    print("Usable H120 training rows:", f"{len(train_rows):,}")

    print()
    print("Computing frozen V6.8 2023 readiness ...")
    base23 = v680.predict(
        base_model,
        rows23,
        arrays,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )
    masks23 = actionable.policy_masks(rows23, base23, arrays, daily, policy)

    print("2023 rows:", f"{len(rows23):,}")
    print("2023 actionable candidates:", f"{masks23['candidate'].sum():,}")
    print("2023 actual selected rows:", f"{masks23['selected'].sum():,}")
    print("2024: NOT EVALUATED")
    print("2025: LOCKED")
    print("2026: LOCKED")

    mean, std = rawprobe.fit_norm(raw, arrays["source"][train_rows])

    model = rawprobe.RawM5DirectionProbe(len(rawprobe.CHANNEL_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rawprobe.LR,
        weight_decay=rawprobe.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    history = []
    best = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        loss = rawprobe.train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
            raw,
            stable,
            mean,
            std,
            device,
            args.batch,
            rng,
            scaler,
        )

        pred23 = rawprobe.predict(
            model,
            rows23,
            arrays,
            raw,
            mean,
            std,
            device,
            args.batch,
        )

        print()
        print(f"EPOCH {epoch}/{args.epochs} HEAD COMPARISON")
        print("-" * 132)

        epoch_rows = []
        for label, side_prob in candidate_side_probs(pred23).items():
            system = v691.system_eval(
                rows23,
                side_prob,
                base23,
                arrays,
                execution,
                daily,
                policy,
                f"V6.9.3C {label}",
            )

            row = {
                "epoch": epoch,
                "head": label,
                "loss": loss,
                "trades": system["trades"],
                "coverage": system["coverage"],
                "win": system["win"],
                "mean": system["mean"],
                "pf": system["pf"],
            }
            epoch_rows.append(row)
            history.append(row)

            score = (
                5.0 * system["win"]
                + 0.20 * np.tanh(system["mean"] / 5.0)
                + 0.12 * np.log(max(system["pf"], 1e-6))
            )

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    **row,
                    "score": float(score),
                }

        table = pd.DataFrame(epoch_rows).sort_values(
            ["win", "mean", "pf"],
            ascending=False,
        )

        print()
        print(
            table[
                ["head", "trades", "coverage", "win", "mean", "pf"]
            ].to_string(
                index=False,
                formatters={
                    "coverage": lambda x: f"{x:.2%}",
                    "win": lambda x: f"{x:.2%}",
                    "mean": lambda x: f"{x:+.3f}",
                    "pf": lambda x: f"{x:.3f}",
                },
            )
        )

        pd.DataFrame(history).to_csv(
            OUT / "history_v693c.csv",
            index=False,
        )

        sec = time.time() - t0
        print(
            f"EPOCH {epoch}/{args.epochs} loss={loss:.5f} "
            f"BEST_SO_FAR={best['head']} "
            f"WR={best['win']:.2%} PF={best['pf']:.3f} sec={sec:.1f}"
        )

    if best["win"] >= 0.56:
        verdict = "USABLE_ACTIONABLE_HEAD_FOUND"
    elif best["win"] >= 0.53:
        verdict = "WEAK_ACTIONABLE_HEAD_FOUND"
    else:
        verdict = "NO_USABLE_ACTIONABLE_HEAD"

    summary = {
        "version": "v6.9.3c",
        "best": best,
        "verdict": verdict,
        "2024_evaluated": False,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v693c.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 132)
    print("BEST 2023 HEAD / ENSEMBLE")
    print(json.dumps(best, indent=2))
    print("VERDICT:", verdict)

    if verdict == "USABLE_ACTIONABLE_HEAD_FOUND":
        print("GO: raw-M5 actionable signal is useful when the right head is injected into V6.8.")
        print("Next: freeze the 2023-selected head/ensemble and build a larger production run before opening 2024.")
    elif verdict == "WEAK_ACTIONABLE_HEAD_FOUND":
        print("PARTIAL: keep this signal as an auxiliary and add M1/microstructure next.")
    else:
        print("STOP: none of the learned heads materially improves V6.8.")
        print("Next: inspect/rescue M1 micro-path or add genuinely new exogenous context.")

    print("2024 was not opened by this audit.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
