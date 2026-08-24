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


OUT = Path(
    "training/artifacts/v6/"
    "actionable_direction_learnability_v693b"
)

DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 512
DEFAULT_MAX_TRAIN = 120000
SEED = 20260824


def policy_masks(rows, pred, arrays, daily, policy):
    source = arrays["source"][rows]

    when_score = (
        0.10 * pred["when"][:, 0]
        + 0.30 * pred["when"][:, 1]
        + 0.35 * pred["when"][:, 2]
        + 0.25 * pred["when"][:, 3]
        + 0.20 * pred["rank"]
    )

    confidence = np.abs(pred["side"] - 0.5) * 2.0
    day = daily["day_ns"][source]

    counts = pd.Series(day).value_counts()
    eligible_days = set(
        int(x)
        for x in counts[counts >= v680.MIN_DAY_BARS].index
    )

    eligible = np.fromiter(
        (int(d) in eligible_days for d in day),
        count=len(day),
        dtype=bool,
    )

    candidate = (
        eligible
        & (when_score >= float(policy["threshold"]))
        & (confidence >= float(policy["confidence"]))
    )

    selected = np.zeros(len(rows), dtype=bool)
    used = set()
    for i in range(len(rows)):
        if not candidate[i]:
            continue
        d = int(day[i])
        if d in used:
            continue
        used.add(d)
        selected[i] = True

    return {
        "when_score": when_score.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "candidate": candidate,
        "selected": selected,
        "eligible_days": len(eligible_days),
    }


def subset_metrics(name, rows, pos_mask, pred, arrays, stable):
    source = arrays["source"][rows]
    pos = np.flatnonzero(pos_mask)

    print()
    print(name)
    print("-" * 132)

    result = {"n_rows": int(len(pos))}

    for hi, label in enumerate(("H30", "H60", "H120")):
        valid = stable["consensus_valid"][source[pos], hi]
        y = stable["consensus_side"][source[pos], hi][valid].astype(np.uint8)
        p = pred["horizon"][pos[valid], hi]
        auc = rawprobe.safe_auc(y, p)
        acc = float(((p >= 0.5) == y).mean()) if len(y) else np.nan
        result[f"{label}_auc"] = auc
        result[f"{label}_acc"] = acc
        print(
            f"{label:<6} N={len(y):>5,} ACC={acc:>7.2%} AUC={auc:>7.4f}"
        )

    valid = stable["all9_valid"][source[pos]]
    y = stable["all9_side"][source[pos]][valid].astype(np.uint8)
    p = pred["all9"][pos[valid]]
    auc = rawprobe.safe_auc(y, p)
    acc = float(((p >= 0.5) == y).mean()) if len(y) else np.nan
    result["ALL9_auc"] = auc
    result["ALL9_acc"] = acc
    print(
        f"ALL9   N={len(y):>5,} ACC={acc:>7.2%} AUC={auc:>7.4f}"
    )

    return result


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

    print("TEN V6.9.3B ACTIONABLE DIRECTION LEARNABILITY AUDIT")
    print("=" * 132)
    print("Train direction only where frozen V6.8 WHEN/confidence says the market is actionable.")
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
    train_masks = policy_masks(
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
    masks23 = policy_masks(rows23, base23, arrays, daily, policy)

    print("2023 rows:", f"{len(rows23):,}")
    print("2023 actionable candidates:", f"{masks23['candidate'].sum():,}")
    print("2023 actual selected rows:", f"{masks23['selected'].sum():,}")
    print("2024: NOT EVALUATED")
    print("2025: LOCKED")
    print("2026: LOCKED")

    if len(train_rows) < 1000:
        print("WARNING: actionable training set is small; results may be high variance.")

    mean, std = rawprobe.fit_norm(raw, arrays["source"][train_rows])

    model = rawprobe.RawM5DirectionProbe(len(rawprobe.CHANNEL_NAMES)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("Parameters:", f"{params:,}")

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
    best_wr = -np.inf
    best_selected_auc = -np.inf

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

        cand = subset_metrics(
            "2023 ACTIONABLE CANDIDATE DIRECTION",
            rows23,
            masks23["candidate"],
            pred23,
            arrays,
            stable,
        )
        sel = subset_metrics(
            "2023 ACTUAL SELECTED-ROW DIRECTION",
            rows23,
            masks23["selected"],
            pred23,
            arrays,
            stable,
        )

        print()
        print("2023 FROZEN V6.8 WITH ACTIONABLE-TRAINED RAW H120")
        print("-" * 132)
        system = v691.system_eval(
            rows23,
            pred23["horizon"][:, 2],
            base23,
            arrays,
            execution,
            daily,
            policy,
            "V6.9.3B ACTIONABLE RAW H120",
        )

        sec = time.time() - t0
        selected_auc = sel.get("H120_auc", np.nan)
        best_wr = max(best_wr, system["win"])
        if np.isfinite(selected_auc):
            best_selected_auc = max(best_selected_auc, selected_auc)

        row = {
            "epoch": epoch,
            "loss": loss,
            "candidate_h120_auc": cand.get("H120_auc"),
            "selected_h120_auc": selected_auc,
            "system_win": system["win"],
            "system_mean": system["mean"],
            "system_pf": system["pf"],
            "seconds": sec,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            OUT / "history_v693b.csv",
            index=False,
        )

        print(
            f"EPOCH {epoch}/{args.epochs} loss={loss:.5f} "
            f"SELECTED_H120_AUC={selected_auc:.4f} "
            f"WR={system['win']:.2%} PF={system['pf']:.3f} sec={sec:.1f}"
        )

    if best_wr >= 0.56 or best_selected_auc >= 0.60:
        verdict = "ACTIONABLE_DIRECTION_SIGNAL_FOUND"
    elif best_wr >= 0.53 or best_selected_auc >= 0.55:
        verdict = "WEAK_ACTIONABLE_DIRECTION_SIGNAL"
    else:
        verdict = "NO_ACTIONABLE_DIRECTION_SIGNAL"

    summary = {
        "version": "v6.9.3b",
        "train_actionable_candidates": int(train_masks["candidate"].sum()),
        "train_actual_selected": int(train_masks["selected"].sum()),
        "train_used": int(len(train_rows)),
        "dev_actionable_candidates": int(masks23["candidate"].sum()),
        "dev_actual_selected": int(masks23["selected"].sum()),
        "best_selected_h120_auc_2023": float(best_selected_auc),
        "best_system_win_2023": float(best_wr),
        "verdict": verdict,
        "2024_evaluated": False,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v693b.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 132)
    print("VERDICT:", verdict)
    print("Best selected-row H120 AUC:", f"{best_selected_auc:.4f}")
    print("Best 2023 downstream WR:", f"{best_wr:.2%}")

    if verdict == "ACTIONABLE_DIRECTION_SIGNAL_FOUND":
        print("GO: direction is learnable specifically where V6.8 trades.")
        print("Next: production direction model should train with readiness-conditioned sampling/weights.")
    elif verdict == "WEAK_ACTIONABLE_DIRECTION_SIGNAL":
        print("PARTIAL: actionable conditioning helps, but more context or M1 may still be needed.")
    else:
        print("STOP: conditioning on V6.8 actionable rows does not rescue direction.")
        print("Next: inspect/rescue M1 micro-path or add exogenous context.")

    print("2024 was not opened by this audit.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
