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
import training.v6.audits.audit_actionable_direction_head_selection_v693c as headselect


M1_DIR = Path(
    "training/vendor/dukascopy_xau_m1/xauusd/bid/m1"
)
ALIGN_TARGET = Path(
    "training/v6/data_lake/large_move_v60/large_move_targets_v60.parquet"
)
OUT = Path(
    "training/artifacts/v6/m1_direction_learnability_v693e"
)

GRID_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
GRID_END = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
HISTORICAL_INDEX_ORIGIN = pd.Timestamp("2016-01-03 20:26:00", tz="UTC")
MINUTE_NS = 60_000_000_000
SEQ = 240
DEFAULT_EPOCHS = 8
DEFAULT_BATCH = 256
DEFAULT_MAX_TRAIN = 60000
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 20260824

CHANNEL_NAMES = (
    "close_return_bps",
    "open_gap_bps",
    "body_bps",
    "range_bps",
    "upper_wick_bps",
    "lower_wick_bps",
    "close_position",
    "time_sin",
    "time_cos",
    "gap_flag",
    "reopen_flag",
)
PRICE_CHANNELS = 7


def bps_ratio(a, b):
    return ((a / b) - 1.0) * 10000.0


def load_dense_m1():
    files = sorted(M1_DIR.glob("xauusd_bid_m1_*.csv"))
    files = [
        p for p in files
        if "_2020_" in p.name
        or "_2021_" in p.name
        or "_2022_" in p.name
        or "_2023_" in p.name
    ]

    if len(files) != 48:
        raise RuntimeError(
            f"Expected 48 monthly M1 files for 2020-2023, found {len(files)}"
        )

    frames = []
    for p in files:
        frames.append(
            pd.read_csv(
                p,
                usecols=["timestamp", "open", "high", "low", "close"],
            )
        )

    df = pd.concat(frames, ignore_index=True)
    ts_ms = pd.to_numeric(df["timestamp"], errors="raise").to_numpy(np.int64)
    ts_ns = ts_ms * 1_000_000

    order = np.argsort(ts_ns, kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        df = df.iloc[order].reset_index(drop=True)
        ts_ns = ts_ns[order]

    if np.any(np.diff(ts_ns) <= 0):
        raise RuntimeError("Recovered M1 timestamps are duplicated or unsorted")
    if np.any(ts_ns % MINUTE_NS != 0):
        raise RuntimeError("Recovered M1 contains non-minute timestamps")

    start_ns = int(GRID_START.value)
    end_ns = int(GRID_END.value)
    keep = (ts_ns >= start_ns) & (ts_ns < end_ns)
    df = df.loc[keep].reset_index(drop=True)
    ts_ns = ts_ns[keep]

    grid_n = (end_ns - start_ns) // MINUTE_NS
    grid_idx = ((ts_ns - start_ns) // MINUTE_NS).astype(np.int64)

    if np.any(grid_idx < 0) or np.any(grid_idx >= grid_n):
        raise RuntimeError("M1 timestamp fell outside dense grid")
    if len(np.unique(grid_idx)) != len(grid_idx):
        raise RuntimeError("More than one M1 candle mapped to a dense minute")

    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)

    if (
        np.any(o <= 0)
        or np.any(h <= 0)
        or np.any(l <= 0)
        or np.any(c <= 0)
        or np.any(h < np.maximum.reduce([o, l, c]))
        or np.any(l > np.minimum.reduce([o, h, c]))
    ):
        raise RuntimeError("Invalid OHLC in recovered M1")

    dense = np.zeros((grid_n, len(CHANNEL_NAMES)), dtype=np.float32)
    valid = np.zeros(grid_n, dtype=bool)
    valid[grid_idx] = True

    contiguous = np.zeros(len(df), dtype=bool)
    contiguous[1:] = np.diff(grid_idx) == 1

    close_ret = np.zeros(len(df), np.float32)
    open_gap = np.zeros(len(df), np.float32)
    j = np.flatnonzero(contiguous)
    close_ret[j] = bps_ratio(c[j], c[j - 1]).astype(np.float32)
    open_gap[j] = bps_ratio(o[j], c[j - 1]).astype(np.float32)

    body = bps_ratio(c, o).astype(np.float32)
    bar_range = bps_ratio(h, l).astype(np.float32)
    top = np.maximum(o, c)
    bottom = np.minimum(o, c)
    upper = bps_ratio(h, top).astype(np.float32)
    lower = bps_ratio(bottom, l).astype(np.float32)
    denom = np.maximum(h - l, 1e-12)
    close_pos = ((c - l) / denom - 0.5).astype(np.float32)

    dense[grid_idx, 0] = close_ret
    dense[grid_idx, 1] = open_gap
    dense[grid_idx, 2] = body
    dense[grid_idx, 3] = bar_range
    dense[grid_idx, 4] = upper
    dense[grid_idx, 5] = lower
    dense[grid_idx, 6] = close_pos

    minute_of_day = (np.arange(grid_n, dtype=np.int64) % 1440).astype(np.float32)
    angle = 2.0 * np.pi * minute_of_day / 1440.0
    dense[:, 7] = np.sin(angle).astype(np.float32)
    dense[:, 8] = np.cos(angle).astype(np.float32)
    dense[:, 9] = (~valid).astype(np.float32)

    prev_valid = np.zeros(grid_n, dtype=bool)
    prev_valid[1:] = valid[:-1]
    dense[:, 10] = (valid & ~prev_valid).astype(np.float32)

    dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)

    print("Recovered M1 files:", len(files))
    print("Recovered real candles:", f"{len(df):,}")
    print("Dense calendar minutes:", f"{grid_n:,}")
    print("Dense missing minutes:", f"{(~valid).sum():,}")
    print("Real-minute coverage:", f"{valid.mean():.2%}")

    return dense, valid, start_ns, end_ns


def load_available_ns(n_source):
    if not ALIGN_TARGET.exists():
        raise RuntimeError(f"Missing historical alignment target: {ALIGN_TARGET}")

    t = pd.read_parquet(
        ALIGN_TARGET,
        columns=["source_row", "available_at", "m1_end_index"],
    )
    source = t["source_row"].to_numpy(np.int64)
    available = pd.to_datetime(t["available_at"], utc=True).astype("int64").to_numpy(np.int64)
    old_idx = t["m1_end_index"].to_numpy(np.int64)

    origin_ns = int(HISTORICAL_INDEX_ORIGIN.value)
    delta = available - origin_ns
    minute_aligned = (delta % MINUTE_NS) == 0
    reconstructed_idx = delta // MINUTE_NS
    exact = minute_aligned & (reconstructed_idx == old_idx)
    share = float(exact.mean())

    print("Historical dense-index origin:", HISTORICAL_INDEX_ORIGIN)
    print("Historical dense-index exact share:", f"{share:.6%}")
    if share != 1.0:
        raise RuntimeError(
            "Historical m1_end_index is not exactly reproducible as dense calendar minutes"
        )

    out = np.full(n_source, -1, dtype=np.int64)
    ok = (source >= 0) & (source < n_source)
    out[source[ok]] = available[ok]
    return out


def fit_norm(dense, valid, start_ns):
    train_end_ns = int(pd.Timestamp("2023-01-01 00:00:00", tz="UTC").value)
    n_train_minutes = (train_end_ns - start_ns) // MINUTE_NS
    mask = valid.copy()
    mask[int(n_train_minutes):] = False
    z = dense[mask, :PRICE_CHANNELS].astype(np.float64)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std[std < 1e-4] = 1.0

    full_mean = np.zeros(len(CHANNEL_NAMES), dtype=np.float32)
    full_std = np.ones(len(CHANNEL_NAMES), dtype=np.float32)
    full_mean[:PRICE_CHANNELS] = mean.astype(np.float32)
    full_std[:PRICE_CHANNELS] = std.astype(np.float32)
    return full_mean, full_std


def usable_rows(rows, arrays, available_ns, start_ns, end_ns):
    source = arrays["source"][rows]
    a = available_ns[source]
    ok = (
        (a >= start_ns + SEQ * MINUTE_NS)
        & (a <= end_ns)
        & (a % MINUTE_NS == 0)
    )
    return rows[ok]


def make_batch(rows, arrays, available_ns, dense, mean, std, start_ns, device):
    source = arrays["source"][rows]
    available = available_ns[source]
    cutoff = ((available - start_ns) // MINUTE_NS).astype(np.int64)
    offsets = np.arange(SEQ, 0, -1, dtype=np.int64)
    idx = cutoff[:, None] - offsets[None, :]

    if np.any(idx < 0) or np.any(idx >= len(dense)):
        raise RuntimeError("M1 sequence index outside dense grid")

    x = np.asarray(dense[idx], dtype=np.float32)
    gap = x[:, :, 9] > 0.5
    price = (x[:, :, :PRICE_CHANNELS] - mean[None, None, :PRICE_CHANNELS]) / std[
        None, None, :PRICE_CHANNELS
    ]
    price[gap] = 0.0
    x[:, :, :PRICE_CHANNELS] = price
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True),
        source,
    )


def train_epoch(
    model,
    optimizer,
    rows,
    arrays,
    available_ns,
    dense,
    stable,
    mean,
    std,
    start_ns,
    device,
    batch,
    rng,
    scaler,
):
    model.train()
    order = rng.permutation(rows)
    total = 0.0
    count = 0

    for start in range(0, len(order), batch):
        br = order[start:start + batch]
        x, source = make_batch(
            br,
            arrays,
            available_ns,
            dense,
            mean,
            std,
            start_ns,
            device,
        )
        target = rawprobe.target_tensors(source, stable, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(x)
            h30 = rawprobe.weighted_bce(
                out["horizon"][:, 0],
                target["side"][:, 0],
                target["valid"][:, 0],
                target["strength"][:, 0],
            )
            h60 = rawprobe.weighted_bce(
                out["horizon"][:, 1],
                target["side"][:, 1],
                target["valid"][:, 1],
                target["strength"][:, 1],
            )
            h120 = rawprobe.weighted_bce(
                out["horizon"][:, 2],
                target["side"][:, 2],
                target["valid"][:, 2],
                target["strength"][:, 2],
            )
            all9 = rawprobe.weighted_bce(
                out["all9"],
                target["all9"],
                target["all9_valid"],
                target["all9_strength"],
            )
            loss = h30 + h60 + h120 + 1.25 * all9

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach())
        count += 1

    return total / max(count, 1)


@torch.no_grad()
def predict(
    model,
    rows,
    arrays,
    available_ns,
    dense,
    mean,
    std,
    start_ns,
    device,
    batch,
):
    model.eval()
    hp = []
    ap = []
    for start in range(0, len(rows), batch):
        br = rows[start:start + batch]
        x, _ = make_batch(
            br,
            arrays,
            available_ns,
            dense,
            mean,
            std,
            start_ns,
            device,
        )
        out = model(x)
        hp.append(torch.sigmoid(out["horizon"]).cpu().numpy())
        ap.append(torch.sigmoid(out["all9"]).cpu().numpy())

    return {
        "horizon": np.concatenate(hp).astype(np.float32),
        "all9": np.concatenate(ap).astype(np.float32),
    }


def max_auc(metrics):
    vals = [
        metrics.get("H30_auc", np.nan),
        metrics.get("H60_auc", np.nan),
        metrics.get("H120_auc", np.nan),
        metrics.get("ALL9_auc", np.nan),
    ]
    vals = [float(x) for x in vals if np.isfinite(x)]
    return max(vals) if vals else float("nan")


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

    print("TEN V6.9.3E M1 DIRECTION LEARNABILITY AUDIT")
    print("=" * 132)
    print("Recovered Dukascopy bid-M1 on a dense calendar-minute grid.")
    print("Train: frozen-V6.8 actionable candidates from 2020-2022. Dev: 2023 only.")
    print("The M1 sequence ends strictly BEFORE available_at; no future minute is visible.")
    print("2024/2025/2026 remain closed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    arrays, split, groups, names, feature_mean, feature_std = brain.load_data()
    execution = v671.load_execution_targets()
    daily = v672.load_daily_targets()
    stable = v692.build_stable_targets(execution)

    dense, dense_valid, grid_start_ns, grid_end_ns = load_dense_m1()
    available_ns = load_available_ns(len(execution["year"]))
    mean, std = fit_norm(dense, dense_valid, grid_start_ns)

    train_all = v680.filter_rows(split["train"], arrays, execution)
    val = split["val"]
    years = execution["year"][arrays["source"]]

    train_year = years[train_all]
    train_all = train_all[(train_year >= 2020) & (train_year <= 2022)]
    rows23 = v680.filter_rows(val[years[val] == 2023], arrays, execution)

    train_all = usable_rows(
        train_all, arrays, available_ns, grid_start_ns, grid_end_ns
    )
    rows23 = usable_rows(
        rows23, arrays, available_ns, grid_start_ns, grid_end_ns
    )

    mean_t = torch.from_numpy(np.asarray(feature_mean, np.float32)).view(1, 1, -1).to(device)
    std_t = torch.from_numpy(np.asarray(feature_std, np.float32)).view(1, 1, -1).to(device)

    base_model, base_checkpoint = v691.load_v680(len(names), device)
    policy = base_checkpoint["policy"]
    print("Frozen V6.8 epoch:", base_checkpoint["epoch"])
    print("Frozen V6.8 policy:", policy)

    print()
    print("Computing frozen V6.8 2020-2022 readiness ...")
    base_train = v680.predict(
        base_model,
        train_all,
        arrays,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )
    train_masks = actionable.policy_masks(
        train_all, base_train, arrays, daily, policy
    )

    train_rows = train_all[train_masks["candidate"]]
    src_train = arrays["source"][train_rows]
    target_ok = (
        stable["all9_valid"][src_train]
        & stable["consensus_valid"][src_train].any(axis=1)
    )
    train_rows = train_rows[target_ok]

    rng = np.random.default_rng(args.seed)
    if args.max_train > 0 and len(train_rows) > args.max_train:
        train_rows = np.sort(
            rng.choice(train_rows, size=args.max_train, replace=False)
        )

    print("2020-2022 V6.8 rows:", f"{len(train_all):,}")
    print("2020-2022 actionable candidates:", f"{train_masks['candidate'].sum():,}")
    print("2020-2022 first-per-day selections:", f"{train_masks['selected'].sum():,}")
    print("M1 training rows:", f"{len(train_rows):,}")

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
    print("2023 first-per-day selections:", f"{masks23['selected'].sum():,}")
    print("M1 channels:", len(CHANNEL_NAMES))
    print("M1 sequence minutes:", SEQ)
    print("2024: NOT EVALUATED")
    print("2025: LOCKED")
    print("2026: LOCKED")

    print()
    print("2023 FROZEN V6.8 BASELINE")
    print("-" * 132)
    baseline = v691.system_eval(
        rows23,
        base23["side"],
        base23,
        arrays,
        execution,
        daily,
        policy,
        "V6.8 baseline",
    )

    model = rawprobe.RawM5DirectionProbe(len(CHANNEL_NAMES)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("M1 probe parameters:", f"{params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    history = []
    best = None
    best_candidate_auc = -np.inf
    best_selected_auc = -np.inf

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
            available_ns,
            dense,
            stable,
            mean,
            std,
            grid_start_ns,
            device,
            args.batch,
            rng,
            scaler,
        )

        pred23 = predict(
            model,
            rows23,
            arrays,
            available_ns,
            dense,
            mean,
            std,
            grid_start_ns,
            device,
            args.batch,
        )

        print()
        print(f"EPOCH {epoch}/{args.epochs} M1 LEARNABILITY")
        print("=" * 132)
        candidate_metrics = actionable.subset_metrics(
            "2023 ACTIONABLE CANDIDATES",
            rows23,
            masks23["candidate"],
            pred23,
            arrays,
            stable,
        )
        selected_metrics = actionable.subset_metrics(
            "2023 FROZEN V6.8 SELECTED ROWS",
            rows23,
            masks23["selected"],
            pred23,
            arrays,
            stable,
        )

        epoch_candidate_auc = max_auc(candidate_metrics)
        epoch_selected_auc = max_auc(selected_metrics)
        best_candidate_auc = max(best_candidate_auc, epoch_candidate_auc)
        best_selected_auc = max(best_selected_auc, epoch_selected_auc)

        print()
        print("2023 FROZEN V6.8 HEAD INJECTION")
        print("-" * 132)

        epoch_systems = []
        for label, side_prob in headselect.candidate_side_probs(pred23).items():
            system = v691.system_eval(
                rows23,
                side_prob,
                base23,
                arrays,
                execution,
                daily,
                policy,
                f"V6.9.3E M1 {label}",
            )
            score = (
                5.0 * system["win"]
                + 0.20 * np.tanh(system["mean"] / 5.0)
                + 0.12 * np.log(max(system["pf"], 1e-6))
            )
            row = {
                "epoch": epoch,
                "head": label,
                "loss": float(loss),
                "candidate_best_auc": float(epoch_candidate_auc),
                "selected_best_auc": float(epoch_selected_auc),
                "trades": int(system["trades"]),
                "coverage": float(system["coverage"]),
                "win": float(system["win"]),
                "mean": float(system["mean"]),
                "pf": float(system["pf"]),
                "score": float(score),
            }
            epoch_systems.append(row)
            history.append(row)

            if best is None or score > best["score"]:
                best = dict(row)
                torch.save(
                    {
                        "version": "v6.9.3e",
                        "epoch": epoch,
                        "head": label,
                        "model": model.state_dict(),
                        "mean": mean,
                        "std": std,
                        "channels": CHANNEL_NAMES,
                        "seq": SEQ,
                        "grid_start": str(GRID_START),
                        "historical_index_origin": str(HISTORICAL_INDEX_ORIGIN),
                    },
                    OUT / "best_v693e.pt",
                )

        table = pd.DataFrame(epoch_systems).sort_values(
            ["win", "mean", "pf"], ascending=False
        )
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
            OUT / "history_v693e.csv", index=False
        )

        print(
            f"EPOCH {epoch}/{args.epochs} loss={loss:.5f} "
            f"CAND_AUC={epoch_candidate_auc:.4f} "
            f"SEL_AUC={epoch_selected_auc:.4f} "
            f"BEST={best['head']} WR={best['win']:.2%} "
            f"PF={best['pf']:.3f} sec={time.time() - t0:.1f}"
        )

    if best["win"] >= 0.56 and best_candidate_auc >= 0.56:
        verdict = "M1_DIRECTION_SIGNAL_FOUND"
    elif best["win"] >= 0.53 or best_candidate_auc >= 0.56:
        verdict = "WEAK_M1_DIRECTION_SIGNAL"
    else:
        verdict = "NO_CLEAR_M1_DIRECTION_SIGNAL"

    summary = {
        "version": "v6.9.3e",
        "train_years": "2020-2022",
        "dev_year": 2023,
        "m1_real_candles": int(dense_valid.sum()),
        "m1_dense_minutes": int(len(dense)),
        "sequence_minutes": SEQ,
        "train_rows": int(len(train_rows)),
        "baseline_2023": {
            "trades": int(baseline["trades"]),
            "coverage": float(baseline["coverage"]),
            "win": float(baseline["win"]),
            "mean": float(baseline["mean"]),
            "pf": float(baseline["pf"]),
        },
        "best": best,
        "best_candidate_auc_2023": float(best_candidate_auc),
        "best_selected_auc_2023": float(best_selected_auc),
        "verdict": verdict,
        "2024_evaluated": False,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v693e.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 132)
    print("BEST 2023 M1 HEAD / ENSEMBLE")
    print(json.dumps(best, indent=2))
    print("Best actionable-candidate AUC:", f"{best_candidate_auc:.4f}")
    print("Best frozen-selected AUC:", f"{best_selected_auc:.4f}")
    print("VERDICT:", verdict)

    if verdict == "M1_DIRECTION_SIGNAL_FOUND":
        print("GO: M1 contains materially useful direction information beyond the M5 probes.")
        print("Next: freeze this research design, extend M1 back to 2016, then train the production direction branch.")
    elif verdict == "WEAK_M1_DIRECTION_SIGNAL":
        print("PARTIAL: M1 adds signal, but not enough yet for a production direction replacement.")
        print("Next: inspect context length and bid/ask microstructure before scaling years/capacity.")
    else:
        print("STOP: do not download 2016-2019 or build a larger M1 network yet.")
        print("Next: direction needs genuinely new information such as bid/ask microstructure or exogenous context.")

    print("2024 was not opened by this audit.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
