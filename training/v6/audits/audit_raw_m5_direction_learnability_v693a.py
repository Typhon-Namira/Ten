from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.models.train_execution_precision_brain_v671 as v671
import training.v6.models.train_daily_opportunity_brain_v672 as v672
import training.v6.models.train_multiscale_execution_brain_v680 as v680
import training.v6.models.train_direction_recovery_v691 as v691
import training.v6.models.train_stable_direction_brain_v692 as v692


RAW_FILE = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "raw_m5_direction_learnability_v693a"
)

SEQ = 288  # 24 hours of M5 rows, with explicit gap markers.
STEP_NS = 300_000_000_000
DEFAULT_TRAIN_SAMPLE = 120000
DEFAULT_EPOCHS = 6
DEFAULT_BATCH = 512
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
    "spread_open_bps",
    "spread_close_bps",
    "hour_sin",
    "hour_cos",
    "gap_flag",
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))


def bps_ratio(a, b):
    return ((a / b) - 1.0) * 10000.0


def load_raw_channels():
    cols = [
        "timestamp",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]

    df = pd.read_parquet(RAW_FILE, columns=cols).reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    ts_ns = ts.astype("int64").to_numpy(np.int64)

    bid_open = df["bid_open"].to_numpy(np.float64)
    bid_high = df["bid_high"].to_numpy(np.float64)
    bid_low = df["bid_low"].to_numpy(np.float64)
    bid_close = df["bid_close"].to_numpy(np.float64)
    ask_open = df["ask_open"].to_numpy(np.float64)
    ask_high = df["ask_high"].to_numpy(np.float64)
    ask_low = df["ask_low"].to_numpy(np.float64)
    ask_close = df["ask_close"].to_numpy(np.float64)

    mid_open = 0.5 * (bid_open + ask_open)
    mid_high = 0.5 * (bid_high + ask_high)
    mid_low = 0.5 * (bid_low + ask_low)
    mid_close = 0.5 * (bid_close + ask_close)

    contiguous = np.zeros(len(df), dtype=bool)
    contiguous[1:] = np.diff(ts_ns) == STEP_NS

    close_ret = np.zeros(len(df), dtype=np.float32)
    open_gap = np.zeros(len(df), dtype=np.float32)
    idx = np.flatnonzero(contiguous)
    close_ret[idx] = bps_ratio(
        mid_close[idx], mid_close[idx - 1]
    ).astype(np.float32)
    open_gap[idx] = bps_ratio(
        mid_open[idx], mid_close[idx - 1]
    ).astype(np.float32)

    body = bps_ratio(mid_close, mid_open).astype(np.float32)
    bar_range = bps_ratio(mid_high, mid_low).astype(np.float32)

    top_body = np.maximum(mid_open, mid_close)
    bottom_body = np.minimum(mid_open, mid_close)
    upper_wick = bps_ratio(mid_high, top_body).astype(np.float32)
    lower_wick = bps_ratio(bottom_body, mid_low).astype(np.float32)

    denom = np.maximum(mid_high - mid_low, 1e-12)
    close_pos = ((mid_close - mid_low) / denom - 0.5).astype(np.float32)

    spread_open = (
        (ask_open - bid_open) / mid_open * 10000.0
    ).astype(np.float32)
    spread_close = (
        (ask_close - bid_close) / mid_close * 10000.0
    ).astype(np.float32)

    minute_of_day = (
        ts.dt.hour.to_numpy(np.float32) * 60.0
        + ts.dt.minute.to_numpy(np.float32)
    )
    angle = 2.0 * np.pi * minute_of_day / 1440.0
    hour_sin = np.sin(angle).astype(np.float32)
    hour_cos = np.cos(angle).astype(np.float32)

    gap_flag = (~contiguous).astype(np.float32)
    gap_flag[0] = 1.0

    x = np.column_stack(
        [
            close_ret,
            open_gap,
            body,
            bar_range,
            upper_wick,
            lower_wick,
            close_pos,
            spread_open,
            spread_close,
            hour_sin,
            hour_cos,
            gap_flag,
        ]
    ).astype(np.float32)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, ts_ns


def fit_norm(x, source, chunk=20000):
    s = np.zeros(x.shape[1], np.float64)
    s2 = np.zeros(x.shape[1], np.float64)
    n = 0

    for i in range(0, len(source), chunk):
        ids = source[i:i + chunk]
        z = x[ids].astype(np.float64)
        s += z.sum(axis=0)
        s2 += np.square(z).sum(axis=0)
        n += len(z)

    mean = s / max(n, 1)
    var = s2 / max(n, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-8))
    std[std < 1e-4] = 1.0

    # Cyclic/gap channels are already bounded and easier to interpret raw.
    mean[9:] = 0.0
    std[9:] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def make_batch(rows, arrays, raw, mean, std, device):
    source = arrays["source"][rows]
    offsets = np.arange(SEQ - 1, -1, -1, dtype=np.int64)
    idx = source[:, None] - offsets[None, :]
    x = np.asarray(raw[idx], dtype=np.float32)
    x = (x - mean[None, None, :]) / std[None, None, :]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        torch.from_numpy(np.ascontiguousarray(x)).to(
            device, non_blocking=True
        ),
        source,
    )


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(0.08)

    def forward(self, x):
        z = F.gelu(self.conv1(x))
        z = self.drop(self.conv2(z))
        return F.gelu(self.norm(x + z))


class RawM5DirectionProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        width = 64
        self.input = nn.Sequential(
            nn.Conv1d(channels, width, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(8, width),
        )
        self.tcn = nn.Sequential(
            ResidualTCNBlock(width, 1),
            ResidualTCNBlock(width, 2),
            ResidualTCNBlock(width, 4),
            ResidualTCNBlock(width, 8),
            ResidualTCNBlock(width, 16),
            ResidualTCNBlock(width, 32),
            ResidualTCNBlock(width, 64),
        )
        self.state = nn.Sequential(
            nn.Linear(width * 3, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(0.08),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.horizon = nn.Linear(128, 3)
        self.all9 = nn.Linear(128, 1)

    def forward(self, x):
        z = self.input(x.transpose(1, 2))
        z = self.tcn(z)
        pooled = torch.cat(
            [z[:, :, -1], z.mean(dim=2), z.amax(dim=2)],
            dim=1,
        )
        h = self.state(pooled)
        return {
            "horizon": self.horizon(h),
            "all9": self.all9(h).squeeze(-1),
        }


def weighted_bce(logit, y, valid, strength):
    if not valid.any():
        return logit.float().sum() * 0.0
    raw = F.binary_cross_entropy_with_logits(
        logit.float()[valid],
        y.float()[valid],
        reduction="none",
    )
    w = torch.clamp(
        1.0 + torch.log1p(strength.float()[valid] / 3.0),
        1.0,
        4.0,
    )
    return (raw * w).sum() / w.sum()


def target_tensors(source, stable, device):
    def t(x, dtype=torch.float32):
        return torch.from_numpy(x).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    return {
        "side": t(stable["consensus_side"][source]),
        "valid": t(stable["consensus_valid"][source], torch.bool),
        "strength": t(stable["consensus_strength"][source]),
        "all9": t(stable["all9_side"][source]),
        "all9_valid": t(stable["all9_valid"][source], torch.bool),
        "all9_strength": t(stable["all9_strength"][source]),
    }


def train_epoch(
    model,
    optimizer,
    rows,
    arrays,
    raw,
    stable,
    mean,
    std,
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
            br, arrays, raw, mean, std, device
        )
        target = target_tensors(source, stable, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(x)
            h30 = weighted_bce(
                out["horizon"][:, 0],
                target["side"][:, 0],
                target["valid"][:, 0],
                target["strength"][:, 0],
            )
            h60 = weighted_bce(
                out["horizon"][:, 1],
                target["side"][:, 1],
                target["valid"][:, 1],
                target["strength"][:, 1],
            )
            h120 = weighted_bce(
                out["horizon"][:, 2],
                target["side"][:, 2],
                target["valid"][:, 2],
                target["strength"][:, 2],
            )
            all9 = weighted_bce(
                out["all9"],
                target["all9"],
                target["all9_valid"],
                target["all9_strength"],
            )
            loss = 0.25 * h30 + 0.60 * h60 + 2.50 * h120 + all9

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach())
        count += 1

    return total / max(count, 1)


@torch.no_grad()
def predict(model, rows, arrays, raw, mean, std, device, batch):
    model.eval()
    hp = []
    ap = []
    for start in range(0, len(rows), batch):
        br = rows[start:start + batch]
        x, _ = make_batch(
            br, arrays, raw, mean, std, device
        )
        out = model(x)
        hp.append(torch.sigmoid(out["horizon"]).cpu().numpy())
        ap.append(torch.sigmoid(out["all9"]).cpu().numpy())
    return {
        "horizon": np.concatenate(hp).astype(np.float32),
        "all9": np.concatenate(ap).astype(np.float32),
    }


def report_metrics(rows, pred, arrays, stable):
    source = arrays["source"][rows]
    result = {}
    print()
    print("2023 RAW-M5 DIRECTION LEARNABILITY")
    print("-" * 132)

    for hi, name in enumerate(("H30", "H60", "H120")):
        valid = stable["consensus_valid"][source, hi]
        y = stable["consensus_side"][source, hi][valid].astype(np.uint8)
        p = pred["horizon"][valid, hi]
        auc = safe_auc(y, p)
        acc = float(((p >= 0.5) == y).mean())
        result[f"{name}_auc"] = auc
        result[f"{name}_acc"] = acc
        print(
            f"{name:<6} N={valid.sum():>7,} "
            f"ACC={acc:>7.2%} AUC={auc:>7.4f}"
        )

    valid = stable["all9_valid"][source]
    y = stable["all9_side"][source][valid].astype(np.uint8)
    p = pred["all9"][valid]
    auc = safe_auc(y, p)
    acc = float(((p >= 0.5) == y).mean())
    result["ALL9_auc"] = auc
    result["ALL9_acc"] = acc
    print(
        f"ALL9   N={valid.sum():>7,} "
        f"ACC={acc:>7.2%} AUC={auc:>7.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sample", type=int, default=DEFAULT_TRAIN_SAMPLE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    seed_all(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9.3A RAW M5 DIRECTION LEARNABILITY AUDIT")
    print("=" * 132)
    print("2016-2022 train -> 2023 dev only.")
    print("Raw causal M5 bid/ask path; 2024/2025/2026 remain closed.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    arrays, split, groups, names, feature_mean, feature_std = brain.load_data()
    execution = v671.load_execution_targets()
    daily = v672.load_daily_targets()
    stable = v692.build_stable_targets(execution)
    raw, raw_ts = load_raw_channels()

    if len(raw) != len(execution["year"]):
        raise RuntimeError(
            f"Raw/execution row mismatch: {len(raw)} vs {len(execution['year'])}"
        )

    train_rows = v680.filter_rows(split["train"], arrays, execution)
    val = split["val"]
    years = execution["year"][arrays["source"]]
    rows23 = v680.filter_rows(
        val[years[val] == 2023], arrays, execution
    )

    train_rows = train_rows[arrays["source"][train_rows] >= SEQ - 1]
    rows23 = rows23[arrays["source"][rows23] >= SEQ - 1]

    h120_ok = stable["consensus_valid"][arrays["source"][train_rows], 2]
    train_rows = train_rows[h120_ok]

    rng = np.random.default_rng(args.seed)
    if args.train_sample > 0 and len(train_rows) > args.train_sample:
        train_rows = np.sort(
            rng.choice(
                train_rows,
                size=args.train_sample,
                replace=False,
            )
        )

    train_source = arrays["source"][train_rows]
    mean, std = fit_norm(raw, train_source)

    print("Raw channels:", len(CHANNEL_NAMES))
    print("Channels:", ", ".join(CHANNEL_NAMES))
    print("Sequence rows:", SEQ)
    print("Train sample:", f"{len(train_rows):,}")
    print("2023 dev:", f"{len(rows23):,}")
    print("2024: NOT EVALUATED")
    print("2025: LOCKED")
    print("2026: LOCKED")

    model = RawM5DirectionProbe(len(CHANNEL_NAMES)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("Parameters:", f"{params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    mean_t = torch.from_numpy(np.asarray(feature_mean, np.float32)).view(1, 1, -1).to(device)
    std_t = torch.from_numpy(np.asarray(feature_std, np.float32)).view(1, 1, -1).to(device)
    base_model, base_checkpoint = v691.load_v680(len(names), device)
    base23 = v680.predict(
        base_model,
        rows23,
        arrays,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )

    history = []
    best_h120_auc = -np.inf
    best_wr = -np.inf

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(
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

        pred = predict(
            model,
            rows23,
            arrays,
            raw,
            mean,
            std,
            device,
            args.batch,
        )
        m = report_metrics(rows23, pred, arrays, stable)

        print()
        print("2023 FROZEN V6.8 WITH RAW-M5 H120")
        print("-" * 132)
        system = v691.system_eval(
            rows23,
            pred["horizon"][:, 2],
            base23,
            arrays,
            execution,
            daily,
            base_checkpoint["policy"],
            "V6.9.3A RAW M5 H120",
        )

        sec = time.time() - t0
        row = {
            "epoch": epoch,
            "loss": loss,
            **m,
            "system_win": system["win"],
            "system_mean": system["mean"],
            "system_pf": system["pf"],
            "seconds": sec,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            OUT / "history_v693a.csv",
            index=False,
        )

        best_h120_auc = max(best_h120_auc, m["H120_auc"])
        best_wr = max(best_wr, system["win"])

        print(
            f"EPOCH {epoch}/{args.epochs} loss={loss:.5f} "
            f"H120_AUC={m['H120_auc']:.4f} "
            f"WR={system['win']:.2%} PF={system['pf']:.3f} sec={sec:.1f}"
        )

    if best_h120_auc >= 0.56:
        verdict = "RAW_M5_SIGNAL_FOUND"
    elif best_h120_auc >= 0.53:
        verdict = "WEAK_RAW_M5_SIGNAL"
    else:
        verdict = "NO_CLEAR_RAW_M5_SIGNAL"

    summary = {
        "version": "v6.9.3a",
        "best_h120_auc_2023": float(best_h120_auc),
        "best_system_win_2023": float(best_wr),
        "verdict": verdict,
        "train_sample": int(len(train_rows)),
        "2024_evaluated": False,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v693a.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 132)
    print("VERDICT:", verdict)
    print("Best 2023 H120 AUC:", f"{best_h120_auc:.4f}")
    print("Best 2023 downstream WR:", f"{best_wr:.2%}")

    if verdict == "RAW_M5_SIGNAL_FOUND":
        print("GO: raw M5 path contains materially learnable direction signal.")
        print("Next: fuse raw-path TCN with technical/regime state for production training.")
    elif verdict == "WEAK_RAW_M5_SIGNAL":
        print("PARTIAL: raw M5 helps, but likely needs M1/microstructure or broader context.")
    else:
        print("STOP: do not spend on a larger M5-only network.")
        print("Next: inspect/rescue M1 micro-path or add genuinely exogenous context.")

    print("2024 was not opened by this audit.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
