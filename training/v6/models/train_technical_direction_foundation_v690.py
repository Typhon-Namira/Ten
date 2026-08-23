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


VERSION = "v6.9.0"

OUT = Path(
    "training/artifacts/v6/"
    "technical_direction_foundation_v690"
)

CHAMPION = OUT / "champion_v690.pt"

HORIZONS = (30, 60, 120)

RECENT_STEPS = 24
RECENT_STRIDE = 1

INTRADAY_STEPS = 96
INTRADAY_STRIDE = 3

REGIME_STEPS = 60
REGIME_STRIDE = 24

DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 16
PATIENCE = 5

LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

TOKEN_DIM = 128
TEMPORAL_HIDDEN = 160
EXPERT_DIM = 32
EXPERT_FUSED = 192
STATE_DIM = 256

TERMINAL_SCALE = 40.0

# Start on obvious directional examples, then progressively
# expose the model to harder / noisier states.
GAP_CURRICULUM = (
    20.0,
    20.0,
    10.0,
    10.0,
    5.0,
    5.0,
    3.0,
    3.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

GAP_REPORT = (0.0, 3.0, 5.0, 10.0, 20.0)
CONF_REPORT = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)

SEED = 20260823
STEP_NS = 300_000_000_000


def seed_all(seed=SEED):
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


def make_offsets(steps, stride):
    return np.arange(
        (steps - 1) * stride,
        -1,
        -stride,
        dtype=np.int64,
    )


RECENT_OFFSETS = make_offsets(RECENT_STEPS, RECENT_STRIDE)
INTRADAY_OFFSETS = make_offsets(INTRADAY_STEPS, INTRADAY_STRIDE)
REGIME_OFFSETS = make_offsets(REGIME_STEPS, REGIME_STRIDE)


def max_lookback():
    return int(
        max(
            RECENT_OFFSETS[0],
            INTRADAY_OFFSETS[0],
            REGIME_OFFSETS[0],
        )
    )


def filter_rows(rows, arrays):
    rows = np.asarray(rows, dtype=np.int64)
    source = arrays["source"][rows]
    return rows[source >= max_lookback()]


def load_direction_targets():
    columns = [
        "source_row",
        "year",
    ]

    for h in HORIZONS:
        columns += [
            f"horizon_valid_h{h}",
            f"long_terminal_bps_h{h}",
            f"short_terminal_bps_h{h}",
        ]

    df = pd.read_parquet(
        brain.TARGET_FILE,
        columns=columns,
    )

    source = df["source_row"].to_numpy(np.int64)

    if not np.array_equal(
        source,
        np.arange(len(df), dtype=np.int64),
    ):
        raise RuntimeError(
            "V6.9 requires contiguous V6.6 source_row alignment."
        )

    n = len(df)

    terminal = np.full(
        (n, 2, len(HORIZONS)),
        np.nan,
        dtype=np.float32,
    )

    valid = np.zeros(
        (n, len(HORIZONS)),
        dtype=bool,
    )

    for hi, h in enumerate(HORIZONS):
        long_v = df[
            f"long_terminal_bps_h{h}"
        ].to_numpy(np.float32)

        short_v = df[
            f"short_terminal_bps_h{h}"
        ].to_numpy(np.float32)

        ok = (
            df[f"horizon_valid_h{h}"].to_numpy(np.uint8)
            == 1
        )
        ok &= np.isfinite(long_v) & np.isfinite(short_v)

        terminal[:, 0, hi] = long_v
        terminal[:, 1, hi] = short_v
        valid[:, hi] = ok

    # Direction is now independent of TP/SL choice.
    # 0 = LONG is economically better at terminal horizon.
    # 1 = SHORT is economically better at terminal horizon.
    gap = (
        terminal[:, 1, :]
        - terminal[:, 0, :]
    ).astype(np.float32)

    side = (
        gap > 0.0
    ).astype(np.float32)

    return {
        "year": df["year"].to_numpy(np.int16),
        "terminal": terminal,
        "valid": valid,
        "gap": gap,
        "side": side,
    }


def context_tensor(
    source,
    offsets,
    features,
    mean_t,
    std_t,
    device,
):
    idx = source[:, None] - offsets[None, :]

    raw = np.ascontiguousarray(
        features[idx],
        dtype=np.float32,
    )

    x = torch.from_numpy(raw).to(
        device,
        non_blocking=True,
    )

    x = (x - mean_t) / std_t

    return torch.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def make_contexts(
    rows,
    arrays,
    mean_t,
    std_t,
    device,
):
    source = arrays["source"][rows]

    recent = context_tensor(
        source,
        RECENT_OFFSETS,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )

    intraday = context_tensor(
        source,
        INTRADAY_OFFSETS,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )

    regime = context_tensor(
        source,
        REGIME_OFFSETS,
        arrays["features"],
        mean_t,
        std_t,
        device,
    )

    return recent, intraday, regime, source


def make_targets(source, direction, excursion, device):
    side = direction["side"][source]
    gap = direction["gap"][source]
    valid = direction["valid"][source]
    terminal = direction["terminal"][source]

    # V6.6 excursion is already log1p(MFE/MAE), ordered as:
    # long H30/H60/H120, short H30/H60/H120.
    exc = excursion[source].astype(np.float32)

    def t(x, dtype=torch.float32):
        return torch.from_numpy(x).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    return {
        "side": t(side),
        "gap": t(gap),
        "valid": t(valid, torch.bool),
        "terminal": t(
            np.clip(
                terminal / TERMINAL_SCALE,
                -3.0,
                3.0,
            ).astype(np.float32)
        ),
        "terminal_bps": t(terminal),
        "excursion": t(exc),
    }


class TechnicalDirectionFoundationV690(nn.Module):
    def __init__(self, feature_dim, groups):
        super().__init__()

        self.group_names = list(groups.keys())
        self.group_indices = {
            name: tuple(int(i) for i in ids)
            for name, ids in groups.items()
        }

        # Shared temporal projection. The temporal branches learn
        # immediate price action, intraday state and multi-day regime.
        self.project = nn.Sequential(
            nn.Linear(feature_dim, TOKEN_DIM),
            nn.GELU(),
            nn.LayerNorm(TOKEN_DIM),
        )

        self.recent_gru = nn.GRU(
            TOKEN_DIM,
            TEMPORAL_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.intraday_gru = nn.GRU(
            TOKEN_DIM,
            TEMPORAL_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.regime_gru = nn.GRU(
            TOKEN_DIM,
            TEMPORAL_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.snapshot = nn.Sequential(
            nn.Linear(feature_dim, 160),
            nn.GELU(),
            nn.LayerNorm(160),
            nn.Linear(160, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        # Explicit technical specialists. Each specialist sees only
        # its own technical family at the current market state.
        self.experts = nn.ModuleDict()
        self.expert_side_heads = nn.ModuleDict()

        for name in self.group_names:
            dim = len(self.group_indices[name])

            self.experts[name] = nn.Sequential(
                nn.Linear(dim, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Dropout(0.08),
                nn.Linear(64, EXPERT_DIM),
                nn.GELU(),
                nn.LayerNorm(EXPERT_DIM),
            )

            # Every technical specialist must independently learn
            # whether its evidence supports LONG or SHORT per horizon.
            self.expert_side_heads[name] = nn.Linear(
                EXPERT_DIM,
                len(HORIZONS),
            )

        expert_cat_dim = EXPERT_DIM * len(self.group_names)

        self.expert_fuse = nn.Sequential(
            nn.Linear(expert_cat_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.10),
            nn.Linear(256, EXPERT_FUSED),
            nn.GELU(),
            nn.LayerNorm(EXPERT_FUSED),
        )

        fusion_dim = (
            TEMPORAL_HIDDEN * 3
            + 128
            + EXPERT_FUSED
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.15),
            nn.Linear(512, STATE_DIM),
            nn.GELU(),
            nn.LayerNorm(STATE_DIM),
        )

        # Dedicated horizon directions: H30 / H60 / H120.
        self.side_head = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, len(HORIZONS)),
        )

        # Predict executable terminal utility for both LONG and SHORT.
        self.terminal_head = nn.Sequential(
            nn.Linear(STATE_DIM, 192),
            nn.GELU(),
            nn.Linear(192, 2 * len(HORIZONS)),
        )

        # Predict path geometry: MFE / MAE for each side+horizon.
        self.excursion_head = nn.Sequential(
            nn.Linear(STATE_DIM, 192),
            nn.GELU(),
            nn.Linear(192, 6 * 2),
        )

    def encode_temporal(self, x, gru):
        z = self.project(x)
        _, h = gru(z)
        return h[-1]

    def forward(self, recent, intraday, regime):
        recent_h = self.encode_temporal(
            recent,
            self.recent_gru,
        )

        intraday_h = self.encode_temporal(
            intraday,
            self.intraday_gru,
        )

        regime_h = self.encode_temporal(
            regime,
            self.regime_gru,
        )

        latest = recent[:, -1, :]
        snap = self.snapshot(latest)

        expert_states = []
        expert_logits = {}

        for name in self.group_names:
            ids = self.group_indices[name]
            e = self.experts[name](latest[:, ids])
            expert_states.append(e)
            expert_logits[name] = self.expert_side_heads[name](e)

        expert_state = self.expert_fuse(
            torch.cat(expert_states, dim=1)
        )

        state = self.fusion(
            torch.cat(
                [
                    recent_h,
                    intraday_h,
                    regime_h,
                    snap,
                    expert_state,
                ],
                dim=1,
            )
        )

        terminal = self.terminal_head(state).view(
            -1,
            2,
            len(HORIZONS),
        )

        excursion = self.excursion_head(state).view(
            -1,
            6,
            2,
        )

        return {
            "state": state,
            "side_logits": self.side_head(state),
            "terminal": terminal,
            "excursion": excursion,
            "expert_side_logits": expert_logits,
        }


def weighted_direction_loss(logits, y, gap, valid, min_gap):
    mask = valid & (torch.abs(gap) >= min_gap)

    if not mask.any():
        return logits.float().sum() * 0.0

    raw = F.binary_cross_entropy_with_logits(
        logits.float()[mask],
        y.float()[mask],
        reduction="none",
    )

    weight = torch.clamp(
        torch.abs(gap[mask]) / 10.0,
        1.0,
        8.0,
    )

    return (raw * weight).sum() / weight.sum()


def compute_loss(out, target, min_gap):
    side_loss = weighted_direction_loss(
        out["side_logits"],
        target["side"],
        target["gap"],
        target["valid"],
        min_gap,
    )

    expert_losses = []

    for logits in out["expert_side_logits"].values():
        expert_losses.append(
            weighted_direction_loss(
                logits,
                target["side"],
                target["gap"],
                target["valid"],
                min_gap,
            )
        )

    expert_loss = torch.stack(expert_losses).mean()

    terminal_mask = target["valid"].unsqueeze(1).expand(
        -1,
        2,
        -1,
    )

    terminal_loss = F.smooth_l1_loss(
        out["terminal"].float()[terminal_mask],
        target["terminal"].float()[terminal_mask],
        beta=0.15,
    )

    # Directional utility margin. This directly teaches the model
    # how much SHORT is better/worse than LONG at each horizon.
    pred_gap = (
        out["terminal"][:, 1, :]
        - out["terminal"][:, 0, :]
    )

    true_gap = torch.clamp(
        target["gap"].float() / TERMINAL_SCALE,
        -3.0,
        3.0,
    )

    margin_loss = F.smooth_l1_loss(
        pred_gap[target["valid"]],
        true_gap[target["valid"]],
        beta=0.10,
    )

    excursion_loss = F.smooth_l1_loss(
        out["excursion"].float(),
        target["excursion"].float(),
        beta=0.15,
    )

    # Direction is deliberately dominant. Auxiliary heads teach
    # future path geometry instead of competing with trading-policy
    # objectives such as WHEN/rank/TP-SL selection.
    total = (
        4.00 * side_loss
        + 0.80 * margin_loss
        + 0.55 * terminal_loss
        + 0.40 * excursion_loss
        + 0.20 * expert_loss
    )

    return total, {
        "total": total,
        "side": side_loss,
        "margin": margin_loss,
        "terminal": terminal_loss,
        "excursion": excursion_loss,
        "experts": expert_loss,
    }


def train_epoch(
    model,
    optimizer,
    rows,
    arrays,
    direction,
    mean_t,
    std_t,
    scaler,
    device,
    rng,
    batch_size,
    min_gap,
):
    model.train()

    order = rng.permutation(rows)
    sums = {}
    batches = 0

    for start in range(0, len(order), batch_size):
        batch_rows = order[start:start + batch_size]

        recent, intraday, regime, source = make_contexts(
            batch_rows,
            arrays,
            mean_t,
            std_t,
            device,
        )

        target = make_targets(
            source,
            direction,
            arrays["excursion"],
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(recent, intraday, regime)
            loss, parts = compute_loss(
                out,
                target,
                min_gap,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(optimizer)
        scaler.update()

        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(
                value.detach()
            )

        batches += 1

    return {
        key: value / max(batches, 1)
        for key, value in sums.items()
    }


@torch.no_grad()
def predict(
    model,
    rows,
    arrays,
    mean_t,
    std_t,
    device,
    batch_size,
):
    model.eval()

    side_chunks = []
    terminal_chunks = []
    expert_chunks = {
        name: []
        for name in model.group_names
    }

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]

        recent, intraday, regime, _ = make_contexts(
            batch_rows,
            arrays,
            mean_t,
            std_t,
            device,
        )

        out = model(recent, intraday, regime)

        side_chunks.append(
            torch.sigmoid(out["side_logits"])
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        terminal_chunks.append(
            out["terminal"]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        for name in model.group_names:
            expert_chunks[name].append(
                torch.sigmoid(
                    out["expert_side_logits"][name]
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return {
        "side": np.concatenate(side_chunks, axis=0),
        "terminal": np.concatenate(terminal_chunks, axis=0),
        "experts": {
            name: np.concatenate(parts, axis=0)
            for name, parts in expert_chunks.items()
        },
    }


def evaluate_predictions(rows, pred, arrays, direction, label):
    source = arrays["source"][rows]
    y = direction["side"][source]
    gap = direction["gap"][source]
    valid = direction["valid"][source]
    p = pred["side"]

    records = []

    print()
    print(f"{label} DIRECTION EVALUATION")
    print("=" * 118)

    for hi, h in enumerate(HORIZONS):
        print(f"H{h}")

        for min_gap in GAP_REPORT:
            mask = valid[:, hi] & (
                np.abs(gap[:, hi]) >= min_gap
            )

            yy = y[mask, hi].astype(np.uint8)
            pp = p[mask, hi]

            auc = safe_auc(yy, pp)
            acc = (
                float(((pp >= 0.5) == yy).mean())
                if len(yy)
                else np.nan
            )

            records.append(
                {
                    "label": label,
                    "horizon": h,
                    "view": "gap",
                    "threshold": min_gap,
                    "n": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "acc": acc,
                    "auc": auc,
                }
            )

            print(
                f"  gap>={min_gap:>4.0f}bps "
                f"N={mask.sum():>7,} "
                f"ACC={acc:>7.2%} "
                f"AUC={auc:>7.4f}"
            )

        confidence = np.abs(p[:, hi] - 0.5) * 2.0

        print("  confidence frontier")

        for conf in CONF_REPORT:
            mask = valid[:, hi] & (confidence >= conf)
            yy = y[mask, hi].astype(np.uint8)
            pp = p[mask, hi]

            acc = (
                float(((pp >= 0.5) == yy).mean())
                if len(yy)
                else np.nan
            )
            auc = safe_auc(yy, pp)

            records.append(
                {
                    "label": label,
                    "horizon": h,
                    "view": "confidence",
                    "threshold": conf,
                    "n": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "acc": acc,
                    "auc": auc,
                }
            )

            print(
                f"    conf>={conf:>4.2f} "
                f"COV={mask.mean():>7.2%} "
                f"N={mask.sum():>7,} "
                f"ACC={acc:>7.2%}"
            )

    # Technical expert diagnostics: each specialist has its own
    # directional head, so we can later identify which technical
    # families truly generalize out-of-sample.
    expert_rows = []

    print()
    print(f"{label} TECHNICAL EXPERT AUC")
    print("-" * 118)

    for name, expert_p in pred["experts"].items():
        values = []

        for hi, h in enumerate(HORIZONS):
            mask = valid[:, hi] & (
                np.abs(gap[:, hi]) >= 5.0
            )

            auc = safe_auc(
                y[mask, hi].astype(np.uint8),
                expert_p[mask, hi],
            )

            values.append(auc)
            expert_rows.append(
                {
                    "label": label,
                    "expert": name,
                    "horizon": h,
                    "gap": 5.0,
                    "auc": auc,
                    "n": int(mask.sum()),
                }
            )

        print(
            f"{name:<24} "
            + " ".join(
                f"H{h}={v:.4f}"
                for h, v in zip(HORIZONS, values)
            )
        )

    return pd.DataFrame(records), pd.DataFrame(expert_rows)


def champion_score(metrics):
    # Select only on 2023. Never use 2024 to choose an epoch.
    base = metrics[
        (metrics["view"] == "gap")
        & (metrics["threshold"] == 0.0)
    ]

    strong = metrics[
        (metrics["view"] == "gap")
        & (metrics["threshold"] == 10.0)
    ]

    high_conf = metrics[
        (metrics["view"] == "confidence")
        & (metrics["threshold"] == 0.30)
        & (metrics["coverage"] >= 0.05)
    ]

    auc = float(base["auc"].mean())
    strong_acc = float(strong["acc"].mean())

    if len(high_conf):
        conf_acc = float(high_conf["acc"].mean())
    else:
        conf_acc = 0.50

    return (
        auc
        + 0.60 * strong_acc
        + 0.20 * conf_acc
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny CPU-safe architecture/data-path validation run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
    )
    args = parser.parse_args()

    started = time.time()
    seed_all()

    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9 TECHNICAL DIRECTION FOUNDATION")
    print("=" * 118)
    print("Version:", VERSION)

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

    direction = load_direction_targets()

    if len(direction["year"]) != len(arrays["source"]):
        raise RuntimeError("Direction target alignment failure.")

    train_rows = filter_rows(split["train"], arrays)
    val_rows = filter_rows(split["val"], arrays)

    years = direction["year"]

    rows23 = val_rows[years[val_rows] == 2023]
    rows24 = val_rows[years[val_rows] == 2024]

    print("Feature dim:", arrays["features"].shape[1])
    print("Technical groups:")

    for name, ids in groups.items():
        print(f"  {name:<24} {len(ids):>4} features")

    print("Train:", f"{len(train_rows):,}")
    print("2023 dev:", f"{len(rows23):,}")
    print("2024 research benchmark:", f"{len(rows24):,}")
    print("2025: LOCKED / NOT EVALUATED")
    print("2026: LOCKED / NOT EVALUATED")

    epochs = args.epochs
    batch_size = args.batch

    if args.smoke:
        # Keep the smoke test genuinely cheap on CPU. It validates
        # loading, alignment, forward/backward, metrics and saving.
        train_rows = train_rows[-min(256, len(train_rows)):]
        rows23 = rows23[:min(128, len(rows23))]
        rows24 = rows24[:min(128, len(rows24))]
        epochs = 1
        batch_size = min(16, batch_size)

        print()
        print("SMOKE MODE")
        print("Train rows:", len(train_rows))
        print("2023 rows:", len(rows23))
        print("2024 rows:", len(rows24))

    mean_t = torch.from_numpy(mean).to(device).view(1, 1, -1)
    std_t = torch.from_numpy(std).to(device).view(1, 1, -1)

    model = TechnicalDirectionFoundationV690(
        arrays["features"].shape[1],
        groups,
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", f"{params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=LR * 0.08,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    rng = np.random.default_rng(SEED)

    best_score = -np.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        min_gap = GAP_CURRICULUM[
            min(epoch - 1, len(GAP_CURRICULUM) - 1)
        ]

        epoch_started = time.time()

        losses = train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
            direction,
            mean_t,
            std_t,
            scaler,
            device,
            rng,
            batch_size,
            min_gap,
        )

        scheduler.step()

        pred23 = predict(
            model,
            rows23,
            arrays,
            mean_t,
            std_t,
            device,
            batch_size,
        )

        metrics23, experts23 = evaluate_predictions(
            rows23,
            pred23,
            arrays,
            direction,
            "2023",
        )

        score = champion_score(metrics23)

        row = {
            "epoch": epoch,
            "min_gap": min_gap,
            "score_2023": score,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_started,
            **{
                f"loss_{k}": v
                for k, v in losses.items()
            },
        }

        history.append(row)

        print()
        print(
            f"EPOCH {epoch}/{epochs} "
            f"gap>={min_gap:g} "
            f"score23={score:.6f} "
            f"loss={losses['total']:.5f} "
            f"sec={row['seconds']:.1f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "version": VERSION,
                    "epoch": epoch,
                    "score_2023": score,
                    "model": model.state_dict(),
                    "feature_names": names,
                    "groups": groups,
                    "mean": mean,
                    "std": std,
                    "config": {
                        "recent_steps": RECENT_STEPS,
                        "recent_stride": RECENT_STRIDE,
                        "intraday_steps": INTRADAY_STEPS,
                        "intraday_stride": INTRADAY_STRIDE,
                        "regime_steps": REGIME_STEPS,
                        "regime_stride": REGIME_STRIDE,
                        "terminal_scale": TERMINAL_SCALE,
                    },
                },
                CHAMPION,
            )

            metrics23.to_csv(
                OUT / "champion_2023_direction.csv",
                index=False,
            )

            experts23.to_csv(
                OUT / "champion_2023_experts.csv",
                index=False,
            )

            print("NEW CHAMPION:", CHAMPION)
        else:
            stale += 1

        pd.DataFrame(history).to_csv(
            OUT / "training_history_v690.csv",
            index=False,
        )

        if not args.smoke and stale >= PATIENCE:
            print("Early stopping.")
            break

    if not CHAMPION.exists():
        raise RuntimeError("No V6.9 champion was saved.")

    checkpoint = torch.load(
        CHAMPION,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])

    print()
    print("=" * 118)
    print("FROZEN CHAMPION")
    print("Epoch:", checkpoint["epoch"])
    print("2023 selection score:", checkpoint["score_2023"])

    # 2024 is reporting only. It never participates in champion
    # selection or threshold tuning.
    pred24 = predict(
        model,
        rows24,
        arrays,
        mean_t,
        std_t,
        device,
        batch_size,
    )

    metrics24, experts24 = evaluate_predictions(
        rows24,
        pred24,
        arrays,
        direction,
        "2024",
    )

    metrics24.to_csv(
        OUT / "frozen_2024_direction.csv",
        index=False,
    )

    experts24.to_csv(
        OUT / "frozen_2024_experts.csv",
        index=False,
    )

    summary = {
        "version": VERSION,
        "champion_epoch": int(checkpoint["epoch"]),
        "selection_2023": float(checkpoint["score_2023"]),
        "parameters": int(params),
        "direction_target": (
            "short_terminal_bps > long_terminal_bps, "
            "independent of TP/SL action choice"
        ),
        "horizons": list(HORIZONS),
        "2025_evaluated": False,
        "2026_evaluated": False,
        "smoke": bool(args.smoke),
        "seconds": float(time.time() - started),
    }

    with open(
        OUT / "summary_v690.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print(json.dumps(summary, indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
