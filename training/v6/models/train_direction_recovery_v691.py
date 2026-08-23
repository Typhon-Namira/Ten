from pathlib import Path
import argparse
import json
import math
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
import training.v6.models.train_technical_direction_foundation_v690 as v690


VERSION = "v6.9.1"

OUT = Path(
    "training/artifacts/v6/"
    "direction_recovery_v691"
)

CHAMPION = OUT / "champion_v691.pt"
V680_CHAMPION = (
    Path("training/artifacts/v6/")
    / "multiscale_execution_brain_v680"
    / "champion_v680.pt"
)

HORIZONS = (30, 60, 120)

DEFAULT_EPOCHS = 24
DEFAULT_BATCH = 160
PATIENCE = 7

LR = 2.5e-4
MIN_LR = 1.5e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NET_SCALE = 30.0

# Heavy direction curriculum. Master direction is the exact same
# best-LONG-vs-best-SHORT target used by V6.8 and by the oracle-side
# decomposition experiment. Training starts on obvious economic
# direction gaps, then progressively includes harder states.
GAP_CURRICULUM = (
    20.0, 20.0,
    15.0, 15.0,
    10.0, 10.0,
    7.0, 7.0,
    5.0, 5.0,
    3.0, 3.0,
    2.0, 2.0,
    1.0, 1.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
)

REPORT_GAPS = (0.0, 1.0, 3.0, 5.0, 10.0, 20.0)

SEED = 20260823


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


def profit_factor(pnl):
    pnl = np.asarray(pnl, np.float64)
    gain = pnl[pnl > 0].sum()
    loss = -pnl[pnl < 0].sum()

    if loss <= 0:
        return np.inf

    return float(gain / loss)


def build_direction_targets(execution):
    valid = execution["valid"]
    gross = execution["gross"].astype(np.float32)

    net = gross - 0.5
    safe = np.where(valid, net, -np.inf)

    best_long = np.max(safe[:, :9], axis=1).astype(np.float32)
    best_short = np.max(safe[:, 9:], axis=1).astype(np.float32)

    master_gap = (best_short - best_long).astype(np.float32)
    master_side = (master_gap > 0.0).astype(np.float32)

    horizon_side = np.zeros(
        (len(gross), 3),
        dtype=np.float32,
    )
    horizon_gap = np.zeros(
        (len(gross), 3),
        dtype=np.float32,
    )

    for hi in range(3):
        lo = hi * 3
        hi_task = lo + 3

        long_h = np.max(
            safe[:, lo:hi_task],
            axis=1,
        )
        short_h = np.max(
            safe[:, 9 + lo:9 + hi_task],
            axis=1,
        )

        horizon_gap[:, hi] = (
            short_h - long_h
        ).astype(np.float32)
        horizon_side[:, hi] = (
            horizon_gap[:, hi] > 0.0
        ).astype(np.float32)

    # Nine matched LONG-vs-SHORT action pairs. These auxiliary labels
    # force direction to be consistent across horizon/barrier choices
    # instead of learning only the noisy max-of-max master label.
    pair_gap = np.zeros(
        (len(gross), 9),
        dtype=np.float32,
    )
    pair_side = np.zeros(
        (len(gross), 9),
        dtype=np.float32,
    )
    pair_valid = np.zeros(
        (len(gross), 9),
        dtype=bool,
    )

    for j in range(9):
        sj = j + 9
        ok = valid[:, j] & valid[:, sj]
        g = net[:, sj] - net[:, j]

        pair_valid[:, j] = ok & np.isfinite(g)
        pair_gap[:, j] = g.astype(np.float32)
        pair_side[:, j] = (g > 0.0).astype(np.float32)

    best_values = np.stack(
        [best_long, best_short],
        axis=1,
    ).astype(np.float32)

    return {
        "master_side": master_side,
        "master_gap": master_gap,
        "horizon_side": horizon_side,
        "horizon_gap": horizon_gap,
        "pair_side": pair_side,
        "pair_gap": pair_gap,
        "pair_valid": pair_valid,
        "best_values": best_values,
    }


class DirectionRecoveryBrainV691(
    v690.TechnicalDirectionFoundationV690
):
    def __init__(self, feature_dim, groups):
        super().__init__(feature_dim, groups)

        # Base V6.9 side_head has three outputs and is used here as
        # H30/H60/H120 execution-aware direction supervision.
        self.master_side_head = nn.Sequential(
            nn.Linear(v690.STATE_DIM, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Dropout(0.12),
            nn.Linear(192, 1),
        )

        self.pair_side_head = nn.Sequential(
            nn.Linear(v690.STATE_DIM, 192),
            nn.GELU(),
            nn.Linear(192, 9),
        )

        self.best_side_value_head = nn.Sequential(
            nn.Linear(v690.STATE_DIM, 192),
            nn.GELU(),
            nn.Linear(192, 2),
        )

    def forward(self, recent, intraday, regime):
        out = super().forward(recent, intraday, regime)
        state = out["state"]

        out["horizon_side_logits"] = out.pop("side_logits")
        out["master_side_logit"] = (
            self.master_side_head(state).squeeze(-1)
        )
        out["pair_side_logits"] = self.pair_side_head(state)
        out["best_side_values"] = self.best_side_value_head(state)

        return out


def focal_gap_bce(
    logits,
    y,
    gap,
    valid,
    min_gap,
    gamma=1.5,
):
    mask = valid & (torch.abs(gap) >= min_gap)

    if not mask.any():
        return logits.float().sum() * 0.0

    z = logits.float()[mask]
    target = y.float()[mask]

    raw = F.binary_cross_entropy_with_logits(
        z,
        target,
        reduction="none",
    )

    prob = torch.sigmoid(z)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    focal = torch.pow(1.0 - pt, gamma)

    economic_weight = torch.clamp(
        torch.abs(gap.float()[mask]) / 8.0,
        1.0,
        10.0,
    )

    weight = focal * economic_weight

    return (raw * weight).sum() / torch.clamp(
        weight.sum(),
        min=1e-6,
    )


def make_targets(
    source,
    target_np,
    path_np,
    excursion,
    device,
):
    def t(x, dtype=torch.float32):
        return torch.from_numpy(x).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    best_values = target_np["best_values"][source]

    terminal = path_np["terminal"][source]
    terminal_valid = path_np["valid"][source]

    return {
        "master_side": t(target_np["master_side"][source]),
        "master_gap": t(target_np["master_gap"][source]),
        "horizon_side": t(target_np["horizon_side"][source]),
        "horizon_gap": t(target_np["horizon_gap"][source]),
        "pair_side": t(target_np["pair_side"][source]),
        "pair_gap": t(target_np["pair_gap"][source]),
        "pair_valid": t(
            target_np["pair_valid"][source],
            torch.bool,
        ),
        "best_values": t(
            np.clip(
                best_values / NET_SCALE,
                -3.0,
                3.0,
            ).astype(np.float32)
        ),
        "terminal": t(
            np.clip(
                terminal / v690.TERMINAL_SCALE,
                -3.0,
                3.0,
            ).astype(np.float32)
        ),
        "terminal_valid": t(
            terminal_valid,
            torch.bool,
        ),
        "excursion": t(
            excursion[source].astype(np.float32)
        ),
    }


def compute_loss(out, target, min_gap):
    batch = len(target["master_side"])

    master_valid = torch.ones(
        batch,
        dtype=torch.bool,
        device=target["master_side"].device,
    )

    master = focal_gap_bce(
        out["master_side_logit"],
        target["master_side"],
        target["master_gap"],
        master_valid,
        min_gap,
        gamma=1.5,
    )

    horizon_losses = []

    for hi in range(3):
        horizon_losses.append(
            focal_gap_bce(
                out["horizon_side_logits"][:, hi],
                target["horizon_side"][:, hi],
                target["horizon_gap"][:, hi],
                master_valid,
                max(min_gap * 0.75, 0.0),
                gamma=1.25,
            )
        )

    horizon = torch.stack(horizon_losses).mean()

    pair_losses = []

    for j in range(9):
        pair_losses.append(
            focal_gap_bce(
                out["pair_side_logits"][:, j],
                target["pair_side"][:, j],
                target["pair_gap"][:, j],
                target["pair_valid"][:, j],
                max(min_gap * 0.50, 0.0),
                gamma=1.0,
            )
        )

    pair = torch.stack(pair_losses).mean()

    best_value = F.smooth_l1_loss(
        out["best_side_values"].float(),
        target["best_values"].float(),
        beta=0.15,
    )

    pred_master_gap = (
        out["best_side_values"][:, 1]
        - out["best_side_values"][:, 0]
    )

    true_master_gap = torch.clamp(
        target["master_gap"].float() / NET_SCALE,
        -3.0,
        3.0,
    )

    margin = F.smooth_l1_loss(
        pred_master_gap,
        true_master_gap,
        beta=0.10,
    )

    terminal_mask = (
        target["terminal_valid"]
        .unsqueeze(1)
        .expand(-1, 2, -1)
    )

    terminal = F.smooth_l1_loss(
        out["terminal"].float()[terminal_mask],
        target["terminal"].float()[terminal_mask],
        beta=0.15,
    )

    excursion = F.smooth_l1_loss(
        out["excursion"].float(),
        target["excursion"].float(),
        beta=0.15,
    )

    # The technical experts are trained against execution-aware
    # horizon direction, not terminal direction. This forces trend,
    # structure, liquidity, breakout/retest, momentum, etc. to carry
    # useful directional information independently.
    expert_losses = []

    for logits in out["expert_side_logits"].values():
        for hi in range(3):
            expert_losses.append(
                focal_gap_bce(
                    logits[:, hi],
                    target["horizon_side"][:, hi],
                    target["horizon_gap"][:, hi],
                    master_valid,
                    max(min_gap * 0.75, 0.0),
                    gamma=1.0,
                )
            )

    experts = torch.stack(expert_losses).mean()

    # Master side dominates because it is the exact variable whose
    # oracle replacement previously unlocked the high-win regime.
    total = (
        5.00 * master
        + 1.75 * horizon
        + 0.85 * pair
        + 0.90 * margin
        + 0.70 * best_value
        + 0.30 * terminal
        + 0.20 * excursion
        + 0.25 * experts
    )

    return total, {
        "total": total,
        "master": master,
        "horizon": horizon,
        "pair": pair,
        "margin": margin,
        "best_value": best_value,
        "terminal": terminal,
        "excursion": excursion,
        "experts": experts,
    }


def lr_factor(epoch, epochs):
    # Two-epoch warmup, then cosine decay.
    warmup = min(2, epochs)

    if epoch <= warmup:
        return epoch / max(warmup, 1)

    progress = (
        (epoch - warmup)
        / max(epochs - warmup, 1)
    )

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = MIN_LR / LR
    return floor + (1.0 - floor) * cosine


def train_epoch(
    model,
    optimizer,
    rows,
    arrays,
    target_np,
    path_np,
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

        recent, intraday, regime, source = v690.make_contexts(
            batch_rows,
            arrays,
            mean_t,
            std_t,
            device,
        )

        target = make_targets(
            source,
            target_np,
            path_np,
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
def predict_direction(
    model,
    rows,
    arrays,
    mean_t,
    std_t,
    device,
    batch_size,
):
    model.eval()

    master = []
    horizon = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]

        recent, intraday, regime, _ = v690.make_contexts(
            batch_rows,
            arrays,
            mean_t,
            std_t,
            device,
        )

        out = model(recent, intraday, regime)

        master.append(
            torch.sigmoid(
                out["master_side_logit"]
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        horizon.append(
            torch.sigmoid(
                out["horizon_side_logits"]
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return {
        "master": np.concatenate(master),
        "horizon": np.concatenate(horizon, axis=0),
    }


def direction_metrics(rows, pred, arrays, target_np, label):
    source = arrays["source"][rows]
    y = target_np["master_side"][source].astype(np.uint8)
    gap = target_np["master_gap"][source]
    p = pred["master"]

    records = []

    print()
    print(f"{label} MASTER DIRECTION")
    print("-" * 125)

    for min_gap in REPORT_GAPS:
        mask = np.abs(gap) >= min_gap
        yy = y[mask]
        pp = p[mask]

        acc = (
            float(((pp >= 0.5) == yy).mean())
            if len(yy)
            else np.nan
        )
        auc = safe_auc(yy, pp)

        records.append(
            {
                "gap": min_gap,
                "n": int(mask.sum()),
                "acc": acc,
                "auc": auc,
            }
        )

        print(
            f"|GAP|>={min_gap:>4.0f}bps "
            f"N={mask.sum():>7,} "
            f"ACC={acc:>7.2%} "
            f"AUC={auc:>7.4f}"
        )

    return pd.DataFrame(records)


def load_v680(
    feature_dim,
    device,
):
    if not V680_CHAMPION.exists():
        raise FileNotFoundError(
            f"Missing V6.8 champion: {V680_CHAMPION}"
        )

    checkpoint = torch.load(
        V680_CHAMPION,
        map_location=device,
        weights_only=False,
    )

    model = v680.MultiScaleExecutionBrainV680(
        feature_dim
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, checkpoint


def system_eval(
    rows,
    side_prob,
    base_pred,
    arrays,
    execution,
    daily,
    frozen_policy,
    label,
):
    source = arrays["source"][rows]
    valid = execution["valid"][source]
    gross = execution["gross"][source]

    chosen_side = (side_prob >= 0.5).astype(np.int8)

    chosen_task = np.full(
        len(rows),
        -1,
        dtype=np.int64,
    )

    long_mask = chosen_side == 0
    short_mask = chosen_side == 1

    if long_mask.any():
        chosen_task[long_mask] = np.argmax(
            base_pred["net"][long_mask, :9],
            axis=1,
        )

    if short_mask.any():
        chosen_task[short_mask] = (
            9
            + np.argmax(
                base_pred["net"][short_mask, 9:],
                axis=1,
            )
        )

    idx = np.arange(len(rows))
    pnl_all = gross[idx, chosen_task] - 0.5

    # CRITICAL ISOLATION RULE:
    # keep V6.8's exact timing/readiness gate. Only the direction
    # decision is replaced. This means an improvement in final win
    # rate is attributable to direction, not to a new signal policy.
    when_score = (
        0.10 * base_pred["when"][:, 0]
        + 0.30 * base_pred["when"][:, 1]
        + 0.35 * base_pred["when"][:, 2]
        + 0.25 * base_pred["when"][:, 3]
        + 0.20 * base_pred["rank"]
    )

    base_confidence = np.abs(
        base_pred["side"] - 0.5
    ) * 2.0

    day = daily["day_ns"][source]
    counts = pd.Series(day).value_counts()

    eligible_days = set(
        int(x)
        for x in counts[
            counts >= v680.MIN_DAY_BARS
        ].index
    )

    threshold = float(frozen_policy["threshold"])
    conf_min = float(frozen_policy["confidence"])

    used = set()
    pnl = []

    for i in range(len(rows)):
        d = int(day[i])

        if d not in eligible_days or d in used:
            continue

        if when_score[i] < threshold:
            continue

        if base_confidence[i] < conf_min:
            continue

        used.add(d)
        pnl.append(float(pnl_all[i]))

    pnl = np.asarray(pnl, np.float64)

    n = len(pnl)
    coverage = (
        n / len(eligible_days)
        if eligible_days
        else np.nan
    )
    win = float((pnl > 0).mean()) if n else np.nan
    mean = float(pnl.mean()) if n else np.nan
    pf = profit_factor(pnl) if n else np.nan

    result = {
        "label": label,
        "trades": int(n),
        "coverage": float(coverage),
        "win": win,
        "mean": mean,
        "pf": pf,
    }

    print(
        f"{label:<30} "
        f"N={n:>4} "
        f"COV={coverage:>7.2%} "
        f"WIN={win:>7.2%} "
        f"MEAN={mean:>+8.3f} "
        f"PF={pf:>7.3f}"
    )

    return result


def selection_score(system, dirmetrics):
    # Downstream V6.8 win rate is the primary champion criterion.
    # Direction AUC at economically meaningful gap>=10 is a small
    # stabilizer so a lucky trading subset cannot dominate selection.
    row10 = dirmetrics[
        dirmetrics["gap"] == 10.0
    ].iloc[0]

    win = system["win"]
    mean = system["mean"]
    pf = system["pf"]
    auc10 = float(row10["auc"])

    if system["trades"] < 150 or system["coverage"] < 0.70:
        return -1e9

    return float(
        5.0 * win
        + 0.50 * auc10
        + 0.25 * np.tanh(mean / 5.0)
        + 0.15 * np.log(max(pf, 1e-6))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    seed_all(args.seed)

    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9.1 DIRECTION RECOVERY")
    print("=" * 130)
    print("Goal: replace ONLY V6.8 direction and recover oracle-side edge.")

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
    target_np = build_direction_targets(execution)
    path_np = v690.load_direction_targets()

    features = arrays["features"]

    years = execution["year"][arrays["source"]]

    train_rows = v680.filter_rows(
        split["train"],
        arrays,
        execution,
    )

    val = split["val"]

    rows23 = v680.filter_rows(
        val[years[val] == 2023],
        arrays,
        execution,
    )

    rows24 = v680.filter_rows(
        val[years[val] == 2024],
        arrays,
        execution,
    )

    print("Features:", features.shape[1])
    print("Train:", f"{len(train_rows):,}")
    print("2023 dev:", f"{len(rows23):,}")
    print("2024 frozen benchmark:", f"{len(rows24):,}")
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

    # Frozen current system. Its action/WHEN outputs are reused exactly.
    base_model, base_checkpoint = load_v680(
        features.shape[1],
        device,
    )

    print("Frozen V6.8 epoch:", base_checkpoint["epoch"])
    print("Frozen V6.8 policy:", base_checkpoint["policy"])

    if args.smoke:
        train_rows = train_rows[-min(256, len(train_rows)):]
        rows23 = rows23[:min(128, len(rows23))]
        rows24 = rows24[:min(128, len(rows24))]
        epochs = 1
        batch_size = min(16, args.batch)
        print("SMOKE MODE")
    else:
        epochs = args.epochs
        batch_size = args.batch

    # Compute current V6.8 predictions for 2023 once. They remain frozen
    # for every V6.9.1 epoch, saving GPU and isolating direction.
    base23 = v680.predict(
        base_model,
        rows23,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )

    source23 = arrays["source"][rows23]
    oracle23 = target_np["master_side"][source23]

    print()
    print("2023 EXACT-SYSTEM CEILING CHECK")
    print("-" * 130)

    baseline23 = system_eval(
        rows23,
        base23["side"],
        base23,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "V6.8 BASELINE",
    )

    oracle_system23 = system_eval(
        rows23,
        oracle23,
        base23,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "ORACLE SIDE CEILING",
    )

    print()
    print(
        "Recoverable WR gap on 2023:",
        f"{oracle_system23['win'] - baseline23['win']:+.2%}"
    )

    model = DirectionRecoveryBrainV691(
        features.shape[1],
        groups,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print("Direction parameters:", f"{params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    rng = np.random.default_rng(args.seed)

    best_selection = -np.inf
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        factor = lr_factor(epoch, epochs)

        for group in optimizer.param_groups:
            group["lr"] = LR * factor

        min_gap = GAP_CURRICULUM[
            min(epoch - 1, len(GAP_CURRICULUM) - 1)
        ]

        epoch_started = time.time()

        losses = train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
            target_np,
            path_np,
            mean_t,
            std_t,
            scaler,
            device,
            rng,
            batch_size,
            min_gap,
        )

        pred23 = predict_direction(
            model,
            rows23,
            arrays,
            mean_t,
            std_t,
            device,
            batch_size,
        )

        dm23 = direction_metrics(
            rows23,
            pred23,
            arrays,
            target_np,
            "2023 DEV",
        )

        print()
        print("2023 FROZEN V6.8 SYSTEM WITH NEW DIRECTION")
        print("-" * 130)

        system23 = system_eval(
            rows23,
            pred23["master"],
            base23,
            arrays,
            execution,
            daily,
            base_checkpoint["policy"],
            "V6.9.1 DIRECTION",
        )

        selection = selection_score(system23, dm23)

        row = {
            "epoch": epoch,
            "min_gap": min_gap,
            "lr": optimizer.param_groups[0]["lr"],
            "selection": selection,
            "system_win": system23["win"],
            "system_mean": system23["mean"],
            "system_pf": system23["pf"],
            "system_trades": system23["trades"],
            "seconds": time.time() - epoch_started,
            **{
                f"loss_{k}": value
                for k, value in losses.items()
            },
        }

        history.append(row)

        print()
        print(
            f"EPOCH {epoch}/{epochs} "
            f"gap>={min_gap:g} "
            f"WR={system23['win']:.2%} "
            f"PF={system23['pf']:.3f} "
            f"selection={selection:.6f} "
            f"sec={row['seconds']:.1f}"
        )

        pd.DataFrame(history).to_csv(
            OUT / "training_history_v691.csv",
            index=False,
        )

        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "version": VERSION,
                    "epoch": epoch,
                    "selection": selection,
                    "system_2023": system23,
                    "model": model.state_dict(),
                    "feature_names": names,
                    "groups": groups,
                    "mean": mean,
                    "std": std,
                    "seed": args.seed,
                },
                CHAMPION,
            )

            dm23.to_csv(
                OUT / "champion_2023_direction.csv",
                index=False,
            )

            print("*** NEW DIRECTION CHAMPION ***")
        else:
            stale += 1
            print(f"No improvement stale={stale}/{PATIENCE}")

        if not args.smoke and epoch >= 12 and stale >= PATIENCE:
            print("Early stopping.")
            break

    if not CHAMPION.exists():
        raise RuntimeError("No V6.9.1 champion saved.")

    champion = torch.load(
        CHAMPION,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(champion["model"])

    print()
    print("=" * 130)
    print("FROZEN V6.9.1 DIRECTION CHAMPION")
    print("Epoch:", champion["epoch"])
    print("Selection:", champion["selection"])

    # Only now open the already-approved 2024 research benchmark.
    base24 = v680.predict(
        base_model,
        rows24,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )

    pred24 = predict_direction(
        model,
        rows24,
        arrays,
        mean_t,
        std_t,
        device,
        batch_size,
    )

    dm24 = direction_metrics(
        rows24,
        pred24,
        arrays,
        target_np,
        "2024 FROZEN",
    )

    source24 = arrays["source"][rows24]
    oracle24 = target_np["master_side"][source24]

    print()
    print("2024 EXACT SAME V6.8 SYSTEM — SIDE ABLATION")
    print("=" * 130)

    baseline24 = system_eval(
        rows24,
        base24["side"],
        base24,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "V6.8 BASELINE",
    )

    recovered24 = system_eval(
        rows24,
        pred24["master"],
        base24,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "V6.9.1 DIRECTION",
    )

    oracle24_system = system_eval(
        rows24,
        oracle24,
        base24,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "ORACLE SIDE CEILING",
    )

    dm24.to_csv(
        OUT / "frozen_2024_direction.csv",
        index=False,
    )

    pd.DataFrame(
        [baseline24, recovered24, oracle24_system]
    ).to_csv(
        OUT / "frozen_2024_system_comparison.csv",
        index=False,
    )

    summary = {
        "version": VERSION,
        "champion_epoch": int(champion["epoch"]),
        "selection_2023": float(champion["selection"]),
        "parameters": int(params),
        "baseline_2024": baseline24,
        "direction_recovered_2024": recovered24,
        "oracle_side_ceiling_2024": oracle24_system,
        "target_80_reached": bool(
            recovered24["win"] >= 0.80
            and recovered24["mean"] > 0.0
            and recovered24["pf"] >= 1.20
        ),
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v691.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(json.dumps(summary, indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
