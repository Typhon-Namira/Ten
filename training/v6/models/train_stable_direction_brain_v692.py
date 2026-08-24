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
import training.v6.models.train_direction_recovery_v691 as v691


VERSION = "v6.9.2"

OUT = Path(
    "training/artifacts/v6/"
    "stable_direction_brain_v692"
)
CHAMPION = OUT / "champion_v692.pt"

DEFAULT_EPOCHS = 22
DEFAULT_BATCH = 160
PATIENCE = 6

LR = 2.0e-4
MIN_LR = 1.0e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NET_SCALE = 30.0

SEED = 20260824

# Curriculum is applied to the economic strength of the H120 majority
# target, not to the old max-of-max side gap.
STRENGTH_CURRICULUM = (
    20.0, 20.0,
    15.0, 15.0,
    10.0, 10.0,
    7.0, 7.0,
    5.0, 5.0,
    3.0, 3.0,
    2.0, 2.0,
    1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

CONSENSUS_NAMES = ("H30", "H60", "H120")


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


def build_stable_targets(execution):
    valid = execution["valid"]
    gross = execution["gross"].astype(np.float32)
    net = gross - 0.5

    n = len(gross)
    pair_side = np.zeros((n, 9), dtype=np.float32)
    pair_gap = np.zeros((n, 9), dtype=np.float32)
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
        pair_side[ok, j] = (gap[ok] > 0.0).astype(np.float32)

    def consensus(ids):
        ids = np.asarray(ids, dtype=np.int64)
        y = pair_side[:, ids]
        v = pair_valid[:, ids]
        g = pair_gap[:, ids]

        votes = (y * v.astype(np.float32)).sum(axis=1)
        count = v.sum(axis=1)

        usable = (count > 0) & (votes * 2 != count)
        label = np.zeros(n, dtype=np.float32)
        label[usable] = (
            votes[usable] * 2 > count[usable]
        ).astype(np.float32)

        # Vote margin captures agreement among the fixed tasks.
        vote_margin = np.zeros(n, dtype=np.float32)
        vote_margin[usable] = (
            np.abs(2.0 * votes[usable] - count[usable])
            / count[usable]
        ).astype(np.float32)

        # Economic strength is median absolute pair gap among valid tasks.
        abs_gap = np.where(v, np.abs(g), np.nan)
        with np.errstate(all="ignore"):
            median_gap = np.nanmedian(abs_gap, axis=1)
        median_gap = np.nan_to_num(
            median_gap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        strength = (
            median_gap * (0.5 + 0.5 * vote_margin)
        ).astype(np.float32)

        return label, usable, strength, vote_margin

    h30 = consensus([0, 1, 2])
    h60 = consensus([3, 4, 5])
    h120 = consensus([6, 7, 8])
    all9 = consensus(list(range(9)))

    consensus_side = np.stack(
        [h30[0], h60[0], h120[0]],
        axis=1,
    ).astype(np.float32)
    consensus_valid = np.stack(
        [h30[1], h60[1], h120[1]],
        axis=1,
    ).astype(bool)
    consensus_strength = np.stack(
        [h30[2], h60[2], h120[2]],
        axis=1,
    ).astype(np.float32)

    safe_net = np.where(valid, net, 0.0).astype(np.float32)

    return {
        "pair_side": pair_side,
        "pair_gap": pair_gap,
        "pair_valid": pair_valid,
        "consensus_side": consensus_side,
        "consensus_valid": consensus_valid,
        "consensus_strength": consensus_strength,
        "all9_side": all9[0],
        "all9_valid": all9[1],
        "all9_strength": all9[2],
        "action_net": safe_net,
        "action_valid": valid,
    }


class StableDirectionBrainV692(
    v690.TechnicalDirectionFoundationV690
):
    def __init__(self, feature_dim, groups):
        super().__init__(feature_dim, groups)

        # The inherited side_head is repurposed as H30/H60/H120
        # stable majority direction.
        self.all9_head = nn.Sequential(
            nn.Linear(v690.STATE_DIM, 160),
            nn.GELU(),
            nn.LayerNorm(160),
            nn.Dropout(0.10),
            nn.Linear(160, 1),
        )

        self.pair_head = nn.Sequential(
            nn.Linear(v690.STATE_DIM, 192),
            nn.GELU(),
            nn.Linear(192, 9),
        )

        self.action_net_head = nn.Sequential(
            nn.Linear(v690.STATE_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 18),
        )

    def forward(self, recent, intraday, regime):
        out = super().forward(recent, intraday, regime)
        state = out["state"]

        out["consensus_logits"] = out.pop("side_logits")
        out["all9_logit"] = self.all9_head(state).squeeze(-1)
        out["pair_logits"] = self.pair_head(state)
        out["action_net_norm"] = self.action_net_head(state)

        return out


def weighted_bce(logits, y, valid, strength, min_strength):
    mask = valid & (strength >= min_strength)
    if not mask.any():
        return logits.float().sum() * 0.0

    raw = F.binary_cross_entropy_with_logits(
        logits.float()[mask],
        y.float()[mask],
        reduction="none",
    )

    # Stable weighting: strong economic separation matters more, but
    # the cap prevents a few extreme bars from dominating training.
    weight = torch.clamp(
        1.0 + torch.log1p(strength.float()[mask] / 3.0),
        1.0,
        5.0,
    )

    return (raw * weight).sum() / weight.sum()


def make_targets(source, stable, device):
    def t(x, dtype=torch.float32):
        return torch.from_numpy(x).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    return {
        "consensus_side": t(stable["consensus_side"][source]),
        "consensus_valid": t(
            stable["consensus_valid"][source],
            torch.bool,
        ),
        "consensus_strength": t(
            stable["consensus_strength"][source]
        ),
        "all9_side": t(stable["all9_side"][source]),
        "all9_valid": t(stable["all9_valid"][source], torch.bool),
        "all9_strength": t(stable["all9_strength"][source]),
        "pair_side": t(stable["pair_side"][source]),
        "pair_valid": t(stable["pair_valid"][source], torch.bool),
        "pair_gap": t(stable["pair_gap"][source]),
        "action_net": t(
            np.clip(
                stable["action_net"][source] / NET_SCALE,
                -3.0,
                3.0,
            ).astype(np.float32)
        ),
        "action_valid": t(
            stable["action_valid"][source],
            torch.bool,
        ),
    }


def compute_loss(out, target, min_strength):
    h_losses = []
    h_weights = (0.45, 0.90, 3.50)

    for hi in range(3):
        h_losses.append(
            weighted_bce(
                out["consensus_logits"][:, hi],
                target["consensus_side"][:, hi],
                target["consensus_valid"][:, hi],
                target["consensus_strength"][:, hi],
                min_strength,
            )
        )

    h30, h60, h120 = h_losses

    all9 = weighted_bce(
        out["all9_logit"],
        target["all9_side"],
        target["all9_valid"],
        target["all9_strength"],
        max(min_strength * 0.75, 0.0),
    )

    pair_losses = []
    for j in range(9):
        pair_strength = torch.abs(target["pair_gap"][:, j])
        pair_losses.append(
            weighted_bce(
                out["pair_logits"][:, j],
                target["pair_side"][:, j],
                target["pair_valid"][:, j],
                pair_strength,
                max(min_strength * 0.50, 0.0),
            )
        )
    pair = torch.stack(pair_losses).mean()

    action_valid = target["action_valid"]
    action_net = F.smooth_l1_loss(
        out["action_net_norm"].float()[action_valid],
        target["action_net"].float()[action_valid],
        beta=0.15,
    )

    # Pair margin regression ties classification to actual economics.
    pred_pair_gap = (
        out["action_net_norm"][:, 9:]
        - out["action_net_norm"][:, :9]
    )
    true_pair_gap = torch.clamp(
        target["pair_gap"].float() / NET_SCALE,
        -3.0,
        3.0,
    )
    pair_margin = F.smooth_l1_loss(
        pred_pair_gap[target["pair_valid"]],
        true_pair_gap[target["pair_valid"]],
        beta=0.10,
    )

    # Technical specialist heads learn the stable horizon labels.
    expert_losses = []
    for logits in out["expert_side_logits"].values():
        for hi in range(3):
            expert_losses.append(
                weighted_bce(
                    logits[:, hi],
                    target["consensus_side"][:, hi],
                    target["consensus_valid"][:, hi],
                    target["consensus_strength"][:, hi],
                    max(min_strength * 0.75, 0.0),
                )
            )
    experts = torch.stack(expert_losses).mean()

    # Primary objective is explicitly H120 majority because it was the
    # most stable high-value target across 2023 and 2024. ALL9 provides
    # a broad consensus regularizer; H60/H30 preserve shorter-horizon
    # structure without dominating the representation.
    total = (
        h_weights[0] * h30
        + h_weights[1] * h60
        + h_weights[2] * h120
        + 1.50 * all9
        + 0.75 * pair
        + 0.55 * pair_margin
        + 0.40 * action_net
        + 0.20 * experts
    )

    return total, {
        "total": total,
        "h30": h30,
        "h60": h60,
        "h120": h120,
        "all9": all9,
        "pair": pair,
        "pair_margin": pair_margin,
        "action_net": action_net,
        "experts": experts,
    }


def lr_factor(epoch, epochs):
    warmup = min(2, epochs)
    if epoch <= warmup:
        return epoch / max(warmup, 1)

    progress = (epoch - warmup) / max(epochs - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = MIN_LR / LR
    return floor + (1.0 - floor) * cosine


def train_epoch(
    model,
    optimizer,
    rows,
    arrays,
    stable,
    mean_t,
    std_t,
    scaler,
    device,
    rng,
    batch_size,
    min_strength,
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

        target = make_targets(source, stable, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            out = model(recent, intraday, regime)
            loss, parts = compute_loss(out, target, min_strength)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        for key, value in parts.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        batches += 1

    return {
        key: value / max(batches, 1)
        for key, value in sums.items()
    }


@torch.no_grad()
def predict(model, rows, arrays, mean_t, std_t, device, batch_size):
    model.eval()
    consensus = []
    all9 = []

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

        consensus.append(
            torch.sigmoid(out["consensus_logits"])
            .cpu().numpy().astype(np.float32)
        )
        all9.append(
            torch.sigmoid(out["all9_logit"])
            .cpu().numpy().astype(np.float32)
        )

    return {
        "consensus": np.concatenate(consensus, axis=0),
        "all9": np.concatenate(all9, axis=0),
    }


def label_metrics(rows, pred, arrays, stable, label):
    source = arrays["source"][rows]
    records = []

    print()
    print(f"{label} STABLE DIRECTION")
    print("-" * 132)

    for hi, name in enumerate(CONSENSUS_NAMES):
        valid = stable["consensus_valid"][source, hi]
        y = stable["consensus_side"][source, hi][valid].astype(np.uint8)
        p = pred["consensus"][valid, hi]
        acc = float(((p >= 0.5) == y).mean())
        auc = safe_auc(y, p)
        print(
            f"{name:<6} N={valid.sum():>7,} "
            f"ACC={acc:>7.2%} AUC={auc:>7.4f}"
        )
        records.append({
            "target": name,
            "n": int(valid.sum()),
            "acc": acc,
            "auc": auc,
        })

    valid = stable["all9_valid"][source]
    y = stable["all9_side"][source][valid].astype(np.uint8)
    p = pred["all9"][valid]
    acc = float(((p >= 0.5) == y).mean())
    auc = safe_auc(y, p)
    print(
        f"ALL9   N={valid.sum():>7,} "
        f"ACC={acc:>7.2%} AUC={auc:>7.4f}"
    )
    records.append({
        "target": "ALL9",
        "n": int(valid.sum()),
        "acc": acc,
        "auc": auc,
    })

    return pd.DataFrame(records)


def system_eval(
    rows,
    side_prob,
    base_pred,
    arrays,
    execution,
    daily,
    policy,
    label,
):
    return v691.system_eval(
        rows,
        side_prob,
        base_pred,
        arrays,
        execution,
        daily,
        policy,
        label,
    )


def stable_oracle_prob(rows, arrays, stable, key):
    source = arrays["source"][rows]

    if key == "H120":
        y = stable["consensus_side"][source, 2]
        valid = stable["consensus_valid"][source, 2]
    elif key == "ALL9":
        y = stable["all9_side"][source]
        valid = stable["all9_valid"][source]
    else:
        raise ValueError(key)

    # Rare unavailable rows fall back to V6.8 at evaluation call site.
    return y.astype(np.float32), valid


def merge_oracle_with_base(y, valid, base_side_prob):
    out = np.asarray(base_side_prob, dtype=np.float32).copy()
    out[valid] = y[valid]
    return out


def selection_score(system, metrics):
    h120 = metrics[metrics["target"] == "H120"].iloc[0]
    all9 = metrics[metrics["target"] == "ALL9"].iloc[0]

    if system["trades"] < 150 or system["coverage"] < 0.70:
        return -1e9

    return float(
        5.0 * system["win"]
        + 0.55 * float(h120["auc"])
        + 0.20 * float(all9["auc"])
        + 0.20 * np.tanh(system["mean"] / 5.0)
        + 0.12 * np.log(max(system["pf"], 1e-6))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    started = time.time()
    seed_all(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    print("TEN V6.9.2 STABLE DIRECTION BRAIN")
    print("=" * 132)
    print("Primary target: H120 majority across 3 fixed action pairs")
    print("Auxiliary: ALL9 majority, H60/H30, pair directions, action values")

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
    stable = build_stable_targets(execution)
    features = arrays["features"]

    years = execution["year"][arrays["source"]]
    train_rows = v680.filter_rows(
        split["train"], arrays, execution
    )
    val = split["val"]
    rows23 = v680.filter_rows(
        val[years[val] == 2023], arrays, execution
    )
    rows24 = v680.filter_rows(
        val[years[val] == 2024], arrays, execution
    )

    print("Features:", features.shape[1])
    print("Train:", f"{len(train_rows):,}")
    print("2023 dev:", f"{len(rows23):,}")
    print("2024 benchmark:", f"{len(rows24):,}")
    print("2025: LOCKED / NOT EVALUATED")
    print("2026: LOCKED / NOT EVALUATED")

    mean_t = torch.from_numpy(
        np.asarray(mean, dtype=np.float32)
    ).view(1, 1, -1).to(device)
    std_np = np.asarray(std, dtype=np.float32).copy()
    std_np[std_np < 1e-6] = 1.0
    std_t = torch.from_numpy(std_np).view(1, 1, -1).to(device)

    base_model, base_checkpoint = v691.load_v680(
        features.shape[1], device
    )
    print("Frozen V6.8 epoch:", base_checkpoint["epoch"])
    print("Frozen V6.8 policy:", base_checkpoint["policy"])

    base23 = v680.predict(
        base_model,
        rows23,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )

    print()
    print("2023 STABLE TARGET CEILINGS")
    print("=" * 132)
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

    h120_y, h120_valid = stable_oracle_prob(
        rows23, arrays, stable, "H120"
    )
    h120_oracle = merge_oracle_with_base(
        h120_y, h120_valid, base23["side"]
    )
    system_eval(
        rows23,
        h120_oracle,
        base23,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "H120 MAJORITY ORACLE",
    )

    all9_y, all9_valid = stable_oracle_prob(
        rows23, arrays, stable, "ALL9"
    )
    all9_oracle = merge_oracle_with_base(
        all9_y, all9_valid, base23["side"]
    )
    system_eval(
        rows23,
        all9_oracle,
        base23,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "ALL9 MAJORITY ORACLE",
    )

    epochs = args.epochs
    batch_size = args.batch

    if args.probe:
        rng_probe = np.random.default_rng(args.seed + 99)
        n_probe = min(80000, len(train_rows))
        train_rows = np.sort(
            rng_probe.choice(
                train_rows,
                size=n_probe,
                replace=False,
            )
        )
        epochs = min(4, epochs)
        print()
        print("PROBE MODE")
        print("Train subset:", f"{len(train_rows):,}")
        print("Epochs:", epochs)
        print("2024 WILL NOT BE OPENED IN PROBE MODE")

    model = StableDirectionBrainV692(
        features.shape[1], groups
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("Parameters:", f"{params:,}")

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

        min_strength = STRENGTH_CURRICULUM[
            min(epoch - 1, len(STRENGTH_CURRICULUM) - 1)
        ]
        epoch_started = time.time()

        losses = train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
            stable,
            mean_t,
            std_t,
            scaler,
            device,
            rng,
            batch_size,
            min_strength,
        )

        pred23 = predict(
            model,
            rows23,
            arrays,
            mean_t,
            std_t,
            device,
            batch_size,
        )
        metrics23 = label_metrics(
            rows23,
            pred23,
            arrays,
            stable,
            "2023 DEV",
        )

        print()
        print("2023 FROZEN V6.8 WITH LEARNED H120 DIRECTION")
        print("-" * 132)
        system23 = system_eval(
            rows23,
            pred23["consensus"][:, 2],
            base23,
            arrays,
            execution,
            daily,
            base_checkpoint["policy"],
            "V6.9.2 H120",
        )

        selection = selection_score(system23, metrics23)
        row = {
            "epoch": epoch,
            "min_strength": min_strength,
            "lr": optimizer.param_groups[0]["lr"],
            "selection": selection,
            "system_win": system23["win"],
            "system_mean": system23["mean"],
            "system_pf": system23["pf"],
            "system_trades": system23["trades"],
            "seconds": time.time() - epoch_started,
            **{f"loss_{k}": v for k, v in losses.items()},
        }
        history.append(row)

        print()
        print(
            f"EPOCH {epoch}/{epochs} "
            f"strength>={min_strength:g} "
            f"WR={system23['win']:.2%} "
            f"PF={system23['pf']:.3f} "
            f"selection={selection:.6f} "
            f"sec={row['seconds']:.1f}"
        )

        pd.DataFrame(history).to_csv(
            OUT / (
                "probe_history_v692.csv"
                if args.probe
                else "training_history_v692.csv"
            ),
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
                    "probe": bool(args.probe),
                },
                CHAMPION,
            )
            metrics23.to_csv(
                OUT / "champion_2023_stable_direction.csv",
                index=False,
            )
            print("*** NEW V6.9.2 CHAMPION ***")
        else:
            stale += 1
            print(f"No improvement stale={stale}/{PATIENCE}")

        if not args.probe and epoch >= 10 and stale >= PATIENCE:
            print("Early stopping.")
            break

    champion = torch.load(
        CHAMPION,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(champion["model"])

    if args.probe:
        summary = {
            "version": VERSION,
            "probe": True,
            "champion_epoch": int(champion["epoch"]),
            "selection_2023": float(champion["selection"]),
            "system_2023": champion["system_2023"],
            "2024_evaluated": False,
            "2025_evaluated": False,
            "2026_evaluated": False,
            "seconds": float(time.time() - started),
        }
        with open(OUT / "probe_summary_v692.json", "w") as f:
            json.dump(summary, f, indent=2)
        print()
        print(json.dumps(summary, indent=2))
        print("PROBE COMPLETE — 2024 NOT OPENED")
        return

    print()
    print("=" * 132)
    print("FROZEN V6.9.2 CHAMPION")
    print("Epoch:", champion["epoch"])
    print("Selection:", champion["selection"])

    base24 = v680.predict(
        base_model,
        rows24,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )
    pred24 = predict(
        model,
        rows24,
        arrays,
        mean_t,
        std_t,
        device,
        batch_size,
    )
    metrics24 = label_metrics(
        rows24,
        pred24,
        arrays,
        stable,
        "2024 FROZEN",
    )

    print()
    print("2024 EXACT SAME V6.8 SYSTEM — DIRECTION ABLATION")
    print("=" * 132)

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
    learned24 = system_eval(
        rows24,
        pred24["consensus"][:, 2],
        base24,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "V6.9.2 H120",
    )

    h120_y24, h120_valid24 = stable_oracle_prob(
        rows24, arrays, stable, "H120"
    )
    h120_oracle24 = merge_oracle_with_base(
        h120_y24, h120_valid24, base24["side"]
    )
    h120_ceiling24 = system_eval(
        rows24,
        h120_oracle24,
        base24,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "H120 MAJORITY ORACLE",
    )

    all9_y24, all9_valid24 = stable_oracle_prob(
        rows24, arrays, stable, "ALL9"
    )
    all9_oracle24 = merge_oracle_with_base(
        all9_y24, all9_valid24, base24["side"]
    )
    all9_ceiling24 = system_eval(
        rows24,
        all9_oracle24,
        base24,
        arrays,
        execution,
        daily,
        base_checkpoint["policy"],
        "ALL9 MAJORITY ORACLE",
    )

    metrics24.to_csv(
        OUT / "frozen_2024_stable_direction.csv",
        index=False,
    )
    pd.DataFrame(
        [baseline24, learned24, h120_ceiling24, all9_ceiling24]
    ).to_csv(
        OUT / "frozen_2024_system_comparison.csv",
        index=False,
    )

    summary = {
        "version": VERSION,
        "probe": False,
        "champion_epoch": int(champion["epoch"]),
        "selection_2023": float(champion["selection"]),
        "parameters": int(params),
        "baseline_2024": baseline24,
        "learned_h120_2024": learned24,
        "h120_oracle_2024": h120_ceiling24,
        "all9_oracle_2024": all9_ceiling24,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seed": int(args.seed),
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v692.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(json.dumps(summary, indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
