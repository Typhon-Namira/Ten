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


OUT = Path(
    "training/artifacts/v6/"
    "direction_learnability_v692b"
)

# Explicit horizons in M5 bars. The deepest lag is about four trading
# days of bars. We use row lags intentionally, matching the existing
# regime branch semantics, while the near lags are protected by V6.8's
# recent-continuity filter.
LAGS = (0, 1, 3, 6, 12, 23, 48, 96, 288, 576, 1152)

DEFAULT_TRAIN_SAMPLE = 120000
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 512
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 20260824


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


def make_explicit_delta_features(
    rows,
    arrays,
    mean,
    std,
    device,
):
    source = arrays["source"][rows]
    idx = source[:, None] - np.asarray(LAGS, dtype=np.int64)[None, :]

    x = np.asarray(
        arrays["features"][idx],
        dtype=np.float32,
    )

    x = (x - mean[None, None, :]) / std[None, None, :]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    latest = x[:, 0, :]
    deltas = latest[:, None, :] - x[:, 1:, :]

    # Latest state + explicitly signed movement from each historical lag.
    z = np.concatenate(
        [latest[:, None, :], deltas],
        axis=1,
    ).reshape(len(rows), -1)

    return torch.from_numpy(
        np.ascontiguousarray(z, dtype=np.float32)
    ).to(device, non_blocking=True), source


class ExplicitDeltaProbe(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        input_dim = feature_dim * len(LAGS)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.10),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.08),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.LayerNorm(192),
        )

        self.horizon_head = nn.Linear(192, 3)
        self.all9_head = nn.Linear(192, 1)

    def forward(self, x):
        h = self.encoder(x)
        return {
            "horizon": self.horizon_head(h),
            "all9": self.all9_head(h).squeeze(-1),
        }


def weighted_bce(logit, y, valid, strength):
    if not valid.any():
        return logit.float().sum() * 0.0

    raw = F.binary_cross_entropy_with_logits(
        logit.float()[valid],
        y.float()[valid],
        reduction="none",
    )
    weight = torch.clamp(
        1.0 + torch.log1p(strength.float()[valid] / 3.0),
        1.0,
        4.0,
    )
    return (raw * weight).sum() / weight.sum()


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
        x, source = make_explicit_delta_features(
            br, arrays, mean, std, device
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

            loss = 0.30 * h30 + 0.65 * h60 + 2.50 * h120 + 1.00 * all9

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
    mean,
    std,
    device,
    batch,
):
    model.eval()
    h = []
    a = []

    for start in range(0, len(rows), batch):
        br = rows[start:start + batch]
        x, _ = make_explicit_delta_features(
            br, arrays, mean, std, device
        )
        out = model(x)
        h.append(torch.sigmoid(out["horizon"]).cpu().numpy())
        a.append(torch.sigmoid(out["all9"]).cpu().numpy())

    return {
        "horizon": np.concatenate(h, axis=0).astype(np.float32),
        "all9": np.concatenate(a, axis=0).astype(np.float32),
    }


def metrics(rows, pred, arrays, stable):
    source = arrays["source"][rows]
    result = {}

    print()
    print("2023 EXPLICIT-DELTA LEARNABILITY")
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

    print("TEN V6.9.2B DIRECTION LEARNABILITY AUDIT")
    print("=" * 132)
    print("Question: do the existing 523 features contain learnable direction")
    print("when temporal deltas are exposed explicitly instead of compressed by GRUs?")
    print("2016-2022 train -> 2023 dev only. 2024/2025/2026 stay closed.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    arrays, split, groups, names, mean, std = brain.load_data()
    execution = v671.load_execution_targets()
    daily = v672.load_daily_targets()
    stable = v692.build_stable_targets(execution)

    train_rows = v680.filter_rows(split["train"], arrays, execution)
    val = split["val"]
    years = execution["year"][arrays["source"]]
    rows23 = v680.filter_rows(
        val[years[val] == 2023], arrays, execution
    )

    # Deepest explicit lag must be available.
    train_rows = train_rows[
        arrays["source"][train_rows] >= max(LAGS)
    ]
    rows23 = rows23[
        arrays["source"][rows23] >= max(LAGS)
    ]

    # Train only where the primary H120 stable target exists.
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

    print("Features:", len(names))
    print("Explicit input dim:", len(names) * len(LAGS))
    print("Lags:", LAGS)
    print("Train sample:", f"{len(train_rows):,}")
    print("2023 dev:", f"{len(rows23):,}")
    print("2024: NOT EVALUATED")
    print("2025: LOCKED")
    print("2026: LOCKED")

    model = ExplicitDeltaProbe(len(names)).to(device)
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

    # Frozen V6.8 outputs are computed once so every probe epoch is judged
    # by the actual downstream 2023 system, not just AUC.
    mean_t = torch.from_numpy(np.asarray(mean, np.float32)).view(1, 1, -1).to(device)
    std_t = torch.from_numpy(np.asarray(std, np.float32)).view(1, 1, -1).to(device)
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
    best_auc = -np.inf

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
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
            mean,
            std,
            device,
            args.batch,
        )
        m = metrics(rows23, pred, arrays, stable)

        print()
        print("2023 FROZEN V6.8 WITH EXPLICIT-DELTA H120")
        print("-" * 132)
        system = v691.system_eval(
            rows23,
            pred["horizon"][:, 2],
            base23,
            arrays,
            execution,
            daily,
            base_checkpoint["policy"],
            "V6.9.2B EXPLICIT DELTA H120",
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
            OUT / "history_v692b.csv",
            index=False,
        )

        best_auc = max(best_auc, m["H120_auc"])
        print(
            f"EPOCH {epoch}/{args.epochs} loss={loss:.5f} "
            f"H120_AUC={m['H120_auc']:.4f} "
            f"WR={system['win']:.2%} PF={system['pf']:.3f} sec={sec:.1f}"
        )

    verdict = (
        "LEARNABLE_REPRESENTATION"
        if best_auc >= 0.56
        else "NO_CLEAR_SIGNAL_IN_EXISTING_FEATURES"
    )

    summary = {
        "version": "v6.9.2b",
        "best_h120_auc_2023": float(best_auc),
        "verdict": verdict,
        "train_sample": int(len(train_rows)),
        "2024_evaluated": False,
        "2025_evaluated": False,
        "2026_evaluated": False,
        "seconds": float(time.time() - started),
    }

    with open(OUT / "summary_v692b.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 132)
    print("VERDICT:", verdict)
    print("Best 2023 H120 AUC:", f"{best_auc:.4f}")
    if best_auc >= 0.56:
        print("GO: explicit temporal differences expose usable directional signal.")
        print("Next: build the production direction encoder around this representation.")
    else:
        print("STOP: do not make the neural net larger yet.")
        print("Next: add genuinely new causal inputs/context rather than more capacity.")
    print("2024 was not opened by this audit.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
