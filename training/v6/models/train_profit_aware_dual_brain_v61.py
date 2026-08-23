from pathlib import Path
import json
import hashlib
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

import training.v6.models.train_dual_brain_v60 as v60


VALUE_FILE = Path(
    "training/v6/data_lake/action_value_v61/"
    "action_value_targets_v61.parquet"
)

RACE_FILE = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

TECH_DIR = Path(
    "training/v6/data_lake/technical_state_v60"
)

V60_ART = Path(
    "training/artifacts/v6/dual_brain_v60"
)

OUT = Path(
    "training/artifacts/v6/"
    "profit_aware_dual_brain_v61"
)

BATCH = 768
EPOCHS = 10
PATIENCE = 3
SEED = 610

VALUE_SCALE = 15.0

COVERAGES = (
    0.05,
    0.02,
    0.01,
    0.005,
)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )


def value_transform(x):
    return np.arcsinh(
        x / VALUE_SCALE
    ).astype(
        np.float32
    )


def value_inverse(x):
    x = np.clip(
        x,
        -3.5,
        3.5,
    )

    return (
        np.sinh(x)
        * VALUE_SCALE
    )


class ProfitDataset(Dataset):
    def __init__(
        self,
        rows,
        m1,
        tech,
        source,
        m1_end,
        race_y,
        race_mask,
        values,
        profitable,
        action,
        m1_mean,
        m1_std,
        tech_mean,
        tech_std,
    ):
        self.rows = rows
        self.m1 = m1
        self.tech = tech
        self.source = source
        self.m1_end = m1_end

        self.race_y = race_y
        self.race_mask = race_mask

        self.values = values
        self.profitable = profitable
        self.action = action

        self.m1_mean = m1_mean
        self.m1_std = m1_std

        self.tech_mean = tech_mean
        self.tech_std = tech_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(
            self.rows[i]
        )

        end = int(
            self.m1_end[r]
        )

        seq = np.asarray(
            self.m1[
                end - v60.SEQ + 1:
                end + 1
            ],
            dtype=np.float32,
        )

        seq = np.nan_to_num(
            seq,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        seq = (
            seq - self.m1_mean
        ) / self.m1_std

        t = np.asarray(
            self.tech[
                self.source[r]
            ],
            dtype=np.float32,
        )

        t = np.nan_to_num(
            t,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        t = (
            t - self.tech_mean
        ) / self.tech_std

        return (
            torch.from_numpy(seq),
            torch.from_numpy(t),
            torch.from_numpy(
                self.race_y[r]
            ),
            torch.from_numpy(
                self.race_mask[r]
            ),
            torch.from_numpy(
                self.values[r]
            ),
            torch.from_numpy(
                self.profitable[r]
            ),
            torch.tensor(
                self.action[r],
                dtype=torch.long,
            ),
        )


class ProfitAwareDualBrain(nn.Module):
    def __init__(self, n_tech):
        super().__init__()

        self.base = v60.DualBrain(
            n_tech=n_tech
        )

        # Continuous EV heads.
        self.hist_value = nn.Linear(
            128,
            2,
        )

        self.tech_value = nn.Linear(
            128,
            2,
        )

        self.fusion_value = nn.Linear(
            128,
            2,
        )

        # P(action profitable) heads.
        self.hist_profit = nn.Linear(
            128,
            2,
        )

        self.tech_profit = nn.Linear(
            128,
            2,
        )

        self.fusion_profit = nn.Linear(
            128,
            2,
        )

        # Explicit:
        # 0 NO TRADE
        # 1 LONG
        # 2 SHORT
        self.action_head = nn.Linear(
            128,
            3,
        )

    def forward(
        self,
        sequence,
        technical,
    ):
        out = self.base(
            sequence,
            technical,
        )

        h = out[
            "embedding_h"
        ]

        t = out[
            "embedding_t"
        ]

        f = out[
            "embedding_f"
        ]

        return {
            "race_h":
                out["historical"],

            "race_t":
                out["technical"],

            "race_f":
                out["fusion"],

            "value_h":
                self.hist_value(h),

            "value_t":
                self.tech_value(t),

            "value_f":
                self.fusion_value(f),

            "profit_h":
                self.hist_profit(h),

            "profit_t":
                self.tech_profit(t),

            "profit_f":
                self.fusion_profit(f),

            "action":
                self.action_head(f),

            "embedding_h":
                h,

            "embedding_t":
                t,

            "embedding_f":
                f,
        }


def make_loader(
    rows,
    shuffle,
    arrays,
    norms,
):
    ds = ProfitDataset(
        rows=rows,
        m1=arrays["m1"],
        tech=arrays["tech"],
        source=arrays["source"],
        m1_end=arrays["m1_end"],
        race_y=arrays["race_y"],
        race_mask=arrays[
            "race_mask"
        ],
        values=arrays["values"],
        profitable=arrays[
            "profitable"
        ],
        action=arrays["action"],
        m1_mean=norms["m1_mean"],
        m1_std=norms["m1_std"],
        tech_mean=norms["tech_mean"],
        tech_std=norms["tech_std"],
    )

    return DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    race_pos_weight,
):
    model.train()

    total = 0.0
    seen = 0

    amp = (
        device.type == "cuda"
    )

    race_head_weight = torch.tensor(
        [
            0.25,
            0.25,
            1.0,
            1.0,
            0.25,
            0.25,
        ],
        dtype=torch.float32,
        device=device,
    )

    for (
        seq,
        tech,
        race_y,
        race_mask,
        values,
        profitable,
        action,
    ) in loader:

        seq = seq.to(
            device,
            non_blocking=True,
        )

        tech = tech.to(
            device,
            non_blocking=True,
        )

        race_y = race_y.to(
            device,
            non_blocking=True,
        )

        race_mask = race_mask.to(
            device,
            non_blocking=True,
        )

        values = values.to(
            device,
            non_blocking=True,
        )

        profitable = profitable.to(
            device,
            non_blocking=True,
        )

        action = action.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            out = model(
                seq,
                tech,
            )

            # --------------------------------
            # VALUE LEARNING — PRIMARY
            # --------------------------------

            value_f = (
                F.smooth_l1_loss(
                    out["value_f"],
                    values,
                )
            )

            value_h = (
                F.smooth_l1_loss(
                    out["value_h"],
                    values,
                )
            )

            value_t = (
                F.smooth_l1_loss(
                    out["value_t"],
                    values,
                )
            )

            # --------------------------------
            # PROFITABILITY
            # --------------------------------

            profit_f = (
                F.binary_cross_entropy_with_logits(
                    out["profit_f"],
                    profitable,
                )
            )

            profit_h = (
                F.binary_cross_entropy_with_logits(
                    out["profit_h"],
                    profitable,
                )
            )

            profit_t = (
                F.binary_cross_entropy_with_logits(
                    out["profit_t"],
                    profitable,
                )
            )

            # --------------------------------
            # EXPLICIT ACTION COMPETITION
            # --------------------------------

            action_loss = (
                F.cross_entropy(
                    out["action"],
                    action,
                )
            )

            # --------------------------------
            # PRESERVE V6.0 FIRST-PASSAGE SKILL
            # --------------------------------

            race_f = v60.masked_bce(
                out["race_f"],
                race_y,
                race_mask,
                race_pos_weight,
                race_head_weight,
            )

            race_h = v60.masked_bce(
                out["race_h"],
                race_y,
                race_mask,
                race_pos_weight,
                race_head_weight,
            )

            race_t = v60.masked_bce(
                out["race_t"],
                race_y,
                race_mask,
                race_pos_weight,
                race_head_weight,
            )

            loss = (
                1.00 * value_f
                + 0.20 * value_h
                + 0.25 * value_t

                + 0.40 * profit_f
                + 0.08 * profit_h
                + 0.10 * profit_t

                + 0.25 * action_loss

                + 0.15 * race_f
                + 0.03 * race_h
                + 0.03 * race_t
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            2.0,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        b = len(seq)

        total += (
            loss.detach().float().item()
            * b
        )

        seen += b

    return total / seen


@torch.inference_mode()
def predict(
    model,
    loader,
    device,
):
    model.eval()

    value_f = []
    value_h = []
    value_t = []

    profit_f = []
    action_p = []

    true_value = []
    true_profit = []
    true_action = []

    for (
        seq,
        tech,
        race_y,
        race_mask,
        values,
        profitable,
        action,
    ) in loader:

        seq = seq.to(
            device,
            non_blocking=True,
        )

        tech = tech.to(
            device,
            non_blocking=True,
        )

        out = model(
            seq,
            tech,
        )

        value_f.append(
            out["value_f"]
            .float()
            .cpu()
            .numpy()
        )

        value_h.append(
            out["value_h"]
            .float()
            .cpu()
            .numpy()
        )

        value_t.append(
            out["value_t"]
            .float()
            .cpu()
            .numpy()
        )

        profit_f.append(
            torch.sigmoid(
                out["profit_f"]
            )
            .float()
            .cpu()
            .numpy()
        )

        action_p.append(
            torch.softmax(
                out["action"],
                dim=1,
            )
            .float()
            .cpu()
            .numpy()
        )

        true_value.append(
            values.numpy()
        )

        true_profit.append(
            profitable.numpy()
        )

        true_action.append(
            action.numpy()
        )

    return {
        "value_f":
            np.concatenate(
                value_f
            ),

        "value_h":
            np.concatenate(
                value_h
            ),

        "value_t":
            np.concatenate(
                value_t
            ),

        "profit_f":
            np.concatenate(
                profit_f
            ),

        "action_p":
            np.concatenate(
                action_p
            ),

        "true_value":
            np.concatenate(
                true_value
            ),

        "true_profit":
            np.concatenate(
                true_profit
            ),

        "true_action":
            np.concatenate(
                true_action
            ),
    }


def policy_metrics(
    name,
    pred,
):
    predicted_value = (
        value_inverse(
            pred["value_f"]
        )
    )

    actual_value = (
        value_inverse(
            pred["true_value"]
        )
    )

    p_profit = pred[
        "profit_f"
    ]

    # Side-specific utility score.
    #
    # Predicted EV remains primary.
    # Profit head acts only as confidence.
    side_score = (
        predicted_value
        + 5.0
        * (
            p_profit - 0.5
        )
    )

    predicted_long = (
        side_score[:, 0]
        >= side_score[:, 1]
    )

    selected_score = np.where(
        predicted_long,
        side_score[:, 0],
        side_score[:, 1],
    )

    selected_ev = np.where(
        predicted_long,
        predicted_value[:, 0],
        predicted_value[:, 1],
    )

    realized = np.where(
        predicted_long,
        actual_value[:, 0],
        actual_value[:, 1],
    )

    print()
    print(name)
    print("-" * 105)

    print(
        "Predicted positive-EV fraction:",
        f"{(selected_ev > 0).mean():.2%}",
    )

    print(
        "Overall selected-side mean:",
        f"{realized.mean():+.3f} bps",
    )

    results = {}

    for coverage in COVERAGES:
        n = max(
            1,
            int(
                round(
                    len(realized)
                    * coverage
                )
            ),
        )

        idx = np.argpartition(
            selected_score,
            -n,
        )[-n:]

        pnl = realized[
            idx
        ]

        long_share = (
            predicted_long[
                idx
            ].mean()
        )

        mean = pnl.mean()

        win = (
            pnl > 0
        ).mean()

        gain = pnl[
            pnl > 0
        ].sum()

        loss = -pnl[
            pnl < 0
        ].sum()

        pf = (
            gain / loss
            if loss > 0
            else np.inf
        )

        results[
            coverage
        ] = {
            "mean": mean,
            "win": win,
            "pf": pf,
            "long": long_share,
        }

        print(
            f"{coverage:>5.1%}"
            f" n={n:>5}"
            f" WIN={win:>6.2%}"
            f" mean={mean:>+7.3f}"
            f" PF={pf:>6.3f}"
            f" LONG={long_share:>6.2%}"
        )

    stable_score = (
        0.40
        * results[0.05]["mean"]

        + 0.35
        * results[0.02]["mean"]

        + 0.25
        * results[0.01]["mean"]
    )

    print()
    print(
        "STABLE UTILITY SCORE:",
        f"{stable_score:+.4f}",
    )

    return (
        stable_score,
        results,
    )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(
                1024 * 1024
            )

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def main():
    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.1 PROFIT-AWARE DUAL-BRAIN"
    )
    print("=" * 105)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    value_df = pd.read_parquet(
        VALUE_FILE
    )

    race_df = pd.read_parquet(
        RACE_FILE
    )

    if not np.array_equal(
        value_df[
            "source_row"
        ].to_numpy(),
        race_df[
            "source_row"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "V6.0 / V6.1 row alignment mismatch"
        )

    tech = np.load(
        TECH_DIR
        / "technical_features.npy",
        mmap_mode="r",
    )

    tech_valid = np.load(
        TECH_DIR
        / "technical_valid.npy",
        mmap_mode="r",
    )

    m1 = np.load(
        v60.M1_FEATURES,
        mmap_mode="r",
    )

    source = value_df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    m1_end = value_df[
        "m1_end_index"
    ].to_numpy(
        np.int64
    )

    years = value_df[
        "year"
    ].to_numpy(
        np.int16
    )

    target_valid = (
        value_df[
            "target_valid"
        ].to_numpy(
            np.int8
        )
        == 1
    )

    race = race_df[
        v60.HEAD_COLS
    ].to_numpy(
        np.int8
    )

    race_y = (
        race == 1
    ).astype(
        np.float32
    )

    race_mask = (
        race >= -1
    ).astype(
        np.float32
    )

    raw_values = value_df[
        [
            "long_value_bps",
            "short_value_bps",
        ]
    ].to_numpy(
        np.float32
    )

    values = value_transform(
        raw_values
    )

    profitable = (
        raw_values > 0
    ).astype(
        np.float32
    )

    action = value_df[
        "best_action"
    ].to_numpy(
        np.int64
    )

    eligible = (
        target_valid
        & (
            tech_valid[source]
            == 1
        )
        & (
            m1_end
            >= v60.SEQ - 1
        )
    )

    train_rows = np.flatnonzero(
        eligible
        & (years <= 2022)
    )

    val_rows = np.flatnonzero(
        eligible
        & (years >= 2023)
        & (years <= 2024)
    )

    test_rows = np.flatnonzero(
        eligible
        & (years == 2025)
    )

    bench_rows = np.flatnonzero(
        eligible
        & (years == 2026)
    )

    print()
    print(
        "TRAIN:",
        f"{len(train_rows):,}",
    )

    print(
        "VAL:",
        f"{len(val_rows):,}",
    )

    print(
        "2025 TEST:",
        f"{len(test_rows):,}",
    )

    print(
        "2026 RESERVED:",
        f"{len(bench_rows):,}",
    )

    print()
    print("TRAIN ACTION POPULATION")
    print("-" * 105)

    for a, name in (
        (0, "NO_TRADE"),
        (1, "LONG"),
        (2, "SHORT"),
    ):
        print(
            name,
            f"{(action[train_rows] == a).mean():.2%}",
        )

    norm = np.load(
        V60_ART
        / "normalization_v60.npz"
    )

    norms = {
        "m1_mean":
            norm["m1_mean"],

        "m1_std":
            norm["m1_std"],

        "tech_mean":
            norm["tech_mean"],

        "tech_std":
            norm["tech_std"],
    }

    arrays = {
        "m1": m1,
        "tech": tech,
        "source": source,
        "m1_end": m1_end,
        "race_y": race_y,
        "race_mask": race_mask,
        "values": values,
        "profitable": profitable,
        "action": action,
    }

    train_loader = make_loader(
        train_rows,
        True,
        arrays,
        norms,
    )

    val_loader = make_loader(
        val_rows,
        False,
        arrays,
        norms,
    )

    test_loader = make_loader(
        test_rows,
        False,
        arrays,
        norms,
    )

    model = ProfitAwareDualBrain(
        n_tech=tech.shape[1]
    ).to(device)

    # Initialize both brains and fusion
    # from frozen V6.0 champion.
    old = torch.load(
        V60_ART
        / "best_dual_brain_v60.pt",
        map_location=device,
        weights_only=False,
    )

    model.base.load_state_dict(
        old["model"]
    )

    print()
    print(
        "Initialized from V6.0 epoch:",
        old["epoch"],
    )

    print(
        "Total parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # Historical brain already showed
    # fast overfitting in V6.0.
    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.base.historical.parameters(),
                "lr":
                    2e-5,
            },
            {
                "params":
                    model.base.technical.parameters(),
                "lr":
                    8e-5,
            },
            {
                "params":
                    model.base.fusion.parameters(),
                "lr":
                    1.5e-4,
            },
            {
                "params": [
                    p
                    for name, p
                    in model.named_parameters()
                    if not (
                        name.startswith(
                            "base.historical."
                        )
                        or name.startswith(
                            "base.technical."
                        )
                        or name.startswith(
                            "base.fusion."
                        )
                    )
                ],
                "lr":
                    3e-4,
            },
        ],
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    )

    # Preserve old race skill.
    pos_weight = []

    for j in range(6):
        valid = (
            race_mask[
                train_rows,
                j,
            ] > 0.5
        )

        yy = race_y[
            train_rows[
                valid
            ],
            j,
        ]

        pos = yy.sum()
        neg = len(yy) - pos

        pos_weight.append(
            min(
                float(
                    neg
                    / max(pos, 1.0)
                ),
                30.0,
            )
        )

    race_pos_weight = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )

    checkpoint = (
        OUT
        / "best_profit_aware_v61.pt"
    )

    best = -np.inf
    bad = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            race_pos_weight,
        )

        val_pred = predict(
            model,
            val_loader,
            device,
        )

        score, _ = policy_metrics(
            f"EPOCH {epoch} VALIDATION",
            val_pred,
        )

        print()
        print(
            f"Epoch {epoch:02d}"
            f" | loss={loss:.5f}"
            f" | utility={score:+.4f}"
        )

        if score > best:
            best = score
            bad = 0

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_utility":
                        score,

                    "value_scale":
                        VALUE_SCALE,

                    "source_checkpoint":
                        str(
                            V60_ART
                            / "best_dual_brain_v60.pt"
                        ),
                },
                checkpoint,
            )

            print(
                "NEW V6.1 CHAMPION"
            )

        else:
            bad += 1

        if bad >= PATIENCE:
            print(
                "EARLY STOP"
            )
            break

    print()
    print("=" * 105)
    print("FROZEN V6.1 CHAMPION")
    print("=" * 105)

    ckpt = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    print(
        "Epoch:",
        ckpt["epoch"],
    )

    print(
        "VAL utility:",
        f"{ckpt['val_utility']:+.4f}",
    )

    val_pred = predict(
        model,
        val_loader,
        device,
    )

    policy_metrics(
        "FINAL 2023-2024 VALIDATION",
        val_pred,
    )

    print()
    print("=" * 105)
    print(
        "2025 OUT-OF-TIME TEST"
    )
    print("=" * 105)

    test_pred = predict(
        model,
        test_loader,
        device,
    )

    policy_metrics(
        "2025 PROFIT-AWARE POLICY",
        test_pred,
    )

    digest = sha256(
        checkpoint
    )

    freeze = {
        "checkpoint":
            str(checkpoint),

        "sha256":
            digest,

        "epoch":
            int(
                ckpt["epoch"]
            ),

        "val_utility":
            float(
                ckpt["val_utility"]
            ),

        "train":
            "2016-2022",

        "validation":
            "2023-2024",

        "development_test":
            "2025",

        "2026_used_for_v61_training":
            False,

        "primary_objective":
            "predict LONG/SHORT realized action value",

        "deployment":
            "LONG / SHORT / NO_TRADE",
    }

    with open(
        OUT
        / "FROZEN_V61_BEFORE_2026.json",
        "w",
    ) as f:
        json.dump(
            freeze,
            f,
            indent=2,
        )

    print()
    print(
        "SHA256:",
        digest,
    )

    print(
        "2026 NOT EVALUATED BY V6.1."
    )


if __name__ == "__main__":
    main()
