from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)


TARGET = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

TECH_DIR = Path(
    "training/v6/data_lake/technical_state_v60"
)

M1_FEATURES = Path(
    "training/v2/data_lake/micro_path_v33/"
    "m1_features.npy"
)

M1_TIMES = Path(
    "training/v2/data_lake/micro_path_v33/"
    "m1_timestamps_ns.npy"
)

OUT = Path(
    "training/artifacts/v6/dual_brain_v60"
)

SEQ = 120
BATCH = 768
EPOCHS = 12
PATIENCE = 3

LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 600

HEAD_COLS = [
    "long_race_tp20_sl10",
    "short_race_tp20_sl10",
    "long_race_tp30_sl15",
    "short_race_tp30_sl15",
    "long_race_tp40_sl20",
    "short_race_tp40_sl20",
]

HEAD_NAMES = [
    "L20_10",
    "S20_10",
    "L30_15",
    "S30_15",
    "L40_20",
    "S40_20",
]

REG_COLS = [
    "long_mfe_bps",
    "long_mae_bps",
    "short_mfe_bps",
    "short_mae_bps",
]


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def streaming_stats(x, stop, chunk=250000):
    total = None
    total_sq = None
    count = None

    for start in range(0, stop, chunk):
        end = min(
            start + chunk,
            stop,
        )

        z = np.asarray(
            x[start:end],
            dtype=np.float64,
        )

        finite = np.isfinite(z)

        safe = np.where(
            finite,
            z,
            0.0,
        )

        s = safe.sum(axis=0)
        ss = (safe * safe).sum(axis=0)
        c = finite.sum(axis=0)

        if total is None:
            total = s
            total_sq = ss
            count = c
        else:
            total += s
            total_sq += ss
            count += c

    mean = total / np.maximum(count, 1)

    var = (
        total_sq
        / np.maximum(count, 1)
        - mean * mean
    )

    std = np.sqrt(
        np.maximum(
            var,
            1e-8,
        )
    )

    std = np.maximum(
        std,
        1e-4,
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def technical_stats(
    tech,
    source,
    train_rows,
    chunk=100000,
):
    total = None
    total_sq = None
    count = None

    for start in range(
        0,
        len(train_rows),
        chunk,
    ):
        rows = train_rows[
            start:start + chunk
        ]

        z = np.asarray(
            tech[
                source[rows]
            ],
            dtype=np.float64,
        )

        finite = np.isfinite(z)

        safe = np.where(
            finite,
            z,
            0.0,
        )

        s = safe.sum(axis=0)
        ss = (safe * safe).sum(axis=0)
        c = finite.sum(axis=0)

        if total is None:
            total = s
            total_sq = ss
            count = c
        else:
            total += s
            total_sq += ss
            count += c

    mean = total / np.maximum(
        count,
        1,
    )

    var = (
        total_sq
        / np.maximum(count, 1)
        - mean * mean
    )

    std = np.sqrt(
        np.maximum(
            var,
            1e-8,
        )
    )

    std = np.maximum(
        std,
        1e-4,
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


class TenDataset(Dataset):
    def __init__(
        self,
        rows,
        m1,
        tech,
        source,
        m1_end,
        labels,
        masks,
        regression,
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

        self.labels = labels
        self.masks = masks
        self.regression = regression

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
                end - SEQ + 1:
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

        technical = np.asarray(
            self.tech[
                self.source[r]
            ],
            dtype=np.float32,
        )

        technical = np.nan_to_num(
            technical,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        technical = (
            technical - self.tech_mean
        ) / self.tech_std

        return (
            torch.from_numpy(seq),
            torch.from_numpy(technical),
            torch.from_numpy(
                self.labels[r]
            ),
            torch.from_numpy(
                self.masks[r]
            ),
            torch.from_numpy(
                self.regression[r]
            ),
        )


class ResidualTCN(nn.Module):
    def __init__(
        self,
        channels=64,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.norm1 = nn.GroupNorm(
            8,
            channels,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.norm2 = nn.GroupNorm(
            8,
            channels,
        )

        self.drop = nn.Dropout(
            0.05
        )

    def forward(self, x):
        z = self.conv1(x)
        z = self.norm1(z)
        z = F.gelu(z)

        z = self.drop(z)

        z = self.conv2(z)
        z = self.norm2(z)

        return F.gelu(
            x + z
        )


class HistoricalBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                20,
                64,
                kernel_size=5,
                padding=2,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            ResidualTCN(64, 1),
            ResidualTCN(64, 2),
            ResidualTCN(64, 4),
            ResidualTCN(64, 8),
        )

        self.out = nn.Sequential(
            nn.Linear(
                128,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        # B,T,F -> B,F,T
        x = x.transpose(1, 2)

        x = self.stem(x)
        x = self.blocks(x)

        avg = x.mean(dim=-1)
        mx = x.amax(dim=-1)

        return self.out(
            torch.cat(
                [avg, mx],
                dim=1,
            )
        )


class TechnicalBrain(nn.Module):
    def __init__(self, n_features=101):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                n_features
            ),
            nn.Linear(
                n_features,
                256,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        return self.net(x)


class DualBrain(nn.Module):
    def __init__(self, n_tech=101):
        super().__init__()

        self.historical = (
            HistoricalBrain()
        )

        self.technical = (
            TechnicalBrain(
                n_tech
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                256,
                256,
            ),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(128),
        )

        self.hist_head = nn.Linear(
            128,
            6,
        )

        self.tech_head = nn.Linear(
            128,
            6,
        )

        self.fusion_head = nn.Linear(
            128,
            6,
        )

        self.reg_head = nn.Linear(
            128,
            4,
        )

    def forward(
        self,
        sequence,
        technical,
    ):
        h = self.historical(
            sequence
        )

        t = self.technical(
            technical
        )

        fused = self.fusion(
            torch.cat(
                [h, t],
                dim=1,
            )
        )

        return {
            "historical":
                self.hist_head(h),

            "technical":
                self.tech_head(t),

            "fusion":
                self.fusion_head(
                    fused
                ),

            "regression":
                self.reg_head(
                    fused
                ),

            "embedding_h":
                h,

            "embedding_t":
                t,

            "embedding_f":
                fused,
        }


def masked_bce(
    logits,
    target,
    mask,
    pos_weight,
    head_weight,
):
    loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
    )

    weight = (
        mask
        * head_weight[
            None, :
        ]
    )

    return (
        (loss * weight).sum()
        / weight.sum().clamp_min(
            1.0
        )
    )


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    preds = {
        "historical": [],
        "technical": [],
        "fusion": [],
    }

    ys = []
    masks = []

    with torch.inference_mode():
        for (
            seq,
            tech,
            y,
            mask,
            reg,
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

            for name in preds:
                preds[name].append(
                    torch.sigmoid(
                        out[name]
                    )
                    .cpu()
                    .numpy()
                )

            ys.append(
                y.numpy()
            )

            masks.append(
                mask.numpy()
            )

    y = np.concatenate(ys)
    mask = np.concatenate(masks)

    pred = {
        name:
            np.concatenate(parts)
        for name, parts
        in preds.items()
    }

    return pred, y, mask


def branch_metrics(
    prediction,
    y,
    mask,
):
    result = []

    for j, name in enumerate(
        HEAD_NAMES
    ):
        valid = (
            mask[:, j]
            > 0.5
        )

        yy = y[
            valid,
            j,
        ]

        pp = prediction[
            valid,
            j,
        ]

        prevalence = yy.mean()

        ap = average_precision_score(
            yy,
            pp,
        )

        auc = roc_auc_score(
            yy,
            pp,
        )

        row = {
            "name": name,
            "prevalence":
                prevalence,
            "ap": ap,
            "auc": auc,
        }

        for coverage in (
            0.05,
            0.02,
            0.01,
            0.005,
        ):
            n = max(
                1,
                int(
                    round(
                        len(pp)
                        * coverage
                    )
                ),
            )

            idx = np.argpartition(
                pp,
                -n,
            )[-n:]

            row[
                f"p{coverage}"
            ] = yy[
                idx
            ].mean()

        result.append(row)

    return result


def print_metrics(
    title,
    prediction,
    y,
    mask,
):
    rows = branch_metrics(
        prediction,
        y,
        mask,
    )

    print()
    print(title)
    print("-" * 105)

    print(
        "head      prev      AP      AUC"
        "     top5     top2"
        "     top1    top0.5"
    )

    for r in rows:
        print(
            f"{r['name']:<8}"
            f"{r['prevalence']:>7.2%}"
            f"{r['ap']:>8.4f}"
            f"{r['auc']:>9.4f}"
            f"{r['p0.05']:>9.2%}"
            f"{r['p0.02']:>9.2%}"
            f"{r['p0.01']:>9.2%}"
            f"{r['p0.005']:>10.2%}"
        )

    primary_ap = (
        rows[2]["ap"]
        + rows[3]["ap"]
    ) / 2.0

    primary_top1 = (
        rows[2]["p0.01"]
        + rows[3]["p0.01"]
    ) / 2.0

    print()
    print(
        "PRIMARY TP30/SL15 AP:",
        f"{primary_ap:.4f}",
    )

    print(
        "PRIMARY TP30/SL15 "
        "Top1% precision:",
        f"{primary_top1:.2%}",
    )

    return (
        primary_ap,
        rows,
    )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def make_loader(
    rows,
    shuffle,
    arrays,
    norms,
):
    ds = TenDataset(
        rows=rows,
        m1=arrays["m1"],
        tech=arrays["tech"],
        source=arrays["source"],
        m1_end=arrays["m1_end"],
        labels=arrays["labels"],
        masks=arrays["masks"],
        regression=arrays["regression"],
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
    pos_weight,
    head_weight,
):
    model.train()

    total = 0.0
    seen = 0

    amp = (
        device.type == "cuda"
    )

    for (
        seq,
        tech,
        y,
        mask,
        reg,
    ) in loader:

        seq = seq.to(
            device,
            non_blocking=True,
        )

        tech = tech.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        mask = mask.to(
            device,
            non_blocking=True,
        )

        reg = reg.to(
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

            loss_f = masked_bce(
                out["fusion"],
                y,
                mask,
                pos_weight,
                head_weight,
            )

            loss_h = masked_bce(
                out["historical"],
                y,
                mask,
                pos_weight,
                head_weight,
            )

            loss_t = masked_bce(
                out["technical"],
                y,
                mask,
                pos_weight,
                head_weight,
            )

            loss_r = (
                F.smooth_l1_loss(
                    out["regression"],
                    reg,
                )
            )

            loss = (
                loss_f
                + 0.25 * loss_h
                + 0.25 * loss_t
                + 0.05 * loss_r
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
            float(loss)
            * b
        )

        seen += b

    return total / seen


def main():
    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.0 DUAL-BRAIN GPU TRAINING"
    )
    print("=" * 105)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    df = pd.read_parquet(
        TARGET
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
        M1_FEATURES,
        mmap_mode="r",
    )

    m1_times = np.load(
        M1_TIMES,
        mmap_mode="r",
    )

    source = df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    m1_end = df[
        "m1_end_index"
    ].to_numpy(
        np.int64
    )

    year = df[
        "year"
    ].to_numpy(
        np.int16
    )

    race = df[
        HEAD_COLS
    ].to_numpy(
        np.int8
    )

    labels = (
        race == 1
    ).astype(
        np.float32
    )

    # Valid labels:
    # -1 timeout
    #  0 SL first
    #  1 TP first
    #
    # -2 ambiguous
    # -3 invalid horizon
    masks = (
        race >= -1
    ).astype(
        np.float32
    )

    regression = np.log1p(
        np.clip(
            df[
                REG_COLS
            ].to_numpy(
                np.float32
            ),
            0.0,
            200.0,
        )
    ).astype(
        np.float32
    )

    eligible = (
        (
            df[
                "horizon_valid"
            ].to_numpy(
                np.int8
            )
            == 1
        )
        & (
            tech_valid[
                source
            ]
            == 1
        )
        & (
            m1_end
            >= SEQ - 1
        )
    )

    train_rows = np.flatnonzero(
        eligible
        & (year <= 2022)
    )

    val_rows = np.flatnonzero(
        eligible
        & (
            (year >= 2023)
            & (year <= 2024)
        )
    )

    test_rows = np.flatnonzero(
        eligible
        & (year == 2025)
    )

    benchmark_rows = np.flatnonzero(
        eligible
        & (year == 2026)
    )

    print()
    print(
        "TRAIN 2016-2022:",
        f"{len(train_rows):,}",
    )

    print(
        "VAL   2023-2024:",
        f"{len(val_rows):,}",
    )

    print(
        "TEST  2025:",
        f"{len(test_rows):,}",
    )

    print(
        "2026 BENCHMARK RESERVED:",
        f"{len(benchmark_rows):,}",
    )

    print()
    print(
        "PRIMARY BASE RATES"
    )
    print("-" * 105)

    for name, rows in (
        ("TRAIN", train_rows),
        ("VAL", val_rows),
        ("2025", test_rows),
        ("2026 RESERVED", benchmark_rows),
    ):
        long_rate = (
            labels[
                rows,
                2,
            ].mean()
        )

        short_rate = (
            labels[
                rows,
                3,
            ].mean()
        )

        print(
            f"{name:<15}",
            f"LONG30={long_rate:.2%}",
            f"SHORT30={short_rate:.2%}",
        )

    norm_file = (
        OUT
        / "normalization_v60.npz"
    )

    if norm_file.exists():
        z = np.load(
            norm_file
        )

        norms = {
            "m1_mean":
                z["m1_mean"],

            "m1_std":
                z["m1_std"],

            "tech_mean":
                z["tech_mean"],

            "tech_std":
                z["tech_std"],
        }

        print()
        print(
            "NORMALIZATION CACHE FOUND"
        )

    else:
        print()
        print(
            "Computing TRAIN-ONLY "
            "normalization..."
        )

        cutoff = int(
            pd.Timestamp(
                "2023-01-01",
                tz="UTC",
            ).value
        )

        stop = int(
            np.searchsorted(
                m1_times,
                cutoff,
                side="left",
            )
        )

        m1_mean, m1_std = (
            streaming_stats(
                m1,
                stop,
            )
        )

        tech_mean, tech_std = (
            technical_stats(
                tech,
                source,
                train_rows,
            )
        )

        norms = {
            "m1_mean":
                m1_mean,

            "m1_std":
                m1_std,

            "tech_mean":
                tech_mean,

            "tech_std":
                tech_std,
        }

        np.savez(
            norm_file,
            **norms,
        )

        print(
            "Normalization saved."
        )

    arrays = {
        "m1": m1,
        "tech": tech,
        "source": source,
        "m1_end": m1_end,
        "labels": labels,
        "masks": masks,
        "regression": regression,
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

    model = DualBrain(
        n_tech=tech.shape[1]
    ).to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model parameters:",
        f"{params:,}",
    )

    # Train-only positive weights.
    pos_weight = []

    for j in range(6):
        valid = (
            masks[
                train_rows,
                j,
            ] > 0.5
        )

        yy = labels[
            train_rows[
                valid
            ],
            j,
        ]

        pos = yy.sum()
        neg = len(yy) - pos

        weight = (
            neg
            / max(
                pos,
                1.0,
            )
        )

        pos_weight.append(
            min(
                float(weight),
                30.0,
            )
        )

    pos_weight = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device,
    )

    # TP30/SL15 is primary.
    head_weight = torch.tensor(
        [
            0.5,
            0.5,
            1.0,
            1.0,
            0.5,
            0.5,
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Positive weights:",
        [
            round(
                float(x),
                2,
            )
            for x in pos_weight
        ],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    )

    best_score = -np.inf
    bad_epochs = 0

    checkpoint = (
        OUT
        / "best_dual_brain_v60.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            pos_weight,
            head_weight,
        )

        pred, y, mask = evaluate(
            model,
            val_loader,
            device,
        )

        fusion_ap, _ = (
            print_metrics(
                f"EPOCH {epoch} VAL — FUSION",
                pred["fusion"],
                y,
                mask,
            )
        )

        hist_ap, _ = (
            print_metrics(
                f"EPOCH {epoch} VAL — HISTORICAL BRAIN",
                pred["historical"],
                y,
                mask,
            )
        )

        tech_ap, _ = (
            print_metrics(
                f"EPOCH {epoch} VAL — TECHNICAL BRAIN",
                pred["technical"],
                y,
                mask,
            )
        )

        print()
        print(
            f"Epoch {epoch:02d}"
            f" | loss={train_loss:.5f}"
            f" | fusion={fusion_ap:.4f}"
            f" | hist={hist_ap:.4f}"
            f" | tech={tech_ap:.4f}"
        )

        if fusion_ap > best_score:
            best_score = fusion_ap
            bad_epochs = 0

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_primary_ap":
                        fusion_ap,

                    "config": {
                        "seq": SEQ,
                        "technical_features":
                            tech.shape[1],
                        "heads":
                            HEAD_NAMES,
                    },
                },
                checkpoint,
            )

            print(
                "NEW CHAMPION SAVED"
            )

        else:
            bad_epochs += 1

        scheduler.step()

        if bad_epochs >= PATIENCE:
            print(
                "EARLY STOP"
            )
            break

    print()
    print("=" * 105)
    print(
        "LOADING FROZEN CHAMPION"
    )
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
        "Selected epoch:",
        ckpt["epoch"],
    )

    print(
        "Validation primary AP:",
        f"{ckpt['val_primary_ap']:.4f}",
    )

    # Final validation report.
    val_pred, val_y, val_mask = (
        evaluate(
            model,
            val_loader,
            device,
        )
    )

    print_metrics(
        "FINAL VALIDATION — HISTORICAL",
        val_pred["historical"],
        val_y,
        val_mask,
    )

    print_metrics(
        "FINAL VALIDATION — TECHNICAL",
        val_pred["technical"],
        val_y,
        val_mask,
    )

    print_metrics(
        "FINAL VALIDATION — FUSION",
        val_pred["fusion"],
        val_y,
        val_mask,
    )

    # 2025 is touched ONCE after model selection.
    print()
    print("=" * 105)
    print(
        "2025 OUT-OF-TIME DEVELOPMENT TEST"
    )
    print("=" * 105)

    test_pred, test_y, test_mask = (
        evaluate(
            model,
            test_loader,
            device,
        )
    )

    print_metrics(
        "2025 — HISTORICAL BRAIN",
        test_pred["historical"],
        test_y,
        test_mask,
    )

    print_metrics(
        "2025 — TECHNICAL BRAIN",
        test_pred["technical"],
        test_y,
        test_mask,
    )

    print_metrics(
        "2025 — FUSION",
        test_pred["fusion"],
        test_y,
        test_mask,
    )

    digest = sha256_file(
        checkpoint
    )

    freeze = {
        "checkpoint":
            str(checkpoint),

        "sha256":
            digest,

        "selected_epoch":
            int(
                ckpt["epoch"]
            ),

        "val_primary_ap":
            float(
                ckpt[
                    "val_primary_ap"
                ]
            ),

        "train":
            "2016-2022",

        "validation":
            "2023-2024",

        "development_test":
            "2025",

        "2026_benchmark_used":
            False,

        "primary_target":
            "TP30 before SL15",

        "architecture":
            "HistoricalBrain + TechnicalBrain + Fusion",
    }

    with open(
        OUT
        / "FROZEN_BEFORE_2026.json",
        "w",
    ) as f:
        json.dump(
            freeze,
            f,
            indent=2,
        )

    print()
    print(
        "CHECKPOINT SHA256:"
    )

    print(digest)

    print()
    print(
        "2026 WAS NOT EVALUATED."
    )

    print(
        "Frozen policy:",
        OUT
        / "FROZEN_BEFORE_2026.json",
    )


if __name__ == "__main__":
    main()
