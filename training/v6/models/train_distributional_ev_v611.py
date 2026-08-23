from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import training.v6.models.train_dual_brain_v60 as v60


DATA = Path(
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
    "training/artifacts/v6/distributional_ev_v611"
)

BATCH = 768
EPOCHS = 10
PATIENCE = 3
SEED = 611

COST = 0.5
TP_VALUE = 30.0 - COST
SL_VALUE = -15.0 - COST

TIMEOUT_SCALE = 10.0

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
        torch.cuda.manual_seed_all(SEED)


def timeout_transform(x):
    x = np.clip(
        x,
        -40.0,
        40.0,
    )

    return np.arcsinh(
        x / TIMEOUT_SCALE
    ).astype(np.float32)


def timeout_inverse(x):
    x = np.clip(
        x,
        -3.0,
        3.0,
    )

    return (
        np.sinh(x)
        * TIMEOUT_SCALE
    )


class DistDataset(Dataset):
    def __init__(
        self,
        rows,
        m1,
        tech,
        source,
        m1_end,
        outcome,
        valid,
        timeout_target,
        best_action,
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

        self.outcome = outcome
        self.valid = valid

        self.timeout_target = (
            timeout_target
        )

        self.best_action = (
            best_action
        )

        self.m1_mean = m1_mean
        self.m1_std = m1_std

        self.tech_mean = tech_mean
        self.tech_std = tech_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        end = int(self.m1_end[r])

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
                self.outcome[r]
            ),
            torch.from_numpy(
                self.valid[r]
            ),
            torch.from_numpy(
                self.timeout_target[r]
            ),
            torch.tensor(
                self.best_action[r],
                dtype=torch.long,
            ),
        )


class DistributionalDualBrain(nn.Module):
    def __init__(self, n_tech):
        super().__init__()

        self.base = v60.DualBrain(
            n_tech=n_tech
        )

        # 2 sides x 3 outcomes:
        # 0 TP
        # 1 SL
        # 2 TIMEOUT
        self.outcome_h = nn.Linear(
            128,
            6,
        )

        self.outcome_t = nn.Linear(
            128,
            6,
        )

        self.outcome_f = nn.Linear(
            128,
            6,
        )

        # Expected terminal PnL,
        # conditioned on timeout.
        self.timeout_h = nn.Linear(
            128,
            2,
        )

        self.timeout_t = nn.Linear(
            128,
            2,
        )

        self.timeout_f = nn.Linear(
            128,
            2,
        )

        # Auxiliary explicit competition.
        self.action_head = nn.Linear(
            128,
            3,
        )

    def forward(
        self,
        sequence,
        technical,
    ):
        base = self.base(
            sequence,
            technical,
        )

        h = base["embedding_h"]
        t = base["embedding_t"]
        f = base["embedding_f"]

        return {
            "outcome_h":
                self.outcome_h(h)
                .view(-1, 2, 3),

            "outcome_t":
                self.outcome_t(t)
                .view(-1, 2, 3),

            "outcome_f":
                self.outcome_f(f)
                .view(-1, 2, 3),

            "timeout_h":
                self.timeout_h(h),

            "timeout_t":
                self.timeout_t(t),

            "timeout_f":
                self.timeout_f(f),

            "action":
                self.action_head(f),

            "embedding_h": h,
            "embedding_t": t,
            "embedding_f": f,
        }


def make_loader(
    rows,
    shuffle,
    arrays,
    norms,
):
    ds = DistDataset(
        rows=rows,
        m1=arrays["m1"],
        tech=arrays["tech"],
        source=arrays["source"],
        m1_end=arrays["m1_end"],
        outcome=arrays["outcome"],
        valid=arrays["valid"],
        timeout_target=arrays[
            "timeout_target"
        ],
        best_action=arrays[
            "best_action"
        ],
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


def outcome_loss(
    logits,
    target,
    valid,
):
    total = 0.0
    denom = 0

    for side in range(2):
        mask = valid[:, side]

        if mask.any():
            total = total + F.cross_entropy(
                logits[
                    mask,
                    side,
                    :
                ],
                target[
                    mask,
                    side,
                ],
            )

            denom += 1

    return total / max(
        denom,
        1,
    )


def timeout_loss(
    prediction,
    target,
    outcome,
    valid,
):
    total = 0.0
    denom = 0

    for side in range(2):
        mask = (
            valid[:, side]
            & (
                outcome[:, side]
                == 2
            )
        )

        if mask.any():
            total = total + (
                F.smooth_l1_loss(
                    prediction[
                        mask,
                        side,
                    ],
                    target[
                        mask,
                        side,
                    ],
                )
            )

            denom += 1

    if denom == 0:
        return (
            prediction.sum()
            * 0.0
        )

    return total / denom


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
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
        outcome,
        valid,
        timeout_target,
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

        outcome = outcome.to(
            device,
            non_blocking=True,
        )

        valid = valid.to(
            device,
            non_blocking=True,
        )

        timeout_target = (
            timeout_target.to(
                device,
                non_blocking=True,
            )
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

            ce_f = outcome_loss(
                out["outcome_f"],
                outcome,
                valid,
            )

            ce_h = outcome_loss(
                out["outcome_h"],
                outcome,
                valid,
            )

            ce_t = outcome_loss(
                out["outcome_t"],
                outcome,
                valid,
            )

            to_f = timeout_loss(
                out["timeout_f"],
                timeout_target,
                outcome,
                valid,
            )

            to_h = timeout_loss(
                out["timeout_h"],
                timeout_target,
                outcome,
                valid,
            )

            to_t = timeout_loss(
                out["timeout_t"],
                timeout_target,
                outcome,
                valid,
            )

            act = F.cross_entropy(
                out["action"],
                action,
            )

            loss = (
                1.00 * ce_f
                + 0.20 * ce_h
                + 0.25 * ce_t

                + 0.25 * to_f
                + 0.04 * to_h
                + 0.05 * to_t

                + 0.20 * act
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
            loss.detach()
            .float()
            .item()
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

    store = {
        "outcome_f": [],
        "outcome_h": [],
        "outcome_t": [],
        "timeout_f": [],
        "action": [],
        "true_outcome": [],
        "true_timeout": [],
        "true_action": [],
    }

    for (
        seq,
        tech,
        outcome,
        valid,
        timeout_target,
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

        for name in (
            "outcome_f",
            "outcome_h",
            "outcome_t",
        ):
            store[name].append(
                torch.softmax(
                    out[name],
                    dim=2,
                )
                .float()
                .cpu()
                .numpy()
            )

        store[
            "timeout_f"
        ].append(
            out["timeout_f"]
            .float()
            .cpu()
            .numpy()
        )

        store[
            "action"
        ].append(
            torch.softmax(
                out["action"],
                dim=1,
            )
            .float()
            .cpu()
            .numpy()
        )

        store[
            "true_outcome"
        ].append(
            outcome.numpy()
        )

        store[
            "true_timeout"
        ].append(
            timeout_target.numpy()
        )

        store[
            "true_action"
        ].append(
            action.numpy()
        )

    return {
        k: np.concatenate(v)
        for k, v in store.items()
    }


def expected_value(
    probability,
    timeout_pred,
):
    timeout_bps = (
        timeout_inverse(
            timeout_pred
        )
    )

    ev = (
        probability[:, :, 0]
        * TP_VALUE

        + probability[:, :, 1]
        * SL_VALUE

        + probability[:, :, 2]
        * timeout_bps
    )

    return ev


def policy_metrics(
    name,
    pred,
    actual_values,
):
    prob = pred[
        "outcome_f"
    ]

    ev = expected_value(
        prob,
        pred["timeout_f"],
    )

    predicted_long = (
        ev[:, 0]
        >= ev[:, 1]
    )

    selected_ev = np.where(
        predicted_long,
        ev[:, 0],
        ev[:, 1],
    )

    realized = np.where(
        predicted_long,
        actual_values[:, 0],
        actual_values[:, 1],
    )

    p_tp = np.where(
        predicted_long,
        prob[:, 0, 0],
        prob[:, 1, 0],
    )

    p_sl = np.where(
        predicted_long,
        prob[:, 0, 1],
        prob[:, 1, 1],
    )

    # Primary deployment ranking:
    # model-derived expected value.
    score = selected_ev

    print()
    print(name)
    print("-" * 112)

    print(
        "EV > 0 fraction:",
        f"{(selected_ev > 0).mean():.2%}",
    )

    print(
        "Mean predicted EV:",
        f"{selected_ev.mean():+.3f}",
    )

    print(
        "Mean realized selected side:",
        f"{realized.mean():+.3f}",
    )

    results = {}

    for coverage in COVERAGES:
        n = max(
            1,
            int(
                round(
                    len(score)
                    * coverage
                )
            ),
        )

        idx = np.argpartition(
            score,
            -n,
        )[-n:]

        pnl = realized[idx]

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

        win = (
            pnl > 0
        ).mean()

        long_share = (
            predicted_long[
                idx
            ].mean()
        )

        result = {
            "mean":
                pnl.mean(),

            "win":
                win,

            "pf":
                pf,

            "long":
                long_share,
        }

        results[
            coverage
        ] = result

        print(
            f"{coverage:>5.1%}"
            f" n={n:>5}"
            f" WIN={win:>6.2%}"
            f" mean={pnl.mean():>+7.3f}"
            f" PF={pf:>6.3f}"
            f" LONG={long_share:>6.2%}"
            f" predEV={selected_ev[idx].mean():>+7.3f}"
            f" PTP={p_tp[idx].mean():>6.2%}"
            f" PSL={p_sl[idx].mean():>6.2%}"
        )

    stable = (
        0.40
        * results[0.05]["mean"]

        + 0.35
        * results[0.02]["mean"]

        + 0.25
        * results[0.01]["mean"]
    )

    print()
    print(
        "STABLE UTILITY:",
        f"{stable:+.4f}",
    )

    return stable


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def main():
    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.1.1 DISTRIBUTIONAL EV"
    )
    print("=" * 112)

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

    df = pd.read_parquet(DATA)

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

    source = df[
        "source_row"
    ].to_numpy(np.int64)

    m1_end = df[
        "m1_end_index"
    ].to_numpy(np.int64)

    years = df[
        "year"
    ].to_numpy(np.int16)

    long_race = df[
        "long_race_tp30_sl15"
    ].to_numpy(np.int8)

    short_race = df[
        "short_race_tp30_sl15"
    ].to_numpy(np.int8)

    race = np.column_stack(
        [
            long_race,
            short_race,
        ]
    )

    valid = (
        (race != -2)
        & (race != -3)
    )

    # 0 TP
    # 1 SL
    # 2 TIMEOUT
    outcome = np.zeros(
        race.shape,
        dtype=np.int64,
    )

    outcome[
        race == 1
    ] = 0

    outcome[
        race == 0
    ] = 1

    outcome[
        race == -1
    ] = 2

    long_terminal = df[
        "long_terminal_bps"
    ].to_numpy(
        np.float32
    ) - COST

    short_terminal = df[
        "short_terminal_bps"
    ].to_numpy(
        np.float32
    ) - COST

    timeout_raw = np.column_stack(
        [
            long_terminal,
            short_terminal,
        ]
    )

    timeout_target = (
        timeout_transform(
            timeout_raw
        )
    )

    # Exact realized action values.
    realized = np.zeros(
        (
            len(df),
            2,
        ),
        dtype=np.float32,
    )

    for side in range(2):
        r = race[:, side]

        realized[
            r == 1,
            side,
        ] = TP_VALUE

        realized[
            r == 0,
            side,
        ] = SL_VALUE

        timeout_mask = (
            r == -1
        )

        realized[
            timeout_mask,
            side,
        ] = timeout_raw[
            timeout_mask,
            side,
        ]

        invalid = (
            (r == -2)
            | (r == -3)
        )

        realized[
            invalid,
            side,
        ] = np.nan

    # Oracle action only as auxiliary target.
    best_action = np.zeros(
        len(df),
        dtype=np.int64,
    )

    long_ok = (
        realized[:, 0] > 0
    )

    short_ok = (
        realized[:, 1] > 0
    )

    choose_long = (
        long_ok
        & (
            ~short_ok
            | (
                realized[:, 0]
                >= realized[:, 1]
            )
        )
    )

    choose_short = (
        short_ok
        & (
            ~long_ok
            | (
                realized[:, 1]
                > realized[:, 0]
            )
        )
    )

    best_action[
        choose_long
    ] = 1

    best_action[
        choose_short
    ] = 2

    eligible = (
        df[
            "horizon_valid"
        ].to_numpy(
            np.int8
        ) == 1
    )

    eligible &= (
        tech_valid[source]
        == 1
    )

    eligible &= (
        m1_end
        >= v60.SEQ - 1
    )

    eligible &= (
        valid.all(axis=1)
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
    print("TRAIN OUTCOMES")
    print("-" * 112)

    for side, name in (
        (0, "LONG"),
        (1, "SHORT"),
    ):
        y = outcome[
            train_rows,
            side,
        ]

        print(
            name,
            f"TP={(y == 0).mean():.2%}",
            f"SL={(y == 1).mean():.2%}",
            f"TO={(y == 2).mean():.2%}",
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
        "outcome": outcome,
        "valid": valid,
        "timeout_target":
            timeout_target,
        "best_action":
            best_action,
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

    model = DistributionalDualBrain(
        n_tech=tech.shape[1]
    ).to(device)

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
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.base.historical.parameters(),
                "lr": 1e-5,
            },
            {
                "params":
                    model.base.technical.parameters(),
                "lr": 6e-5,
            },
            {
                "params":
                    model.base.fusion.parameters(),
                "lr": 1.2e-4,
            },
            {
                "params": (
                    list(
                        model.outcome_h.parameters()
                    )
                    + list(
                        model.outcome_t.parameters()
                    )
                    + list(
                        model.outcome_f.parameters()
                    )
                    + list(
                        model.timeout_h.parameters()
                    )
                    + list(
                        model.timeout_t.parameters()
                    )
                    + list(
                        model.timeout_f.parameters()
                    )
                    + list(
                        model.action_head.parameters()
                    )
                ),
                "lr": 3e-4,
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

    checkpoint = (
        OUT
        / "best_distributional_ev_v611.pt"
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
        )

        pred = predict(
            model,
            val_loader,
            device,
        )

        score = policy_metrics(
            f"EPOCH {epoch} VALIDATION",
            pred,
            realized[
                val_rows
            ],
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

                    "tp_value":
                        TP_VALUE,

                    "sl_value":
                        SL_VALUE,

                    "timeout_scale":
                        TIMEOUT_SCALE,
                },
                checkpoint,
            )

            print(
                "NEW V6.1.1 CHAMPION"
            )

        else:
            bad += 1

        if bad >= PATIENCE:
            print("EARLY STOP")
            break

    print()
    print("=" * 112)
    print("FROZEN V6.1.1 CHAMPION")
    print("=" * 112)

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

    final_val = predict(
        model,
        val_loader,
        device,
    )

    policy_metrics(
        "FINAL 2023-2024",
        final_val,
        realized[val_rows],
    )

    print()
    print("=" * 112)
    print("2025 OUT-OF-TIME TEST")
    print("=" * 112)

    test_pred = predict(
        model,
        test_loader,
        device,
    )

    policy_metrics(
        "2025 DISTRIBUTIONAL EV",
        test_pred,
        realized[test_rows],
    )

    digest = sha256_file(
        checkpoint
    )

    freeze = {
        "checkpoint":
            str(checkpoint),

        "sha256":
            digest,

        "epoch":
            int(ckpt["epoch"]),

        "val_utility":
            float(
                ckpt["val_utility"]
            ),

        "objective":
            (
                "P(TP/SL/timeout) + "
                "timeout conditional return"
            ),

        "2026_used":
            False,
    }

    with open(
        OUT
        / "FROZEN_V611_BEFORE_2026.json",
        "w",
    ) as f:
        json.dump(
            freeze,
            f,
            indent=2,
        )

    print()
    print("SHA256:", digest)
    print(
        "2026 NOT EVALUATED BY V6.1.1."
    )


if __name__ == "__main__":
    main()
