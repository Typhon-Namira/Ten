from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.backtests.backtest_surface_policy_v662 as execmod
import training.v6.models.train_execution_precision_brain_v671 as v671


# ============================================================
# CONFIG
# ============================================================

VERSION = "v6.7.2"

DAILY_TARGET_FILE = Path(
    "training/v6/data_lake/"
    "daily_opportunity_targets_v672/"
    "daily_opportunity_targets_v672.parquet"
)

V671_CHAMPION = Path(
    "training/artifacts/v6/"
    "execution_precision_brain_v671/"
    "champion_v671.pt"
)

OUT = Path(
    "training/artifacts/v6/"
    "daily_opportunity_brain_v672"
)

CHAMPION = (
    OUT
    / "champion_v672.pt"
)

TOTAL_EPOCHS = 8
PATIENCE = 3

# Phase 1: new daily head only
# Phase 2: execution head + upper backbone
# Phase 3: entire network including experts
PHASE1_EPOCHS = 1
PHASE2_EPOCHS = 1

LR_DAILY = 3e-4
LR_EXEC_HEAD = 1e-4
LR_UPPER = 3e-5
LR_EXPERTS = 8e-6

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

DAILY_HIDDEN = 192
BEST_NET_SCALE = 30.0

TOP_COLUMNS = (
    "daily_top80_c05",
    "daily_top90_c05",
    "daily_top95_c05",
    "daily_top98_c05",
)

TOP_NAMES = (
    "TOP20",
    "TOP10",
    "TOP5",
    "TOP2",
)

ENTRY_QUANTILES = (
    0.00,
    0.25,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
)

MIN_DAY_BARS = 100

SEED = 20260823


def seed_everything():
    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )


# ============================================================
# DAILY TARGETS
# ============================================================

def load_daily_targets():
    print(
        "Loading V6.7.2 daily opportunity targets ..."
    )

    columns = [
        "source_row",
        "timestamp",
        "year",
        "trading_day",
        "best_side_c05",
        "best_net_c05",
        "direction_gap_c05",
        "positive_task_count_c05",
        "daily_rank_c05",
        *TOP_COLUMNS,
    ]

    df = pd.read_parquet(
        DAILY_TARGET_FILE,
        columns=columns,
    )

    df[
        "timestamp"
    ] = pd.to_datetime(
        df[
            "timestamp"
        ],
        utc=True,
    )

    df[
        "trading_day"
    ] = pd.to_datetime(
        df[
            "trading_day"
        ]
    )

    source = df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    if not np.array_equal(
        source,
        np.arange(
            len(df),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Daily target source alignment failure."
        )

    top = np.stack(
        [
            df[
                col
            ].to_numpy(
                np.float32
            )
            for col in TOP_COLUMNS
        ],
        axis=1,
    )

    result = {
        "timestamp":
            df[
                "timestamp"
            ].to_numpy(),

        "year":
            df[
                "year"
            ].to_numpy(
                np.int16
            ),

        "day_ns":
            df[
                "trading_day"
            ].astype(
                "int64"
            ).to_numpy(
                np.int64
            ),

        "side":
            df[
                "best_side_c05"
            ].to_numpy(
                np.int8
            ),

        "best_net":
            df[
                "best_net_c05"
            ].to_numpy(
                np.float32
            ),

        "direction_gap":
            df[
                "direction_gap_c05"
            ].to_numpy(
                np.float32
            ),

        "positive_count":
            df[
                "positive_task_count_c05"
            ].to_numpy(
                np.float32
            ),

        "rank":
            df[
                "daily_rank_c05"
            ].to_numpy(
                np.float32
            ),

        "top":
            top,
    }

    print(
        "Daily target rows:",
        f"{len(df):,}",
    )

    return result


# ============================================================
# MODEL
# ============================================================

class DailyOpportunityBrainV672(
    nn.Module
):
    def __init__(
        self,
        execution_brain,
    ):
        super().__init__()

        self.exec = (
            execution_brain
        )

        # market state       128
        # p05                18
        # p10                18
        # predicted net      18
        # race probabilities 54
        #
        # total = 236
        input_dim = (
            128
            + 18
            + 18
            + 18
            + 54
        )

        self.daily_body = (
            nn.Sequential(
                nn.Linear(
                    input_dim,
                    DAILY_HIDDEN,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    DAILY_HIDDEN
                ),
                nn.Dropout(
                    0.12
                ),

                nn.Linear(
                    DAILY_HIDDEN,
                    DAILY_HIDDEN,
                ),
                nn.GELU(),
                nn.Dropout(
                    0.10
                ),

                nn.Linear(
                    DAILY_HIDDEN,
                    96,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    96
                ),
            )
        )

        # Ordered:
        # top20, top10, top5, top2
        self.ordinal_head = (
            nn.Linear(
                96,
                4,
            )
        )

        self.rank_head = (
            nn.Linear(
                96,
                1,
            )
        )

        self.best_net_head = (
            nn.Linear(
                96,
                1,
            )
        )

        self.count_head = (
            nn.Linear(
                96,
                1,
            )
        )

        self.side_head = (
            nn.Linear(
                96,
                1,
            )
        )

    def forward(
        self,
        x,
    ):
        e = self.exec(
            x
        )

        p05 = torch.sigmoid(
            e[
                "win05_logit"
            ]
        )

        p10 = torch.sigmoid(
            e[
                "win10_logit"
            ]
        )

        net = e[
            "net05_norm"
        ]

        race_prob = F.softmax(
            e[
                "race_logits"
            ],
            dim=-1,
        ).flatten(
            start_dim=1
        )

        z = torch.cat(
            [
                e[
                    "market_state"
                ],
                p05,
                p10,
                net,
                race_prob,
            ],
            dim=1,
        )

        h = self.daily_body(
            z
        )

        return {
            "ordinal_logits":
                self.ordinal_head(
                    h
                ),

            "rank_logit":
                self.rank_head(
                    h
                ).squeeze(
                    -1
                ),

            "best_net_norm":
                self.best_net_head(
                    h
                ).squeeze(
                    -1
                ),

            "count_norm":
                self.count_head(
                    h
                ).squeeze(
                    -1
                ),

            "side_logit":
                self.side_head(
                    h
                ).squeeze(
                    -1
                ),

            "exec":
                e,
        }


# ============================================================
# PHASE CONTROL
# ============================================================

def set_phase(
    model,
    phase,
):
    # Daily head always trainable.
    for p in model.parameters():
        p.requires_grad = True

    # Freeze complete execution network first.
    for p in model.exec.parameters():
        p.requires_grad = False

    if phase >= 2:

        modules = [
            model.exec.precision_decoder,
            model.exec.side_embedding,
            model.exec.horizon_embedding,
            model.exec.barrier_embedding,

            model.exec.base.cross_expert,
            model.exec.base.market_state,
            model.exec.base.side_towers,
            model.exec.base.race_decoder,
            model.exec.base.time_decoder,
            model.exec.base.excursion_decoder,
            model.exec.base.side_embedding,
            model.exec.base.horizon_embedding,
            model.exec.base.barrier_embedding,
        ]

        for module in modules:
            for p in module.parameters():
                p.requires_grad = True

    if phase >= 3:
        for p in (
            model.exec
            .base
            .experts
            .parameters()
        ):
            p.requires_grad = True


def make_optimizer(
    model,
):
    groups = {
        "daily": [],
        "exec": [],
        "upper": [],
        "experts": [],
    }

    for name, p in (
        model.named_parameters()
    ):
        if not p.requires_grad:
            continue

        if name.startswith(
            "exec.base.experts."
        ):
            groups[
                "experts"
            ].append(
                p
            )

        elif name.startswith(
            "exec.base."
        ):
            groups[
                "upper"
            ].append(
                p
            )

        elif name.startswith(
            "exec."
        ):
            groups[
                "exec"
            ].append(
                p
            )

        else:
            groups[
                "daily"
            ].append(
                p
            )

    params = []

    mapping = {
        "daily":
            LR_DAILY,

        "exec":
            LR_EXEC_HEAD,

        "upper":
            LR_UPPER,

        "experts":
            LR_EXPERTS,
    }

    for key, values in (
        groups.items()
    ):
        if not values:
            continue

        params.append(
            {
                "params":
                    values,

                "lr":
                    mapping[
                        key
                    ],
            }
        )

    return torch.optim.AdamW(
        params,
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# TARGET GATHERING
# ============================================================

def gather_daily(
    row_ids,
    arrays,
    daily,
    device,
):
    source = arrays[
        "source"
    ][
        row_ids
    ]

    def tensor(
        x,
        dtype=torch.float32,
    ):
        return torch.from_numpy(
            x
        ).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

    top = tensor(
        daily[
            "top"
        ][
            source
        ]
    )

    rank = tensor(
        daily[
            "rank"
        ][
            source
        ]
    )

    best_net = tensor(
        np.clip(
            daily[
                "best_net"
            ][
                source
            ]
            / BEST_NET_SCALE,
            -2.0,
            2.0,
        ).astype(
            np.float32
        )
    )

    positive_count = tensor(
        (
            daily[
                "positive_count"
            ][
                source
            ]
            / 18.0
        ).astype(
            np.float32
        )
    )

    side = tensor(
        daily[
            "side"
        ][
            source
        ].astype(
            np.float32
        )
    )

    direction_gap = tensor(
        daily[
            "direction_gap"
        ][
            source
        ]
    )

    day_ns = tensor(
        daily[
            "day_ns"
        ][
            source
        ],
        dtype=torch.long,
    )

    return {
        "top":
            top,

        "rank":
            rank,

        "best_net":
            best_net,

        "count":
            positive_count,

        "side":
            side,

        "direction_gap":
            direction_gap,

        "day_ns":
            day_ns,
    }


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_pos_weights(
    train_rows,
    arrays,
    daily,
    device,
):
    source = arrays[
        "source"
    ][
        train_rows
    ]

    y = daily[
        "top"
    ][
        source
    ]

    pos = y.sum(
        axis=0
    )

    neg = (
        len(
            y
        )
        - pos
    )

    weight = (
        neg
        / np.maximum(
            pos,
            1.0,
        )
    )

    # Extreme weights make probability
    # calibration unstable. Ranking is
    # more important than raw calibration.
    weight = np.clip(
        weight,
        1.0,
        20.0,
    ).astype(
        np.float32
    )

    print(
        "Ordinal positive weights:",
        {
            TOP_NAMES[i]:
                float(
                    weight[i]
                )
            for i in range(
                len(
                    TOP_NAMES
                )
            )
        },
    )

    return torch.from_numpy(
        weight
    ).to(
        device
    )


# ============================================================
# DAY-GROUPED TRAIN ORDER
# ============================================================

def day_grouped_order(
    rows,
    arrays,
    daily,
    epoch,
):
    source = arrays[
        "source"
    ][
        rows
    ]

    days = daily[
        "day_ns"
    ][
        source
    ]

    unique_days, inverse = (
        np.unique(
            days,
            return_inverse=True,
        )
    )

    rng = np.random.default_rng(
        SEED
        + 1009
        * epoch
    )

    perm = rng.permutation(
        len(
            unique_days
        )
    )

    day_order = np.empty(
        len(
            unique_days
        ),
        dtype=np.int64,
    )

    day_order[
        perm
    ] = np.arange(
        len(
            unique_days
        ),
        dtype=np.int64,
    )

    # Primary sort = shuffled day.
    # Secondary sort = actual source row.
    idx = np.lexsort(
        (
            source,
            day_order[
                inverse
            ],
        )
    )

    return rows[
        idx
    ]


# ============================================================
# CROSS-TIME RANK LOSS
# ============================================================

def daily_pairwise_loss(
    score,
    daily_rank,
    best_net_norm,
    day_ns,
):
    losses = []

    unique_days = torch.unique(
        day_ns
    )

    for d in unique_days:

        mask = (
            day_ns == d
        )

        if (
            mask.sum()
            < 8
        ):
            continue

        r = daily_rank[
            mask
        ]

        n = best_net_norm[
            mask
        ]

        s = score[
            mask
        ]

        high = (
            (r >= 0.90)
            & (n > 0)
        )

        low = (
            r <= 0.70
        )

        if (
            not high.any()
            or not low.any()
        ):
            continue

        high_score = s[
            high
        ].mean()

        # Hard negative emphasis.
        low_score = torch.logsumexp(
            s[
                low
            ],
            dim=0,
        ) - torch.log(
            torch.tensor(
                float(
                    low.sum()
                ),
                device=s.device,
            )
        )

        losses.append(
            F.softplus(
                0.50
                - (
                    high_score
                    - low_score
                )
            )
        )

    if not losses:
        return (
            score.sum()
            * 0.0
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# DAILY LOSS
# ============================================================

def compute_daily_loss(
    output,
    target,
    pos_weight,
):
    logits = output[
        "ordinal_logits"
    ]

    top = target[
        "top"
    ]

    ordinal = (
        F.binary_cross_entropy_with_logits(
            logits.float(),
            top.float(),
            pos_weight=pos_weight,
        )
    )

    probability = torch.sigmoid(
        logits.float()
    )

    # top2 <= top5 <= top10 <= top20
    monotonic = (
        F.relu(
            probability[
                :,
                1
            ]
            - probability[
                :,
                0
            ]
        ).mean()

        + F.relu(
            probability[
                :,
                2
            ]
            - probability[
                :,
                1
            ]
        ).mean()

        + F.relu(
            probability[
                :,
                3
            ]
            - probability[
                :,
                2
            ]
        ).mean()
    )

    rank_pred = torch.sigmoid(
        output[
            "rank_logit"
        ].float()
    )

    finite_rank = torch.isfinite(
        target[
            "rank"
        ]
    )

    if finite_rank.any():
        rank_loss = F.smooth_l1_loss(
            rank_pred[
                finite_rank
            ],
            target[
                "rank"
            ][
                finite_rank
            ],
            beta=0.10,
        )
    else:
        rank_loss = (
            rank_pred.sum()
            * 0.0
        )

    best_net_loss = (
        F.smooth_l1_loss(
            output[
                "best_net_norm"
            ].float(),
            target[
                "best_net"
            ].float(),
            beta=0.25,
        )
    )

    count_pred = torch.sigmoid(
        output[
            "count_norm"
        ].float()
    )

    count_loss = (
        F.smooth_l1_loss(
            count_pred,
            target[
                "count"
            ].float(),
            beta=0.10,
        )
    )

    # Ignore direction ties. If LONG and SHORT
    # are nearly identical economically, forcing
    # a binary label only adds noise.
    side_mask = (
        torch.isfinite(
            target[
                "direction_gap"
            ]
        )
        & (
            torch.abs(
                target[
                    "direction_gap"
                ]
            )
            >= 1.0
        )
        & (
            target[
                "side"
            ]
            >= 0
        )
    )

    if side_mask.any():
        side_loss = (
            F.binary_cross_entropy_with_logits(
                output[
                    "side_logit"
                ].float()[
                    side_mask
                ],
                target[
                    "side"
                ].float()[
                    side_mask
                ],
            )
        )
    else:
        side_loss = (
            output[
                "side_logit"
            ].float().sum()
            * 0.0
        )

    # One scalar used for cross-time ordering.
    opportunity_score = (
        0.10
        * logits[
            :,
            0
        ].float()

        + 0.30
        * logits[
            :,
            1
        ].float()

        + 0.35
        * logits[
            :,
            2
        ].float()

        + 0.25
        * logits[
            :,
            3
        ].float()

        + 0.20
        * output[
            "rank_logit"
        ].float()
    )

    pairwise = daily_pairwise_loss(
        opportunity_score,
        target[
            "rank"
        ],
        target[
            "best_net"
        ],
        target[
            "day_ns"
        ],
    )

    total = (
        1.00
        * ordinal

        + 0.25
        * monotonic

        + 0.35
        * rank_loss

        + 0.30
        * best_net_loss

        + 0.15
        * count_loss

        + 0.15
        * side_loss

        + 0.75
        * pairwise
    )

    parts = {
        "daily_total":
            total,

        "ordinal":
            ordinal,

        "mono":
            monotonic,

        "rank":
            rank_loss,

        "bestnet":
            best_net_loss,

        "count":
            count_loss,

        "side":
            side_loss,

        "pair":
            pairwise,
    }

    return (
        total,
        parts,
    )


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    optimizer,
    rows,
    epoch,
    arrays,
    mean,
    std,
    daily,
    execution_targets,
    pos_weight,
    device,
    scaler,
):
    model.train()

    order = day_grouped_order(
        rows,
        arrays,
        daily,
        epoch,
    )

    loader = brain.make_loader(
        order,
        False,
        arrays,
        mean,
        std,
    )

    cursor = 0

    keys = [
        "total",
        "daily_total",
        "ordinal",
        "mono",
        "rank",
        "bestnet",
        "count",
        "side",
        "pair",
        "exec",
    ]

    sums = {
        k: 0.0
        for k in keys
    }

    batches = 0

    amp_enabled = (
        device.type
        == "cuda"
    )

    for batch in loader:

        x = batch[
            0
        ].to(
            device,
            non_blocking=True,
        )

        race_true = batch[
            1
        ].to(
            device,
            non_blocking=True,
        )

        b = x.shape[
            0
        ]

        batch_rows = order[
            cursor:
            cursor + b
        ]

        cursor += b

        dt = gather_daily(
            batch_rows,
            arrays,
            daily,
            device,
        )

        (
            valid,
            win05,
            win10,
            net05_norm,
        ) = v671.gather_targets(
            batch_rows,
            arrays,
            execution_targets,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            output = model(
                x
            )

            daily_loss, parts = (
                compute_daily_loss(
                    output,
                    dt,
                    pos_weight,
                )
            )

            exec_loss, _ = (
                v671.compute_loss(
                    output[
                        "exec"
                    ],
                    valid,
                    win05,
                    win10,
                    net05_norm,
                    race_true,
                )
            )

            # Execution loss is auxiliary.
            # Main objective is now WHEN.
            loss = (
                daily_loss
                + 0.25
                * exec_loss
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            GRAD_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        sums[
            "total"
        ] += float(
            loss.detach()
        )

        sums[
            "exec"
        ] += float(
            exec_loss.detach()
        )

        for key, value in (
            parts.items()
        ):
            sums[
                key
            ] += float(
                value.detach()
            )

        batches += 1

    if cursor != len(
        order
    ):
        raise RuntimeError(
            "Training row alignment failure."
        )

    return {
        k:
            v
            / max(
                batches,
                1,
            )
        for k, v
        in sums.items()
    }


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    rows,
    arrays,
    mean,
    std,
    device,
):
    model.eval()

    loader = brain.make_loader(
        rows,
        False,
        arrays,
        mean,
        std,
    )

    ordinal = []
    rank = []
    best_net = []
    side = []

    p05 = []
    exec_net = []

    total = 0

    for batch in loader:

        x = batch[
            0
        ].to(
            device,
            non_blocking=True,
        )

        output = model(
            x
        )

        ordinal.append(
            torch.sigmoid(
                output[
                    "ordinal_logits"
                ]
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        rank.append(
            torch.sigmoid(
                output[
                    "rank_logit"
                ]
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        best_net.append(
            (
                output[
                    "best_net_norm"
                ]
                * BEST_NET_SCALE
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        side.append(
            torch.sigmoid(
                output[
                    "side_logit"
                ]
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        p05.append(
            torch.sigmoid(
                output[
                    "exec"
                ][
                    "win05_logit"
                ]
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        exec_net.append(
            (
                output[
                    "exec"
                ][
                    "net05_norm"
                ]
                * v671.NET_SCALE
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        total += (
            x.shape[
                0
            ]
        )

    if total != len(
        rows
    ):
        raise RuntimeError(
            "Prediction alignment failure."
        )

    ordinal = np.concatenate(
        ordinal,
        axis=0,
    )

    rank = np.concatenate(
        rank,
        axis=0,
    )

    # Probability-based score.
    # No absolute threshold assumption.
    score = (
        0.10
        * ordinal[
            :,
            0
        ]

        + 0.30
        * ordinal[
            :,
            1
        ]

        + 0.35
        * ordinal[
            :,
            2
        ]

        + 0.25
        * ordinal[
            :,
            3
        ]

        + 0.20
        * rank
    ).astype(
        np.float32
    )

    return {
        "ordinal":
            ordinal,

        "rank":
            rank,

        "score":
            score,

        "best_net":
            np.concatenate(
                best_net
            ),

        "side":
            np.concatenate(
                side
            ),

        "p05":
            np.concatenate(
                p05,
                axis=0,
            ),

        "exec_net":
            np.concatenate(
                exec_net,
                axis=0,
            ),
    }


# ============================================================
# BINARY METRIC
# ============================================================

def binary_metric(
    y,
    p,
):
    y = np.asarray(
        y,
        np.uint8,
    )

    p = np.asarray(
        p,
        np.float64,
    )

    ap = float(
        average_precision_score(
            y,
            p,
        )
    )

    auc = (
        float(
            roc_auc_score(
                y,
                p,
            )
        )
        if y.min() != y.max()
        else np.nan
    )

    return {
        "base":
            float(
                y.mean()
            ),

        "ap":
            ap,

        "auc":
            auc,

        "gain":
            float(
                ap
                - y.mean()
            ),
    }


def profit_factor(
    pnl,
):
    pnl = np.asarray(
        pnl,
        np.float64,
    )

    gains = pnl[
        pnl > 0
    ].sum()

    losses = -pnl[
        pnl < 0
    ].sum()

    if losses <= 0:
        return np.inf

    return float(
        gains
        / losses
    )


# ============================================================
# CAUSAL ONE-TRADE-PER-DAY FRONTIER
# ============================================================

def make_thresholds(
    score,
):
    return {
        float(q):
            float(
                np.quantile(
                    score,
                    q,
                )
            )
        for q in ENTRY_QUANTILES
    }


def causal_frontier(
    pred,
    rows,
    arrays,
    daily,
    execution_targets,
    thresholds=None,
):
    source = arrays[
        "source"
    ][
        rows
    ]

    day = daily[
        "day_ns"
    ][
        source
    ]

    score = pred[
        "score"
    ]

    p05 = pred[
        "p05"
    ]

    exec_net_pred = pred[
        "exec_net"
    ]

    valid = execution_targets[
        "valid"
    ][
        source
    ]

    gross = execution_targets[
        "gross"
    ][
        source
    ]

    # Small predicted-net tie breaker.
    task_score = (
        p05
        + 0.01
        * np.tanh(
            exec_net_pred
            / 10.0
        )
    )

    task_score[
        ~valid
    ] = -np.inf

    chosen_task = np.argmax(
        task_score,
        axis=1,
    )

    idx = np.arange(
        len(
            rows
        )
    )

    chosen_net = (
        gross[
            idx,
            chosen_task
        ]
        - 0.5
    )

    counts = pd.Series(
        day
    ).value_counts()

    eligible_days = set(
        counts[
            counts
            >= MIN_DAY_BARS
        ].index
    )

    if thresholds is None:
        thresholds = (
            make_thresholds(
                score
            )
        )

    results = []

    for q in ENTRY_QUANTILES:

        threshold = thresholds[
            float(
                q
            )
        ]

        used = set()

        pnl = []
        task_ids = []

        for i in range(
            len(
                rows
            )
        ):
            d = int(
                day[
                    i
                ]
            )

            if (
                d
                not in eligible_days
            ):
                continue

            if d in used:
                continue

            if (
                score[
                    i
                ]
                < threshold
            ):
                continue

            used.add(
                d
            )

            pnl.append(
                float(
                    chosen_net[
                        i
                    ]
                )
            )

            task_ids.append(
                int(
                    chosen_task[
                        i
                    ]
                )
            )

        pnl = np.asarray(
            pnl,
            np.float64,
        )

        n = len(
            pnl
        )

        coverage = (
            n
            / len(
                eligible_days
            )
            if eligible_days
            else np.nan
        )

        results.append(
            {
                "quantile":
                    q,

                "threshold":
                    threshold,

                "days":
                    len(
                        eligible_days
                    ),

                "trades":
                    n,

                "coverage":
                    coverage,

                "win":
                    (
                        float(
                            (
                                pnl > 0
                            ).mean()
                        )
                        if n
                        else np.nan
                    ),

                "mean_net":
                    (
                        float(
                            pnl.mean()
                        )
                        if n
                        else np.nan
                    ),

                "pf":
                    (
                        profit_factor(
                            pnl
                        )
                        if n
                        else np.nan
                    ),
            }
        )

    return (
        pd.DataFrame(
            results
        ),
        thresholds,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    rows,
    arrays,
    mean,
    std,
    daily,
    execution_targets,
    device,
    label,
    frozen_thresholds=None,
):
    pred = predict(
        model,
        rows,
        arrays,
        mean,
        std,
        device,
    )

    source = arrays[
        "source"
    ][
        rows
    ]

    truth = daily[
        "top"
    ][
        source
    ]

    metric_rows = []

    for j, name in enumerate(
        TOP_NAMES
    ):
        m = binary_metric(
            truth[
                :,
                j
            ],
            pred[
                "ordinal"
            ][
                :,
                j
            ],
        )

        metric_rows.append(
            {
                "target":
                    name,

                **m,
            }
        )

    metric_df = pd.DataFrame(
        metric_rows
    )

    frontier, thresholds = (
        causal_frontier(
            pred,
            rows,
            arrays,
            daily,
            execution_targets,
            frozen_thresholds,
        )
    )

    print()
    print(
        f"{label} DAILY OPPORTUNITY METRICS"
    )

    print(
        "-" * 120
    )

    print(
        metric_df.to_string(
            index=False,
            formatters={
                "base":
                    lambda x:
                        f"{x:.4f}",

                "ap":
                    lambda x:
                        f"{x:.4f}",

                "auc":
                    lambda x:
                        f"{x:.4f}",

                "gain":
                    lambda x:
                        f"{x:+.4f}",
            },
        )
    )

    print()
    print(
        f"{label} CAUSAL FRONTIER "
        "(one trade max / trading day)"
    )

    print(
        "-" * 120
    )

    print(
        frontier.to_string(
            index=False,
            formatters={
                "coverage":
                    lambda x:
                        f"{x:.2%}",

                "win":
                    lambda x:
                        (
                            "nan"
                            if not np.isfinite(
                                x
                            )
                            else f"{x:.2%}"
                        ),

                "mean_net":
                    lambda x:
                        (
                            "nan"
                            if not np.isfinite(
                                x
                            )
                            else f"{x:+.3f}"
                        ),

                "pf":
                    lambda x:
                        (
                            "nan"
                            if not np.isfinite(
                                x
                            )
                            else f"{x:.3f}"
                        ),
            },
        )
    )

    top10 = metric_df[
        metric_df[
            "target"
        ]
        == "TOP10"
    ].iloc[
        0
    ]

    top5 = metric_df[
        metric_df[
            "target"
        ]
        == "TOP5"
    ].iloc[
        0
    ]

    eligible = frontier[
        frontier[
            "coverage"
        ]
        >= 0.80
    ].copy()

    if len(
        eligible
    ):
        eligible[
            "daily_quality"
        ] = (
            1.7
            * eligible[
                "win"
            ]

            + 0.5
            * eligible[
                "coverage"
            ]

            + 0.15
            * np.tanh(
                eligible[
                    "mean_net"
                ]
                / 5.0
            )
        )

        best = eligible.sort_values(
            [
                "daily_quality",
                "pf",
            ],
            ascending=False,
        ).iloc[
            0
        ]

        daily_quality = float(
            best[
                "daily_quality"
            ]
        )

    else:
        best = None
        daily_quality = 0.0

    selection = (
        0.20
        * float(
            top10[
                "ap"
            ]
        )

        + 0.10
        * float(
            top10[
                "auc"
            ]
        )

        + 0.20
        * float(
            top5[
                "ap"
            ]
        )

        + 0.10
        * float(
            top5[
                "auc"
            ]
        )

        + 0.40
        * daily_quality
    )

    print()
    print(
        "Selection score:",
        f"{selection:.6f}",
    )

    if best is not None:
        print(
            "Best >=80% coverage point:",
            best.to_dict(),
        )

    return {
        "selection":
            float(
                selection
            ),

        "metrics":
            metric_df,

        "frontier":
            frontier,

        "thresholds":
            thresholds,

        "pred":
            pred,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    started = time.time()

    seed_everything()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.2 DAILY OPPORTUNITY BRAIN"
    )

    print(
        "=" * 130
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    ) = brain.load_data()

    execution_targets = (
        v671.load_execution_targets()
    )

    daily = (
        load_daily_targets()
    )

    source_all = arrays[
        "source"
    ]

    years = daily[
        "year"
    ][
        source_all
    ]

    train_rows = split[
        "train"
    ]

    val_rows = split[
        "val"
    ]

    rows2023 = val_rows[
        years[
            val_rows
        ]
        == 2023
    ]

    rows2024 = val_rows[
        years[
            val_rows
        ]
        == 2024
    ]

    if not np.all(
        years[
            train_rows
        ]
        <= 2022
    ):
        raise RuntimeError(
            "Post-2022 row detected in training."
        )

    print(
        "Train 2016-2022:",
        f"{len(train_rows):,}",
    )

    print(
        "Validation 2023:",
        f"{len(rows2023):,}",
    )

    print(
        "Frozen benchmark 2024:",
        f"{len(rows2024):,}",
    )

    print(
        "2025 NOT USED:",
        f"{len(split['test2025']):,}",
    )

    print(
        "2026 NOT USED:",
        f"{len(split['reserved2026']):,}",
    )

    # --------------------------------------------------------
    # Base V6.6.1
    # --------------------------------------------------------

    base = (
        brain.MultiSurfaceTechnicalBrain(
            groups
        )
        .to(
            device
        )
    )

    old = torch.load(
        execmod.CKPT,
        map_location=device,
        weights_only=False,
    )

    base.load_state_dict(
        old[
            "model"
        ]
    )

    # --------------------------------------------------------
    # Execution V6.7.1 initialization
    # --------------------------------------------------------

    execution_brain = (
        v671.ExecutionPrecisionBrainV671(
            base
        )
        .to(
            device
        )
    )

    if not V671_CHAMPION.exists():
        raise RuntimeError(
            "Missing V6.7.1 champion."
        )

    c671 = torch.load(
        V671_CHAMPION,
        map_location=device,
        weights_only=False,
    )

    execution_brain.load_state_dict(
        c671[
            "model"
        ]
    )

    print(
        "Initialized execution brain "
        f"from V6.7.1 epoch {c671['epoch']}"
    )

    model = (
        DailyOpportunityBrainV672(
            execution_brain
        )
        .to(
            device
        )
    )

    pos_weight = (
        compute_pos_weights(
            train_rows,
            arrays,
            daily,
            device,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    )

    best_selection = -np.inf
    best_epoch = None
    stale = 0

    history = []

    current_phase = None
    optimizer = None

    for epoch in range(
        1,
        TOTAL_EPOCHS + 1,
    ):

        if epoch <= PHASE1_EPOCHS:
            phase = 1

        elif (
            epoch
            <= PHASE1_EPOCHS
            + PHASE2_EPOCHS
        ):
            phase = 2

        else:
            phase = 3

        if (
            phase
            != current_phase
        ):
            current_phase = phase

            set_phase(
                model,
                phase,
            )

            optimizer = make_optimizer(
                model
            )

            trainable = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            print()
            print(
                "=" * 130
            )

            print(
                f"ENTER TRAINING PHASE {phase}"
            )

            if phase == 1:
                print(
                    "DAILY HEAD ONLY"
                )

            elif phase == 2:
                print(
                    "DAILY + EXECUTION HEAD "
                    "+ UPPER BACKBONE"
                )

            else:
                print(
                    "FULL NETWORK FINE-TUNE "
                    "WITH LOW-LR EXPERTS"
                )

            print(
                "Trainable parameters:",
                f"{trainable:,}",
            )

            print(
                "=" * 130
            )

        print()
        print(
            f"EPOCH {epoch}/{TOTAL_EPOCHS}"
        )

        train_stats = train_epoch(
            model,
            optimizer,
            train_rows,
            epoch,
            arrays,
            mean,
            std,
            daily,
            execution_targets,
            pos_weight,
            device,
            scaler,
        )

        print(
            "TRAIN "
            + " ".join(
                [
                    f"{k}={v:.5f}"
                    for k, v in (
                        train_stats.items()
                    )
                ]
            )
        )

        val = evaluate(
            model,
            rows2023,
            arrays,
            mean,
            std,
            daily,
            execution_targets,
            device,
            "2023 VALIDATION",
        )

        row = {
            "epoch":
                epoch,

            "phase":
                phase,

            "selection":
                val[
                    "selection"
                ],
        }

        for _, r in (
            val[
                "metrics"
            ].iterrows()
        ):
            name = str(
                r[
                    "target"
                ]
            )

            row[
                f"{name}_ap"
            ] = float(
                r[
                    "ap"
                ]
            )

            row[
                f"{name}_auc"
            ] = float(
                r[
                    "auc"
                ]
            )

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            OUT
            / "training_history_v672.csv",
            index=False,
        )

        if (
            val[
                "selection"
            ]
            > best_selection
        ):
            best_selection = (
                val[
                    "selection"
                ]
            )

            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "version":
                        VERSION,

                    "epoch":
                        epoch,

                    "phase":
                        phase,

                    "selection":
                        best_selection,

                    "model":
                        model.state_dict(),

                    "thresholds_2023":
                        val[
                            "thresholds"
                        ],

                    "feature_count":
                        len(
                            names
                        ),
                },
                CHAMPION,
            )

            val[
                "metrics"
            ].to_csv(
                OUT
                / "champion_2023_metrics.csv",
                index=False,
            )

            val[
                "frontier"
            ].to_csv(
                OUT
                / "champion_2023_frontier.csv",
                index=False,
            )

            print()
            print(
                "*** NEW V6.7.2 CHAMPION ***"
            )

            print(
                "Epoch:",
                epoch,
            )

            print(
                "Selection:",
                f"{best_selection:.6f}",
            )

        else:
            stale += 1

            print(
                f"No improvement. "
                f"stale={stale}/{PATIENCE}"
            )

        # Do not early stop before experts
        # have been allowed to adapt.
        if (
            phase == 3
            and stale >= PATIENCE
        ):
            print(
                "Early stopping."
            )
            break

    if not CHAMPION.exists():
        raise RuntimeError(
            "No V6.7.2 champion saved."
        )

    # ========================================================
    # FROZEN CHAMPION
    # ========================================================

    champion = torch.load(
        CHAMPION,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        champion[
            "model"
        ]
    )

    print()
    print(
        "=" * 130
    )

    print(
        "FROZEN V6.7.2 CHAMPION"
    )

    print(
        "=" * 130
    )

    print(
        "Epoch:",
        champion[
            "epoch"
        ],
    )

    print(
        "Phase:",
        champion[
            "phase"
        ],
    )

    print(
        "Selection:",
        champion[
            "selection"
        ],
    )

    final23 = evaluate(
        model,
        rows2023,
        arrays,
        mean,
        std,
        daily,
        execution_targets,
        device,
        "FINAL 2023",
    )

    # Freeze 2023 score thresholds before
    # opening 2024.
    frozen_thresholds = (
        champion[
            "thresholds_2023"
        ]
    )

    # PyTorch serialization can preserve
    # keys as floats, but normalize anyway.
    frozen_thresholds = {
        float(
            k
        ):
            float(
                v
            )
        for k, v in (
            frozen_thresholds.items()
        )
    }

    print()
    print(
        "=" * 130
    )

    print(
        "OPENING 2024 ONLY AFTER "
        "V6.7.2 CHAMPION FREEZE"
    )

    print(
        "=" * 130
    )

    final24 = evaluate(
        model,
        rows2024,
        arrays,
        mean,
        std,
        daily,
        execution_targets,
        device,
        "2024 FROZEN BENCHMARK",
        frozen_thresholds=frozen_thresholds,
    )

    final24[
        "metrics"
    ].to_csv(
        OUT
        / "frozen_2024_metrics.csv",
        index=False,
    )

    final24[
        "frontier"
    ].to_csv(
        OUT
        / "frozen_2024_frontier.csv",
        index=False,
    )

    summary = {
        "version":
            VERSION,

        "champion_epoch":
            int(
                champion[
                    "epoch"
                ]
            ),

        "champion_phase":
            int(
                champion[
                    "phase"
                ]
            ),

        "selection_2023":
            float(
                champion[
                    "selection"
                ]
            ),

        "2025_evaluated":
            False,

        "2026_evaluated":
            False,
    }

    with open(
        OUT
        / "summary_v672.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 130
    )

    print(
        "TEN V6.7.2 TRAINING COMPLETE"
    )

    print(
        "=" * 130
    )

    print(
        "Champion epoch:",
        champion[
            "epoch"
        ],
    )

    print(
        "Champion phase:",
        champion[
            "phase"
        ],
    )

    print(
        "2025: NOT EVALUATED"
    )

    print(
        "2026: NOT EVALUATED"
    )

    print(
        "Saved:",
        OUT,
    )

    print(
        "Runtime:",
        f"{time.time() - started:.2f}s",
    )


if __name__ == "__main__":
    main()
