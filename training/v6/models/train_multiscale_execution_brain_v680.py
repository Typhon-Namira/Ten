from pathlib import Path
import json
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


VERSION = "v6.8.0"

OUT = Path(
    "training/artifacts/v6/"
    "multiscale_execution_brain_v680"
)

CHAMPION = OUT / "champion_v680.pt"


RECENT_STEPS = 24
RECENT_STRIDE = 1

INTRADAY_STEPS = 96
INTRADAY_STRIDE = 3

REGIME_STEPS = 60
REGIME_STRIDE = 24


BATCH = 128
EPOCHS = 12
PATIENCE = 4

LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

TOKEN_DIM = 96
HIDDEN = 128

NET_SCALE = 30.0

MIN_DAY_BARS = 100

WHEN_QUANTILES = (
    0.00,
    0.25,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
)

CONF_LEVELS = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
)

SEED = 20260823


def seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def max_lookback():
    return max(
        (RECENT_STEPS - 1)
        * RECENT_STRIDE,

        (INTRADAY_STEPS - 1)
        * INTRADAY_STRIDE,

        (REGIME_STEPS - 1)
        * REGIME_STRIDE,
    )


def filter_rows(
    rows,
    arrays,
    execution,
):
    rows = np.asarray(
        rows,
        dtype=np.int64,
    )

    source = arrays[
        "source"
    ][rows]

    keep = (
        source
        >= max_lookback()
    )

    rows = rows[
        keep
    ]

    source = source[
        keep
    ]

    # Immediate 2h context must be continuous.
    ts = (
        pd.to_datetime(
            execution[
                "timestamp"
            ],
            utc=True,
        )
        .astype("int64")
        .to_numpy(
            np.int64
        )
    )

    recent_back = (
        RECENT_STEPS - 1
    )

    continuous = (
        ts[source]
        - ts[
            source
            - recent_back
        ]
        == recent_back
        * 300_000_000_000
    )

    valid_any = execution[
        "valid"
    ][
        source
    ].any(
        axis=1
    )

    return rows[
        continuous
        & valid_any
    ]


class MultiScaleExecutionBrainV680(
    nn.Module
):
    def __init__(
        self,
        feature_dim,
    ):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(
                feature_dim,
                TOKEN_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                TOKEN_DIM
            ),
        )

        self.recent_gru = nn.GRU(
            TOKEN_DIM,
            HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.intraday_gru = nn.GRU(
            TOKEN_DIM,
            HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.regime_gru = nn.GRU(
            TOKEN_DIM,
            HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=0.10,
        )

        self.snapshot = nn.Sequential(
            nn.Linear(
                feature_dim,
                HIDDEN,
            ),
            nn.GELU(),
            nn.LayerNorm(
                HIDDEN
            ),
        )

        fusion_dim = (
            HIDDEN * 4
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                384,
            ),
            nn.GELU(),
            nn.LayerNorm(
                384
            ),
            nn.Dropout(
                0.15
            ),

            nn.Linear(
                384,
                256,
            ),
            nn.GELU(),
            nn.LayerNorm(
                256
            ),
        )

        # Master LONG / SHORT direction.
        self.side_head = nn.Sequential(
            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                1,
            ),
        )

        # Direction separately for
        # H30 / H60 / H120.
        self.horizon_side_head = nn.Sequential(
            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                3,
            ),
        )

        # Economic value of all 18 actions.
        self.net_head = nn.Sequential(
            nn.Linear(
                256,
                192,
            ),
            nn.GELU(),
            nn.Linear(
                192,
                18,
            ),
        )

        self.win05_head = nn.Sequential(
            nn.Linear(
                256,
                192,
            ),
            nn.GELU(),
            nn.Linear(
                192,
                18,
            ),
        )

        self.win10_head = nn.Sequential(
            nn.Linear(
                256,
                192,
            ),
            nn.GELU(),
            nn.Linear(
                192,
                18,
            ),
        )

        # top20/top10/top5/top2
        self.when_head = nn.Sequential(
            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                4,
            ),
        )

        self.rank_head = nn.Sequential(
            nn.Linear(
                256,
                64,
            ),
            nn.GELU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def encode(
        self,
        x,
        gru,
    ):
        z = self.project(x)

        _, h = gru(z)

        return h[-1]

    def forward(
        self,
        recent,
        intraday,
        regime,
    ):
        recent_h = self.encode(
            recent,
            self.recent_gru,
        )

        intraday_h = self.encode(
            intraday,
            self.intraday_gru,
        )

        regime_h = self.encode(
            regime,
            self.regime_gru,
        )

        snap = self.snapshot(
            recent[
                :,
                -1,
                :
            ]
        )

        state = self.fusion(
            torch.cat(
                [
                    recent_h,
                    intraday_h,
                    regime_h,
                    snap,
                ],
                dim=1,
            )
        )

        return {
            "state":
                state,

            "side_logit":
                self.side_head(
                    state
                ).squeeze(-1),

            "horizon_side":
                self.horizon_side_head(
                    state
                ),

            "net_norm":
                self.net_head(
                    state
                ),

            "win05_logit":
                self.win05_head(
                    state
                ),

            "win10_logit":
                self.win10_head(
                    state
                ),

            "when_logits":
                self.when_head(
                    state
                ),

            "rank_logit":
                self.rank_head(
                    state
                ).squeeze(-1),
        }


def make_offsets(
    steps,
    stride,
):
    return np.arange(
        (
            steps - 1
        )
        * stride,
        -1,
        -stride,
        dtype=np.int64,
    )


RECENT_OFFSETS = make_offsets(
    RECENT_STEPS,
    RECENT_STRIDE,
)

INTRADAY_OFFSETS = make_offsets(
    INTRADAY_STEPS,
    INTRADAY_STRIDE,
)

REGIME_OFFSETS = make_offsets(
    REGIME_STEPS,
    REGIME_STRIDE,
)


def context_tensor(
    source,
    offsets,
    features,
    mean_t,
    std_t,
    device,
):
    idx = (
        source[:, None]
        - offsets[None, :]
    )

    raw = np.ascontiguousarray(
        features[idx]
    )

    x = torch.from_numpy(
        raw
    ).to(
        device,
        non_blocking=True,
    )

    x = (
        x - mean_t
    ) / std_t

    return torch.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def make_contexts(
    rows,
    arrays,
    features,
    mean_t,
    std_t,
    device,
):
    source = arrays[
        "source"
    ][rows]

    recent = context_tensor(
        source,
        RECENT_OFFSETS,
        features,
        mean_t,
        std_t,
        device,
    )

    intraday = context_tensor(
        source,
        INTRADAY_OFFSETS,
        features,
        mean_t,
        std_t,
        device,
    )

    regime = context_tensor(
        source,
        REGIME_OFFSETS,
        features,
        mean_t,
        std_t,
        device,
    )

    return (
        recent,
        intraday,
        regime,
        source,
    )


def make_targets(
    source,
    execution,
    daily,
    device,
):
    valid_np = execution[
        "valid"
    ][source]

    gross = execution[
        "gross"
    ][source].astype(
        np.float32
    )

    net05 = (
        gross - 0.5
    )

    net_norm = np.clip(
        net05 / NET_SCALE,
        -2.0,
        2.0,
    ).astype(
        np.float32
    )

    win05 = execution[
        "win05"
    ][source].astype(
        np.float32
    )

    win10 = execution[
        "win10"
    ][source].astype(
        np.float32
    )

    safe = np.where(
        valid_np,
        net05,
        -np.inf,
    )

    best_long = np.max(
        safe[
            :,
            :9
        ],
        axis=1,
    )

    best_short = np.max(
        safe[
            :,
            9:
        ],
        axis=1,
    )

    side = (
        best_short
        > best_long
    ).astype(
        np.float32
    )

    side_gap = (
        best_short
        - best_long
    ).astype(
        np.float32
    )

    horizon_side = np.zeros(
        (
            len(source),
            3,
        ),
        dtype=np.float32,
    )

    horizon_gap = np.zeros(
        (
            len(source),
            3,
        ),
        dtype=np.float32,
    )

    for h in range(3):

        lo = h * 3
        hi = lo + 3

        long_h = np.max(
            safe[
                :,
                lo:hi
            ],
            axis=1,
        )

        short_h = np.max(
            safe[
                :,
                9 + lo:
                9 + hi
            ],
            axis=1,
        )

        horizon_side[
            :,
            h
        ] = (
            short_h
            > long_h
        ).astype(
            np.float32
        )

        horizon_gap[
            :,
            h
        ] = (
            short_h
            - long_h
        )

    top = daily[
        "top"
    ][source].astype(
        np.float32
    )

    rank = daily[
        "rank"
    ][source].astype(
        np.float32
    )

    def t(
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

    return {
        "valid":
            t(
                valid_np,
                torch.bool,
            ),

        "net":
            t(net_norm),

        "net_bps":
            t(net05),

        "win05":
            t(win05),

        "win10":
            t(win10),

        "side":
            t(side),

        "side_gap":
            t(side_gap),

        "horizon_side":
            t(horizon_side),

        "horizon_gap":
            t(horizon_gap),

        "top":
            t(top),

        "rank":
            t(rank),
    }


def weighted_side_loss(
    logit,
    y,
    gap,
    min_gap=3.0,
):
    mask = (
        torch.abs(gap)
        >= min_gap
    )

    if not mask.any():
        return (
            logit.float().sum()
            * 0.0
        )

    raw = (
        F.binary_cross_entropy_with_logits(
            logit.float()[
                mask
            ],
            y.float()[
                mask
            ],
            reduction="none",
        )
    )

    weight = torch.clamp(
        torch.abs(
            gap[
                mask
            ]
        )
        / 10.0,
        1.0,
        8.0,
    )

    return (
        raw
        * weight
    ).sum() / weight.sum()


def listwise_action_loss(
    pred_net_norm,
    target_net_bps,
    valid,
):
    pred = (
        pred_net_norm.float()
        * NET_SCALE
    )

    pred = pred.masked_fill(
        ~valid,
        -1e4,
    )

    target = (
        target_net_bps.float()
        / 5.0
    ).masked_fill(
        ~valid,
        -1e4,
    )

    target_prob = F.softmax(
        target,
        dim=1,
    )

    pred_logprob = F.log_softmax(
        pred / 5.0,
        dim=1,
    )

    return -(
        target_prob
        * pred_logprob
    ).sum(
        dim=1
    ).mean()


def compute_loss(
    out,
    target,
    when_pos_weight,
):
    valid = target[
        "valid"
    ]

    net_loss = (
        F.smooth_l1_loss(
            out[
                "net_norm"
            ].float()[
                valid
            ],
            target[
                "net"
            ].float()[
                valid
            ],
            beta=0.20,
        )
    )

    win05 = (
        F.binary_cross_entropy_with_logits(
            out[
                "win05_logit"
            ].float()[
                valid
            ],
            target[
                "win05"
            ].float()[
                valid
            ],
        )
    )

    win10 = (
        F.binary_cross_entropy_with_logits(
            out[
                "win10_logit"
            ].float()[
                valid
            ],
            target[
                "win10"
            ].float()[
                valid
            ],
        )
    )

    side = weighted_side_loss(
        out[
            "side_logit"
        ],
        target[
            "side"
        ],
        target[
            "side_gap"
        ],
        min_gap=3.0,
    )

    horizon_losses = []

    for h in range(3):

        horizon_losses.append(
            weighted_side_loss(
                out[
                    "horizon_side"
                ][
                    :,
                    h
                ],
                target[
                    "horizon_side"
                ][
                    :,
                    h
                ],
                target[
                    "horizon_gap"
                ][
                    :,
                    h
                ],
                min_gap=3.0,
            )
        )

    horizon_side = torch.stack(
        horizon_losses
    ).mean()

    listwise = listwise_action_loss(
        out[
            "net_norm"
        ],
        target[
            "net_bps"
        ],
        valid,
    )

    pred_long = (
        out[
            "net_norm"
        ][
            :,
            :9
        ]
        .float()
        .max(
            dim=1
        )
        .values
    )

    pred_short = (
        out[
            "net_norm"
        ][
            :,
            9:
        ]
        .float()
        .max(
            dim=1
        )
        .values
    )

    true_gap_norm = torch.clamp(
        target[
            "side_gap"
        ].float()
        / NET_SCALE,
        -2.0,
        2.0,
    )

    utility_direction = (
        F.smooth_l1_loss(
            pred_short
            - pred_long,
            true_gap_norm,
            beta=0.15,
        )
    )

    when = (
        F.binary_cross_entropy_with_logits(
            out[
                "when_logits"
            ].float(),
            target[
                "top"
            ].float(),
            pos_weight=when_pos_weight,
        )
    )

    rank_mask = torch.isfinite(
        target[
            "rank"
        ]
    )

    rank = (
        F.smooth_l1_loss(
            torch.sigmoid(
                out[
                    "rank_logit"
                ].float()[
                    rank_mask
                ]
            ),
            target[
                "rank"
            ].float()[
                rank_mask
            ],
            beta=0.10,
        )
    )

    total = (
        3.00 * side
        + 1.20 * horizon_side
        + 0.85 * net_loss
        + 0.55 * win05
        + 0.25 * win10
        + 0.80 * listwise
        + 0.60 * utility_direction
        + 0.35 * when
        + 0.15 * rank
    )

    return (
        total,
        {
            "total":
                total,

            "side":
                side,

            "hside":
                horizon_side,

            "net":
                net_loss,

            "win05":
                win05,

            "win10":
                win10,

            "list":
                listwise,

            "udir":
                utility_direction,

            "when":
                when,

            "rank":
                rank,
        },
    )


def compute_when_pos_weight(
    train_rows,
    arrays,
    daily,
    device,
):
    source = arrays[
        "source"
    ][train_rows]

    y = daily[
        "top"
    ][source]

    pos = y.sum(
        axis=0
    )

    neg = (
        len(y)
        - pos
    )

    weight = (
        neg
        / np.maximum(
            pos,
            1.0,
        )
    )

    weight = np.clip(
        weight,
        1.0,
        20.0,
    ).astype(
        np.float32
    )

    print(
        "WHEN positive weights:",
        weight.tolist(),
    )

    return torch.from_numpy(
        weight
    ).to(
        device
    )


def profit_factor(pnl):
    pnl = np.asarray(
        pnl,
        np.float64,
    )

    gain = pnl[
        pnl > 0
    ].sum()

    loss = -pnl[
        pnl < 0
    ].sum()

    if loss <= 0:
        return np.inf

    return float(
        gain / loss
    )


def train_epoch(
    model,
    optimizer,
    rows,
    arrays,
    features,
    execution,
    daily,
    mean_t,
    std_t,
    when_pos_weight,
    scaler,
    device,
    rng,
):
    model.train()

    order = rng.permutation(
        rows
    )

    sums = {}
    batches = 0

    for start in range(
        0,
        len(order),
        BATCH,
    ):
        batch_rows = order[
            start:
            start + BATCH
        ]

        (
            recent,
            intraday,
            regime,
            source,
        ) = make_contexts(
            batch_rows,
            arrays,
            features,
            mean_t,
            std_t,
            device,
        )

        target = make_targets(
            source,
            execution,
            daily,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):
            out = model(
                recent,
                intraday,
                regime,
            )

            loss, parts = compute_loss(
                out,
                target,
                when_pos_weight,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        for k, value in parts.items():
            sums[k] = (
                sums.get(
                    k,
                    0.0,
                )
                + float(
                    value.detach()
                )
            )

        batches += 1

    return {
        k:
            value
            / max(
                batches,
                1,
            )
        for k, value
        in sums.items()
    }


@torch.no_grad()
def predict(
    model,
    rows,
    arrays,
    features,
    mean_t,
    std_t,
    device,
):
    model.eval()

    chunks = {
        "side": [],
        "hside": [],
        "net": [],
        "win05": [],
        "when": [],
        "rank": [],
    }

    for start in range(
        0,
        len(rows),
        BATCH,
    ):
        batch_rows = rows[
            start:
            start + BATCH
        ]

        (
            recent,
            intraday,
            regime,
            source,
        ) = make_contexts(
            batch_rows,
            arrays,
            features,
            mean_t,
            std_t,
            device,
        )

        out = model(
            recent,
            intraday,
            regime,
        )

        chunks["side"].append(
            torch.sigmoid(
                out[
                    "side_logit"
                ]
            )
            .cpu()
            .numpy()
        )

        chunks["hside"].append(
            torch.sigmoid(
                out[
                    "horizon_side"
                ]
            )
            .cpu()
            .numpy()
        )

        chunks["net"].append(
            (
                out[
                    "net_norm"
                ]
                * NET_SCALE
            )
            .cpu()
            .numpy()
        )

        chunks["win05"].append(
            torch.sigmoid(
                out[
                    "win05_logit"
                ]
            )
            .cpu()
            .numpy()
        )

        chunks["when"].append(
            torch.sigmoid(
                out[
                    "when_logits"
                ]
            )
            .cpu()
            .numpy()
        )

        chunks["rank"].append(
            torch.sigmoid(
                out[
                    "rank_logit"
                ]
            )
            .cpu()
            .numpy()
        )

    return {
        k:
            np.concatenate(
                v,
                axis=0,
            ).astype(
                np.float32
            )
        for k, v
        in chunks.items()
    }




def evaluate(
    model,
    rows,
    arrays,
    features,
    execution,
    daily,
    mean_t,
    std_t,
    device,
    label,
    frozen_policy=None,
):
    pred = predict(
        model,
        rows,
        arrays,
        features,
        mean_t,
        std_t,
        device,
    )

    source = arrays[
        "source"
    ][rows]

    valid = execution[
        "valid"
    ][source]

    gross = execution[
        "gross"
    ][source]

    actual = np.where(
        valid,
        gross - 0.5,
        -np.inf,
    )

    best_long = np.max(
        actual[
            :,
            :9
        ],
        axis=1,
    )

    best_short = np.max(
        actual[
            :,
            9:
        ],
        axis=1,
    )

    true_side = (
        best_short
        > best_long
    ).astype(
        np.uint8
    )

    gap = (
        best_short
        - best_long
    )

    print()
    print(
        f"{label} DIRECTION"
    )

    print(
        "-" * 125
    )

    direction_metrics = {}

    for min_gap in (
        3.0,
        10.0,
        20.0,
    ):
        mask = (
            np.abs(gap)
            >= min_gap
        )

        auc = roc_auc_score(
            true_side[
                mask
            ],
            pred[
                "side"
            ][
                mask
            ],
        )

        acc = (
            (
                pred[
                    "side"
                ][
                    mask
                ]
                >= 0.5
            )
            == true_side[
                mask
            ]
        ).mean()

        direction_metrics[
            min_gap
        ] = (
            float(acc),
            float(auc),
        )

        print(
            f"|GAP|>={min_gap:>4.0f}bps "
            f"N={mask.sum():>6} "
            f"ACC={acc:>6.2%} "
            f"AUC={auc:.4f}"
        )

    chosen_side = (
        pred[
            "side"
        ]
        >= 0.5
    ).astype(
        np.int8
    )

    chosen_task = np.full(
        len(rows),
        -1,
        dtype=np.int64,
    )

    lm = (
        chosen_side == 0
    )

    sm = (
        chosen_side == 1
    )

    if lm.any():
        chosen_task[lm] = np.argmax(
            pred[
                "net"
            ][
                lm,
                :9
            ],
            axis=1,
        )

    if sm.any():
        chosen_task[sm] = (
            9
            + np.argmax(
                pred[
                    "net"
                ][
                    sm,
                    9:
                ],
                axis=1,
            )
        )

    idx = np.arange(
        len(rows)
    )

    pnl_all = (
        gross[
            idx,
            chosen_task
        ]
        - 0.5
    )

    when_score = (
        0.10
        * pred["when"][:, 0]

        + 0.30
        * pred["when"][:, 1]

        + 0.35
        * pred["when"][:, 2]

        + 0.25
        * pred["when"][:, 3]

        + 0.20
        * pred["rank"]
    )

    confidence = (
        np.abs(
            pred["side"]
            - 0.5
        )
        * 2.0
    )

    day = daily[
        "day_ns"
    ][source]

    counts = pd.Series(
        day
    ).value_counts()

    eligible_days = set(
        int(x)
        for x in counts[
            counts >= MIN_DAY_BARS
        ].index
    )

    if frozen_policy is None:
        thresholds = {
            q:
                float(
                    np.quantile(
                        when_score,
                        q,
                    )
                )
            for q in WHEN_QUANTILES
        }

        policies = [
            (
                q,
                thresholds[q],
                conf,
            )
            for q in WHEN_QUANTILES
            for conf in CONF_LEVELS
        ]

    else:
        policies = [
            (
                frozen_policy[
                    "quantile"
                ],
                frozen_policy[
                    "threshold"
                ],
                frozen_policy[
                    "confidence"
                ],
            )
        ]

    output_rows = []

    for q, threshold, conf_min in policies:
        used = set()
        pnl = []

        for i in range(
            len(rows)
        ):
            d = int(
                day[i]
            )

            if d not in eligible_days:
                continue

            if d in used:
                continue

            if (
                when_score[i]
                < threshold
            ):
                continue

            if (
                confidence[i]
                < conf_min
            ):
                continue

            used.add(d)

            pnl.append(
                float(
                    pnl_all[i]
                )
            )

        pnl = np.asarray(
            pnl,
            np.float64,
        )

        n = len(pnl)

        coverage = (
            n
            / len(
                eligible_days
            )
            if eligible_days
            else np.nan
        )

        win = (
            float(
                (
                    pnl > 0
                ).mean()
            )
            if n
            else np.nan
        )

        mean_net = (
            float(
                pnl.mean()
            )
            if n
            else np.nan
        )

        pf = (
            profit_factor(
                pnl
            )
            if n
            else np.nan
        )

        output_rows.append(
            {
                "quantile":
                    q,

                "threshold":
                    threshold,

                "confidence":
                    conf_min,

                "trades":
                    n,

                "coverage":
                    coverage,

                "win":
                    win,

                "mean":
                    mean_net,

                "pf":
                    pf,
            }
        )

    frontier = pd.DataFrame(
        output_rows
    )

    robust = frontier[
        (
            frontier[
                "coverage"
            ]
            >= 0.80
        )
        & (
            frontier[
                "trades"
            ]
            >= 150
        )
    ].copy()

    if len(robust):
        robust[
            "quality"
        ] = (
            3.00
            * robust[
                "win"
            ]

            + 0.40
            * robust[
                "coverage"
            ]

            + 0.25
            * np.tanh(
                robust[
                    "mean"
                ]
                / 5.0
            )

            + 0.15
            * np.log(
                np.maximum(
                    robust[
                        "pf"
                    ],
                    1e-6,
                )
            )
        )

        best = robust.sort_values(
            "quality",
            ascending=False,
        ).iloc[0]

        quality = float(
            best[
                "quality"
            ]
        )

        best_policy = {
            "quantile":
                float(
                    best[
                        "quantile"
                    ]
                ),

            "threshold":
                float(
                    best[
                        "threshold"
                    ]
                ),

            "confidence":
                float(
                    best[
                        "confidence"
                    ]
                ),
        }

    else:
        best = None
        best_policy = None
        quality = -1.0

    auc10 = direction_metrics[
        10.0
    ][1]

    selection = (
        0.85 * quality
        + 0.15 * auc10
    )

    print()
    print(
        f"{label} DAILY EXECUTION"
    )

    print(
        "-" * 125
    )

    if best is not None:
        print(
            "BEST >=80% COVERAGE:",
            best.to_dict(),
        )

        print(
            "TARGET 80 STATUS:",
            bool(
                best["win"] >= 0.80
                and best["mean"] > 0
                and best["pf"] >= 1.20
            ),
        )
    else:
        print(
            "NO >=80% COVERAGE POLICY"
        )

    print(
        "Selection:",
        f"{selection:.6f}",
    )

    return {
        "selection":
            float(selection),

        "policy":
            best_policy,

        "frontier":
            frontier,

        "direction":
            direction_metrics,
    }


def main():
    started = time.time()

    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.8.0 MULTISCALE EXECUTION BRAIN"
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

    execution = (
        v671.load_execution_targets()
    )

    daily = (
        v672.load_daily_targets()
    )

    features = arrays[
        "features"
    ]

    years = execution[
        "year"
    ][
        arrays[
            "source"
        ]
    ]

    train_rows = filter_rows(
        split[
            "train"
        ],
        arrays,
        execution,
    )

    val = split[
        "val"
    ]

    rows23 = filter_rows(
        val[
            years[
                val
            ]
            == 2023
        ],
        arrays,
        execution,
    )

    rows24 = filter_rows(
        val[
            years[
                val
            ]
            == 2024
        ],
        arrays,
        execution,
    )

    print(
        "Train:",
        f"{len(train_rows):,}",
    )

    print(
        "2023 validation:",
        f"{len(rows23):,}",
    )

    print(
        "2024 benchmark:",
        f"{len(rows24):,}",
    )

    print(
        "2025 NOT USED:",
        f"{len(split['test2025']):,}",
    )

    print(
        "2026 NOT USED:",
        f"{len(split['reserved2026']):,}",
    )

    mean_t = torch.from_numpy(
        np.asarray(
            mean,
            dtype=np.float32,
        )
    ).view(
        1,
        1,
        -1,
    ).to(device)

    std_np = np.asarray(
        std,
        dtype=np.float32,
    ).copy()

    std_np[
        std_np < 1e-6
    ] = 1.0

    std_t = torch.from_numpy(
        std_np
    ).view(
        1,
        1,
        -1,
    ).to(device)

    model = MultiScaleExecutionBrainV680(
        features.shape[1]
    ).to(device)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    when_pos_weight = (
        compute_when_pos_weight(
            train_rows,
            arrays,
            daily,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    rng = np.random.default_rng(
        SEED
    )

    best_selection = -np.inf
    stale = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print()
        print(
            "=" * 130
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        stats = train_epoch(
            model,
            optimizer,
            train_rows,
            arrays,
            features,
            execution,
            daily,
            mean_t,
            std_t,
            when_pos_weight,
            scaler,
            device,
            rng,
        )

        print(
            "TRAIN "
            + " ".join(
                f"{k}={v:.5f}"
                for k, v
                in stats.items()
            )
        )

        result = evaluate(
            model,
            rows23,
            arrays,
            features,
            execution,
            daily,
            mean_t,
            std_t,
            device,
            "2023 VALIDATION",
        )

        if (
            result[
                "policy"
            ]
            is not None
            and result[
                "selection"
            ]
            > best_selection
        ):
            best_selection = (
                result[
                    "selection"
                ]
            )

            stale = 0

            torch.save(
                {
                    "version":
                        VERSION,

                    "epoch":
                        epoch,

                    "selection":
                        best_selection,

                    "policy":
                        result[
                            "policy"
                        ],

                    "model":
                        model.state_dict(),
                },
                CHAMPION,
            )

            print(
                "*** NEW V6.8 CHAMPION ***"
            )

        else:
            stale += 1

            print(
                f"No improvement "
                f"stale={stale}/{PATIENCE}"
            )

        if (
            epoch >= 6
            and stale >= PATIENCE
        ):
            print(
                "Early stopping."
            )
            break

    if not CHAMPION.exists():
        raise RuntimeError(
            "No V6.8 champion saved."
        )

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
        "FROZEN V6.8 CHAMPION"
    )

    print(
        "Epoch:",
        champion[
            "epoch"
        ],
    )

    print(
        "Policy:",
        champion[
            "policy"
        ],
    )

    final23 = evaluate(
        model,
        rows23,
        arrays,
        features,
        execution,
        daily,
        mean_t,
        std_t,
        device,
        "FINAL 2023",
        frozen_policy=champion[
            "policy"
        ],
    )

    print()
    print(
        "=" * 130
    )

    print(
        "OPENING 2024 AFTER CHAMPION FREEZE"
    )

    print(
        "=" * 130
    )

    final24 = evaluate(
        model,
        rows24,
        arrays,
        features,
        execution,
        daily,
        mean_t,
        std_t,
        device,
        "2024 FROZEN BENCHMARK",
        frozen_policy=champion[
            "policy"
        ],
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

        "selection_2023":
            float(
                champion[
                    "selection"
                ]
            ),

        "policy":
            champion[
                "policy"
            ],

        "2025_evaluated":
            False,

        "2026_evaluated":
            False,
    }

    with open(
        OUT
        / "summary_v680.json",
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
        "TEN V6.8 TRAINING COMPLETE"
    )

    print(
        "=" * 130
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
        f"{time.time()-started:.2f}s",
    )


if __name__ == "__main__":
    main()
