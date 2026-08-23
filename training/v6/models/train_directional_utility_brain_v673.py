from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
)

import training.v6.models.train_multisurface_technical_brain_v661 as brain
import training.v6.backtests.backtest_surface_policy_v662 as execmod
import training.v6.models.train_execution_precision_brain_v671 as v671
import training.v6.models.train_daily_opportunity_brain_v672 as v672


VERSION = "v6.7.3"

OUT = Path(
    "training/artifacts/v6/"
    "directional_utility_brain_v673"
)

CHAMPION = (
    OUT
    / "champion_v673.pt"
)

DAILY_FILE = Path(
    "training/v6/data_lake/"
    "daily_opportunity_targets_v672/"
    "daily_opportunity_targets_v672.parquet"
)

TOTAL_EPOCHS = 7
PHASE1_END = 2
PHASE2_END = 4
PATIENCE = 3

LR_NEW = 3e-4
LR_EXEC = 6e-5
LR_UPPER = 2e-5
LR_EXPERT = 6e-6

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NET_SCALE = 30.0
SIDE_GAP_MIN = 3.0

AUX_LEVELS = (
    5.0,
    10.0,
    20.0,
)

MIN_DAY_BARS = 100

SEED = 20260823


def seed_all():
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


def load_direction_targets():
    print(
        "Loading directional utility targets ..."
    )

    df = pd.read_parquet(
        DAILY_FILE,
        columns=[
            "source_row",
            "year",
            "best_side_c05",
            "best_long_net_c05",
            "best_short_net_c05",
            "direction_gap_c05",
        ],
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
            "Directional target alignment failure."
        )

    return {
        "year":
            df[
                "year"
            ].to_numpy(
                np.int16
            ),

        "side":
            df[
                "best_side_c05"
            ].to_numpy(
                np.int8
            ),

        "long_net":
            df[
                "best_long_net_c05"
            ].to_numpy(
                np.float32
            ),

        "short_net":
            df[
                "best_short_net_c05"
            ].to_numpy(
                np.float32
            ),

        "gap":
            df[
                "direction_gap_c05"
            ].to_numpy(
                np.float32
            ),
    }


def tensor_output(
    value,
):
    if isinstance(
        value,
        torch.Tensor,
    ):
        return value

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for x in value:
            if isinstance(
                x,
                torch.Tensor,
            ):
                return x

    raise RuntimeError(
        "Hook output is not a tensor."
    )


class DirectionalUtilityBrainV673(
    nn.Module
):
    def __init__(
        self,
        daily_model,
    ):
        super().__init__()

        self.daily = daily_model

        self.side_states = [
            None,
            None,
        ]

        self.daily_latent = None

        self.daily.exec.base.side_towers[
            0
        ].register_forward_hook(
            self._long_hook
        )

        self.daily.exec.base.side_towers[
            1
        ].register_forward_hook(
            self._short_hook
        )

        self.daily.daily_body.register_forward_hook(
            self._daily_hook
        )

        self.long_encoder = nn.Sequential(
            nn.LazyLinear(
                192
            ),
            nn.GELU(),
            nn.LayerNorm(
                192
            ),
            nn.Dropout(
                0.10
            ),

            nn.Linear(
                192,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(
                128
            ),
        )

        self.short_encoder = nn.Sequential(
            nn.LazyLinear(
                192
            ),
            nn.GELU(),
            nn.LayerNorm(
                192
            ),
            nn.Dropout(
                0.10
            ),

            nn.Linear(
                192,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(
                128
            ),
        )

        # utility + P(net>5/10/20)
        self.long_utility = nn.Linear(
            128,
            4,
        )

        self.short_utility = nn.Linear(
            128,
            4,
        )

        # side logit + continuous gap
        self.comparator = nn.Sequential(
            nn.LazyLinear(
                192
            ),
            nn.GELU(),
            nn.LayerNorm(
                192
            ),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                192,
                96,
            ),
            nn.GELU(),
            nn.Linear(
                96,
                2,
            ),
        )

    def _long_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.side_states[
            0
        ] = tensor_output(
            output
        )

    def _short_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.side_states[
            1
        ] = tensor_output(
            output
        )

    def _daily_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.daily_latent = tensor_output(
            output
        )

    def forward(
        self,
        x,
    ):
        self.side_states = [
            None,
            None,
        ]

        self.daily_latent = None

        d = self.daily(
            x
        )

        if (
            self.side_states[0]
            is None
            or self.side_states[1]
            is None
            or self.daily_latent
            is None
        ):
            raise RuntimeError(
                "Directional hooks did not fire."
            )

        e = d[
            "exec"
        ]

        p05 = torch.sigmoid(
            e[
                "win05_logit"
            ]
        )

        predicted_net = e[
            "net05_norm"
        ]

        race = F.softmax(
            e[
                "race_logits"
            ],
            dim=-1,
        )

        market = e[
            "market_state"
        ]

        def context(
            side_id,
        ):
            if side_id == 0:
                sl = slice(
                    0,
                    9,
                )
            else:
                sl = slice(
                    9,
                    18,
                )

            return torch.cat(
                [
                    market,
                    self.side_states[
                        side_id
                    ],
                    self.daily_latent,

                    p05[
                        :,
                        sl
                    ],

                    predicted_net[
                        :,
                        sl
                    ],

                    race[
                        :,
                        sl,
                        :
                    ].flatten(
                        start_dim=1
                    ),
                ],
                dim=1,
            )

        long_context = context(
            0
        )

        short_context = context(
            1
        )

        long_latent = self.long_encoder(
            long_context
        )

        short_latent = self.short_encoder(
            short_context
        )

        long_out = self.long_utility(
            long_latent
        )

        short_out = self.short_utility(
            short_latent
        )

        compare = self.comparator(
            torch.cat(
                [
                    long_latent,
                    short_latent,
                    self.daily_latent,
                    long_out[
                        :,
                        :1
                    ],
                    short_out[
                        :,
                        :1
                    ],
                ],
                dim=1,
            )
        )

        return {
            "side_logit":
                compare[
                    :,
                    0
                ],

            "gap_norm":
                compare[
                    :,
                    1
                ],

            "long_net_norm":
                long_out[
                    :,
                    0
                ],

            "short_net_norm":
                short_out[
                    :,
                    0
                ],

            "long_aux":
                long_out[
                    :,
                    1:
                ],

            "short_aux":
                short_out[
                    :,
                    1:
                ],

            "daily":
                d,
        }


def set_phase(
    model,
    phase,
):
    for p in model.parameters():
        p.requires_grad = True

    # Freeze old network first.
    for p in model.daily.parameters():
        p.requires_grad = False

    if phase >= 2:

        modules = [
            model.daily.daily_body,

            model.daily.exec.precision_decoder,

            model.daily.exec.base.cross_expert,

            model.daily.exec.base.market_state,

            model.daily.exec.base.side_towers,
        ]

        for module in modules:
            for p in module.parameters():
                p.requires_grad = True

    if phase >= 3:
        for p in (
            model.daily
            .exec
            .base
            .experts
            .parameters()
        ):
            p.requires_grad = True


def make_optimizer(
    model,
):
    buckets = {
        "new": [],
        "exec": [],
        "upper": [],
        "expert": [],
    }

    for name, p in (
        model.named_parameters()
    ):
        if not p.requires_grad:
            continue

        if name.startswith(
            "daily.exec.base.experts."
        ):
            buckets[
                "expert"
            ].append(
                p
            )

        elif name.startswith(
            "daily.exec.base."
        ) or name.startswith(
            "daily.daily_body."
        ):
            buckets[
                "upper"
            ].append(
                p
            )

        elif name.startswith(
            "daily.exec."
        ):
            buckets[
                "exec"
            ].append(
                p
            )

        else:
            buckets[
                "new"
            ].append(
                p
            )

    rates = {
        "new":
            LR_NEW,

        "exec":
            LR_EXEC,

        "upper":
            LR_UPPER,

        "expert":
            LR_EXPERT,
    }

    groups = []

    for key, params in (
        buckets.items()
    ):
        if params:
            groups.append(
                {
                    "params":
                        params,

                    "lr":
                        rates[
                            key
                        ],
                }
            )

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY,
    )

def gather_direction(
    rows,
    arrays,
    direction,
    device,
):
    source = arrays["source"][rows]

    def t(x):
        return torch.from_numpy(
            x.astype(np.float32)
        ).to(
            device,
            non_blocking=True,
        )

    long_net = direction[
        "long_net"
    ][source]

    short_net = direction[
        "short_net"
    ][source]

    gap = direction[
        "gap"
    ][source]

    side = direction[
        "side"
    ][source]

    long_aux = np.stack(
        [
            (
                long_net > level
            ).astype(np.float32)
            for level in AUX_LEVELS
        ],
        axis=1,
    )

    short_aux = np.stack(
        [
            (
                short_net > level
            ).astype(np.float32)
            for level in AUX_LEVELS
        ],
        axis=1,
    )

    return {
        "long_net":
            t(
                np.clip(
                    long_net / NET_SCALE,
                    -2,
                    2,
                )
            ),

        "short_net":
            t(
                np.clip(
                    short_net / NET_SCALE,
                    -2,
                    2,
                )
            ),

        "gap":
            t(
                np.clip(
                    gap / NET_SCALE,
                    -2,
                    2,
                )
            ),

        "gap_bps":
            t(gap),

        "side":
            t(side),

        "long_aux":
            t(long_aux),

        "short_aux":
            t(short_aux),
    }


def direction_loss(
    output,
    target,
):
    long_reg = F.smooth_l1_loss(
        output[
            "long_net_norm"
        ].float(),
        target[
            "long_net"
        ].float(),
        beta=0.20,
    )

    short_reg = F.smooth_l1_loss(
        output[
            "short_net_norm"
        ].float(),
        target[
            "short_net"
        ].float(),
        beta=0.20,
    )

    utility_reg = (
        0.5
        * (
            long_reg
            + short_reg
        )
    )

    gap_reg = F.smooth_l1_loss(
        output[
            "gap_norm"
        ].float(),
        target[
            "gap"
        ].float(),
        beta=0.15,
    )

    utility_gap = (
        output[
            "short_net_norm"
        ].float()
        -
        output[
            "long_net_norm"
        ].float()
    )

    consistency = F.smooth_l1_loss(
        utility_gap,
        target[
            "gap"
        ].float(),
        beta=0.15,
    )

    meaningful = (
        torch.abs(
            target[
                "gap_bps"
            ]
        )
        >= SIDE_GAP_MIN
    )

    if meaningful.any():
        raw = (
            F.binary_cross_entropy_with_logits(
                output[
                    "side_logit"
                ].float()[
                    meaningful
                ],
                target[
                    "side"
                ].float()[
                    meaningful
                ],
                reduction="none",
            )
        )

        weight = torch.clamp(
            torch.abs(
                target[
                    "gap_bps"
                ][
                    meaningful
                ]
            )
            / 10.0,
            1.0,
            5.0,
        )

        side_bce = (
            raw
            * weight
        ).sum() / weight.sum()

    else:
        side_bce = (
            output[
                "side_logit"
            ].float().sum()
            * 0.0
        )

    long_aux = (
        F.binary_cross_entropy_with_logits(
            output[
                "long_aux"
            ].float(),
            target[
                "long_aux"
            ].float(),
        )
    )

    short_aux = (
        F.binary_cross_entropy_with_logits(
            output[
                "short_aux"
            ].float(),
            target[
                "short_aux"
            ].float(),
        )
    )

    aux = (
        0.5
        * (
            long_aux
            + short_aux
        )
    )

    total = (
        2.00 * side_bce
        + 0.60 * gap_reg
        + 0.40 * utility_reg
        + 0.35 * consistency
        + 0.25 * aux
    )

    return (
        total,
        {
            "direction":
                total,

            "side":
                side_bce,

            "gap":
                gap_reg,

            "utility":
                utility_reg,

            "consistent":
                consistency,

            "aux":
                aux,
        },
    )


def train_epoch(
    model,
    optimizer,
    rows,
    epoch,
    arrays,
    mean,
    std,
    direction,
    daily_targets,
    execution_targets,
    pos_weight,
    device,
    scaler,
):
    model.train()

    order = v672.day_grouped_order(
        rows,
        arrays,
        daily_targets,
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

    sums = {
        "total": 0.0,
        "direction": 0.0,
        "side": 0.0,
        "gap": 0.0,
        "utility": 0.0,
        "consistent": 0.0,
        "aux": 0.0,
        "when": 0.0,
        "exec": 0.0,
    }

    batches = 0

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

        b = x.shape[0]

        batch_rows = order[
            cursor:
            cursor + b
        ]

        cursor += b

        target = gather_direction(
            batch_rows,
            arrays,
            direction,
            device,
        )

        daily_target = v672.gather_daily(
            batch_rows,
            arrays,
            daily_targets,
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
            enabled=(
                device.type
                == "cuda"
            ),
        ):

            output = model(x)

            dloss, parts = direction_loss(
                output,
                target,
            )

            when_loss, _ = (
                v672.compute_daily_loss(
                    output[
                        "daily"
                    ],
                    daily_target,
                    pos_weight,
                )
            )

            exec_loss, _ = (
                v671.compute_loss(
                    output[
                        "daily"
                    ][
                        "exec"
                    ],
                    valid,
                    win05,
                    win10,
                    net05_norm,
                    race_true,
                )
            )

            loss = (
                dloss
                + 0.08 * when_loss
                + 0.06 * exec_loss
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
            "when"
        ] += float(
            when_loss.detach()
        )

        sums[
            "exec"
        ] += float(
            exec_loss.detach()
        )

        for k, value in parts.items():
            sums[
                k
            ] += float(
                value.detach()
            )

        batches += 1

    if cursor != len(order):
        raise RuntimeError(
            "V6.7.3 row alignment failure."
        )

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

    side_prob = []
    gap = []
    long_net = []
    short_net = []

    when_score = []
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

        output = model(x)

        d = output[
            "daily"
        ]

        ordinal = torch.sigmoid(
            d[
                "ordinal_logits"
            ]
        )

        rank = torch.sigmoid(
            d[
                "rank_logit"
            ]
        )

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
        )

        side_prob.append(
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

        gap.append(
            (
                output[
                    "gap_norm"
                ]
                * NET_SCALE
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        long_net.append(
            (
                output[
                    "long_net_norm"
                ]
                * NET_SCALE
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        short_net.append(
            (
                output[
                    "short_net_norm"
                ]
                * NET_SCALE
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        when_score.append(
            score
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        p05.append(
            torch.sigmoid(
                d[
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
                d[
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

        total += x.shape[0]

    if total != len(rows):
        raise RuntimeError(
            "Prediction alignment failure."
        )

    return {
        "side_prob":
            np.concatenate(
                side_prob
            ),

        "gap":
            np.concatenate(
                gap
            ),

        "long_net":
            np.concatenate(
                long_net
            ),

        "short_net":
            np.concatenate(
                short_net
            ),

        "when":
            np.concatenate(
                when_score
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


def profit_factor(
    pnl,
):
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


def evaluate(
    model,
    rows,
    arrays,
    mean,
    std,
    direction,
    daily_targets,
    execution_targets,
    thresholds,
    device,
    label,
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

    true_side = direction[
        "side"
    ][
        source
    ]

    gap = direction[
        "gap"
    ][
        source
    ]

    print()
    print(
        f"{label} DIRECTION METRICS"
    )

    print(
        "-" * 125
    )

    direction_scores = {}

    for min_gap in (
        1.0,
        3.0,
        5.0,
        10.0,
        20.0,
    ):

        mask = (
            np.abs(
                gap
            )
            >= min_gap
        )

        y = true_side[
            mask
        ]

        p = pred[
            "side_prob"
        ][
            mask
        ]

        acc = (
            (
                p >= 0.5
            )
            == y
        ).mean()

        auc = roc_auc_score(
            y,
            p,
        )

        direction_scores[
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

    task_score = (
        pred[
            "p05"
        ]
        + 0.01
        * np.tanh(
            pred[
                "exec_net"
            ]
            / 10.0
        )
    )

    task_score[
        ~valid
    ] = -np.inf

    chosen_side = (
        pred[
            "side_prob"
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

    long_mask = (
        chosen_side == 0
    )

    short_mask = (
        chosen_side == 1
    )

    if long_mask.any():

        local = np.argmax(
            task_score[
                long_mask,
                :9
            ],
            axis=1,
        )

        chosen_task[
            long_mask
        ] = local

    if short_mask.any():

        local = np.argmax(
            task_score[
                short_mask,
                9:
            ],
            axis=1,
        )

        chosen_task[
            short_mask
        ] = (
            9 + local
        )

    idx = np.arange(
        len(rows)
    )

    selected_net = (
        gross[
            idx,
            chosen_task
        ]
        - 0.5
    )

    day = daily_targets[
        "day_ns"
    ][
        source
    ]

    counts = pd.Series(
        day
    ).value_counts()

    eligible_days = set(
        int(x)
        for x in counts[
            counts >= MIN_DAY_BARS
        ].index
    )

    frontier = []

    for q in v672.ENTRY_QUANTILES:

        threshold = thresholds[
            float(q)
        ]

        used = set()
        pnl = []

        for i in range(
            len(rows)
        ):

            d = int(
                day[i]
            )

            if (
                d
                not in eligible_days
            ):
                continue

            if d in used:
                continue

            if (
                pred[
                    "when"
                ][
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
                    selected_net[i]
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

        frontier.append(
            {
                "q":
                    q,

                "threshold":
                    threshold,

                "coverage":
                    coverage,

                "trades":
                    n,

                "win":
                    win,

                "mean":
                    mean_net,

                "pf":
                    pf,

                "target80":
                    bool(
                        n >= 150
                        and coverage >= 0.80
                        and win >= 0.80
                        and pf >= 1.20
                    ),
            }
        )

    frontier = pd.DataFrame(
        frontier
    )

    print()
    print(
        f"{label} CAUSAL DAILY FRONTIER"
    )

    print(
        "-" * 125
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
                        f"{x:.2%}",

                "mean":
                    lambda x:
                        f"{x:+.3f}",

                "pf":
                    lambda x:
                        f"{x:.3f}",
            },
        )
    )

    robust = frontier[
        frontier[
            "coverage"
        ]
        >= 0.80
    ].copy()

    if len(robust):

        robust[
            "quality"
        ] = (
            2.2
            * robust[
                "win"
            ]

            + 0.35
            * robust[
                "coverage"
            ]

            + 0.15
            * np.tanh(
                robust[
                    "mean"
                ]
                / 5.0
            )

            + 0.10
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

        daily_quality = float(
            best[
                "quality"
            ]
        )

    else:
        best = None
        daily_quality = -1.0

    auc3 = direction_scores[
        3.0
    ][1]

    auc10 = direction_scores[
        10.0
    ][1]

    selection = (
        0.20 * auc3
        + 0.20 * auc10
        + 0.60 * daily_quality
    )

    print()
    print(
        "Selection:",
        f"{selection:.6f}",
    )

    if best is not None:
        print(
            "Best >=80% coverage:",
            best.to_dict(),
        )

    return {
        "selection":
            float(selection),

        "frontier":
            frontier,

        "direction":
            direction_scores,
    }


def build_v672(
    groups,
    device,
):
    base = (
        brain.MultiSurfaceTechnicalBrain(
            groups
        )
        .to(device)
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

    execution = (
        v671.ExecutionPrecisionBrainV671(
            base
        )
        .to(device)
    )

    daily = (
        v672.DailyOpportunityBrainV672(
            execution
        )
        .to(device)
    )

    c672 = torch.load(
        v672.CHAMPION,
        map_location=device,
        weights_only=False,
    )

    daily.load_state_dict(
        c672[
            "model"
        ]
    )

    return (
        daily,
        c672,
    )


def main():
    started = time.time()

    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.3 "
        "DIRECTIONAL UTILITY BRAIN"
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

    daily_targets = (
        v672.load_daily_targets()
    )

    direction = (
        load_direction_targets()
    )

    years = direction[
        "year"
    ][
        arrays[
            "source"
        ]
    ]

    train_rows = split[
        "train"
    ]

    val = split[
        "val"
    ]

    rows23 = val[
        years[
            val
        ]
        == 2023
    ]

    rows24 = val[
        years[
            val
        ]
        == 2024
    ]

    print(
        "Train 2016-2022:",
        f"{len(train_rows):,}",
    )

    print(
        "Validation 2023:",
        f"{len(rows23):,}",
    )

    print(
        "Frozen benchmark 2024:",
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

    daily_model, c672 = (
        build_v672(
            groups,
            device,
        )
    )

    print(
        "Initialized from "
        f"V6.7.2 epoch {c672['epoch']}"
    )

    model = (
        DirectionalUtilityBrainV673(
            daily_model
        )
        .to(device)
    )

    # Initialize LazyLinear layers.
    loader = brain.make_loader(
        train_rows[:32],
        False,
        arrays,
        mean,
        std,
    )

    x = next(
        iter(loader)
    )[0].to(
        device
    )

    with torch.no_grad():
        smoke = model(
            x
        )

    print(
        "Warmup side:",
        tuple(
            smoke[
                "side_logit"
            ].shape
        ),
    )

    print(
        "Warmup long utility:",
        tuple(
            smoke[
                "long_net_norm"
            ].shape
        ),
    )

    thresholds = {
        float(k):
            float(v)
        for k, v in c672[
            "thresholds_2023"
        ].items()
    }

    pos_weight = (
        v672.compute_pos_weights(
            train_rows,
            arrays,
            daily_targets,
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

    current_phase = None
    optimizer = None

    best_selection = -np.inf
    stale = 0
    history = []

    for epoch in range(
        1,
        TOTAL_EPOCHS + 1,
    ):

        if epoch <= PHASE1_END:
            phase = 1

        elif epoch <= PHASE2_END:
            phase = 2

        else:
            phase = 3

        if phase != current_phase:

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
                f"ENTER PHASE {phase}"
            )

            if phase == 1:
                print(
                    "NEW DIRECTIONAL HEADS ONLY"
                )

            elif phase == 2:
                print(
                    "DIRECTION + SIDE TOWERS "
                    "+ UPPER BACKBONE"
                )

            else:
                print(
                    "FULL DIRECTIONAL FINE-TUNE "
                    "WITH LOW-LR EXPERTS"
                )

            print(
                "Trainable:",
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
            direction,
            daily_targets,
            execution_targets,
            pos_weight,
            device,
            scaler,
        )

        print(
            "TRAIN "
            + " ".join(
                f"{k}={v:.5f}"
                for k, v
                in train_stats.items()
            )
        )

        result = evaluate(
            model,
            rows23,
            arrays,
            mean,
            std,
            direction,
            daily_targets,
            execution_targets,
            thresholds,
            device,
            "2023 VALIDATION",
        )

        history.append(
            {
                "epoch":
                    epoch,

                "phase":
                    phase,

                "selection":
                    result[
                        "selection"
                    ],

                "side_auc_gap3":
                    result[
                        "direction"
                    ][
                        3.0
                    ][1],

                "side_auc_gap10":
                    result[
                        "direction"
                    ][
                        10.0
                    ][1],
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            OUT
            / "training_history_v673.csv",
            index=False,
        )

        if (
            result[
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

                    "phase":
                        phase,

                    "selection":
                        best_selection,

                    "model":
                        model.state_dict(),

                    "thresholds_2023":
                        thresholds,
                },
                CHAMPION,
            )

            result[
                "frontier"
            ].to_csv(
                OUT
                / "champion_2023_frontier.csv",
                index=False,
            )

            print()
            print(
                "*** NEW V6.7.3 CHAMPION ***"
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
                f"No improvement "
                f"stale={stale}/{PATIENCE}"
            )

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
            "No V6.7.3 champion."
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
        "FROZEN V6.7.3 CHAMPION"
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

    final23 = evaluate(
        model,
        rows23,
        arrays,
        mean,
        std,
        direction,
        daily_targets,
        execution_targets,
        thresholds,
        device,
        "FINAL 2023",
    )

    print()
    print(
        "=" * 130
    )

    print(
        "OPENING 2024 AFTER "
        "V6.7.3 CHAMPION FREEZE"
    )

    print(
        "=" * 130
    )

    final24 = evaluate(
        model,
        rows24,
        arrays,
        mean,
        std,
        direction,
        daily_targets,
        execution_targets,
        thresholds,
        device,
        "2024 FROZEN BENCHMARK",
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
        / "summary_v673.json",
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
        "TEN V6.7.3 COMPLETE"
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
