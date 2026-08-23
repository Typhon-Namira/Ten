from pathlib import Path
import json
import math
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


# ============================================================
# CONFIG
# ============================================================

VERSION = "v6.7.1"

TARGET_FILE = Path(
    "training/v6/data_lake/"
    "execution_aligned_targets_v670/"
    "execution_aligned_targets_v670.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "execution_precision_brain_v671"
)

CHAMPION = (
    OUT
    / "champion_v671.pt"
)

HEAD_ONLY_EPOCHS = 2
TOTAL_EPOCHS = 6
PATIENCE = 3

HEAD_LR = 3e-4
BACKBONE_LR = 3e-5

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

STATE_DIM = 128

EMBED_DIM = 8
TASK_HIDDEN = 128

NET_SCALE = 30.0

DAILY_THRESHOLDS = (
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
)

MIN_DAY_BARS = 100

SELECTION_MIN_COVERAGE = 0.70

SEED = 20260823


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed=SEED,
):
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# TASK META
# ============================================================

TASK_META = []

for j, meta in enumerate(
    brain.TASKS
):
    TASK_META.append(
        {
            "task":
                j,

            "side":
                str(
                    meta["side"]
                ).lower(),

            "side_id":
                int(
                    meta["side_id"]
                ),

            "horizon":
                int(
                    meta["horizon"]
                ),

            "horizon_id":
                int(
                    meta["horizon_id"]
                ),

            "barrier_id":
                int(
                    meta["barrier_id"]
                ),

            "tp":
                int(
                    meta["tp"]
                ),

            "sl":
                int(
                    meta["sl"]
                ),
        }
    )


# ============================================================
# EXECUTION TARGETS
# ============================================================

def load_execution_targets():
    print(
        "Loading V6.7.0 execution targets ..."
    )

    columns = [
        "source_row",
        "timestamp",
        "year",
        "horizon_valid_h30",
        "horizon_valid_h60",
        "horizon_valid_h120",
    ]

    for meta in TASK_META:

        side = meta[
            "side"
        ]

        h = meta[
            "horizon"
        ]

        tp = meta[
            "tp"
        ]

        sl = meta[
            "sl"
        ]

        key = (
            f"h{h}_"
            f"tp{tp}_"
            f"sl{sl}"
        )

        columns += [
            f"{side}_gross_bps_{key}",
            f"{side}_win_c05_{key}",
            f"{side}_win_c10_{key}",
        ]

    df = pd.read_parquet(
        TARGET_FILE,
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

    source_row = df[
        "source_row"
    ].to_numpy(
        np.int64
    )

    if not np.array_equal(
        source_row,
        np.arange(
            len(df),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "V6.7.0 source_row is not contiguous."
        )

    n = len(
        df
    )

    n_tasks = len(
        TASK_META
    )

    win05 = np.zeros(
        (
            n,
            n_tasks,
        ),
        dtype=np.uint8,
    )

    win10 = np.zeros(
        (
            n,
            n_tasks,
        ),
        dtype=np.uint8,
    )

    gross = np.full(
        (
            n,
            n_tasks,
        ),
        np.nan,
        dtype=np.float32,
    )

    valid = np.zeros(
        (
            n,
            n_tasks,
        ),
        dtype=bool,
    )

    for meta in TASK_META:

        j = meta[
            "task"
        ]

        side = meta[
            "side"
        ]

        h = meta[
            "horizon"
        ]

        tp = meta[
            "tp"
        ]

        sl = meta[
            "sl"
        ]

        key = (
            f"h{h}_"
            f"tp{tp}_"
            f"sl{sl}"
        )

        gross[
            :,
            j
        ] = df[
            f"{side}_gross_bps_{key}"
        ].to_numpy(
            np.float32
        )

        win05[
            :,
            j
        ] = df[
            f"{side}_win_c05_{key}"
        ].to_numpy(
            np.uint8
        )

        win10[
            :,
            j
        ] = df[
            f"{side}_win_c10_{key}"
        ].to_numpy(
            np.uint8
        )

        valid[
            :,
            j
        ] = (
            df[
                f"horizon_valid_h{h}"
            ].to_numpy(
                np.uint8
            )
            == 1
        )

    timestamps = df[
        "timestamp"
    ].to_numpy()

    years = df[
        "year"
    ].to_numpy(
        np.int16
    )

    print(
        "Execution target rows:",
        f"{n:,}",
    )

    return {
        "win05":
            win05,

        "win10":
            win10,

        "gross":
            gross,

        "valid":
            valid,

        "timestamp":
            timestamps,

        "year":
            years,
    }


# ============================================================
# BASE OUTPUT EXTRACTION
# ============================================================

def find_race_logits(
    output,
):
    if isinstance(
        output,
        torch.Tensor,
    ):
        if (
            output.ndim == 3
            and output.shape[1] == 18
            and output.shape[2] == 3
        ):
            return output

    if isinstance(
        output,
        dict,
    ):
        preferred = (
            "race_logits",
            "race_logit",
            "race",
        )

        for key in preferred:
            if key in output:
                value = output[
                    key
                ]

                if (
                    isinstance(
                        value,
                        torch.Tensor,
                    )
                    and value.ndim == 3
                    and value.shape[1] == 18
                    and value.shape[2] == 3
                ):
                    return value

        for value in output.values():
            if (
                isinstance(
                    value,
                    torch.Tensor,
                )
                and value.ndim == 3
                and value.shape[1] == 18
                and value.shape[2] == 3
            ):
                return value

    if isinstance(
        output,
        (
            list,
            tuple,
        ),
    ):
        for value in output:
            if (
                isinstance(
                    value,
                    torch.Tensor,
                )
                and value.ndim == 3
                and value.shape[1] == 18
                and value.shape[2] == 3
            ):
                return value

    raise RuntimeError(
        "Could not locate 18x3 race logits "
        "in V6.6.1 forward output."
    )


# ============================================================
# V6.7.1 MODEL
# ============================================================

class ExecutionPrecisionBrainV671(
    nn.Module
):
    def __init__(
        self,
        base_model,
    ):
        super().__init__()

        self.base = (
            base_model
        )

        self._captured_state = (
            None
        )

        self._hook_handle = (
            self.base
            .market_state
            .register_forward_hook(
                self._capture_market_state
            )
        )

        self.side_embedding = (
            nn.Embedding(
                2,
                EMBED_DIM,
            )
        )

        self.horizon_embedding = (
            nn.Embedding(
                3,
                EMBED_DIM,
            )
        )

        self.barrier_embedding = (
            nn.Embedding(
                3,
                EMBED_DIM,
            )
        )

        input_dim = (
            STATE_DIM
            + 3 * EMBED_DIM
            + 3
        )

        self.precision_decoder = (
            nn.Sequential(
                nn.Linear(
                    input_dim,
                    TASK_HIDDEN,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    TASK_HIDDEN
                ),
                nn.Dropout(
                    0.10
                ),

                nn.Linear(
                    TASK_HIDDEN,
                    TASK_HIDDEN,
                ),
                nn.GELU(),
                nn.Dropout(
                    0.10
                ),

                nn.Linear(
                    TASK_HIDDEN,
                    3,
                ),
            )
        )

        side_ids = torch.tensor(
            [
                x["side_id"]
                for x in TASK_META
            ],
            dtype=torch.long,
        )

        horizon_ids = torch.tensor(
            [
                x["horizon_id"]
                for x in TASK_META
            ],
            dtype=torch.long,
        )

        barrier_ids = torch.tensor(
            [
                x["barrier_id"]
                for x in TASK_META
            ],
            dtype=torch.long,
        )

        self.register_buffer(
            "task_side_ids",
            side_ids,
        )

        self.register_buffer(
            "task_horizon_ids",
            horizon_ids,
        )

        self.register_buffer(
            "task_barrier_ids",
            barrier_ids,
        )

    def _capture_market_state(
        self,
        module,
        inputs,
        output,
    ):
        if isinstance(
            output,
            torch.Tensor,
        ):
            self._captured_state = (
                output
            )

        elif isinstance(
            output,
            (
                list,
                tuple,
            ),
        ):
            self._captured_state = (
                output[0]
            )

        else:
            raise RuntimeError(
                "Unexpected market_state output."
            )

    def forward(
        self,
        x,
    ):
        self._captured_state = None

        base_output = self.base(
            x
        )

        race_logits = (
            find_race_logits(
                base_output
            )
        )

        market_state = (
            self._captured_state
        )

        if market_state is None:
            raise RuntimeError(
                "market_state hook did not fire."
            )

        if market_state.ndim != 2:
            raise RuntimeError(
                "Expected market_state [B,D], "
                f"got {tuple(market_state.shape)}"
            )

        if (
            market_state.shape[1]
            != STATE_DIM
        ):
            raise RuntimeError(
                "Unexpected market state dimension: "
                f"{market_state.shape[1]} "
                f"(expected {STATE_DIM})"
            )

        batch_size = (
            market_state.shape[0]
        )

        race_prob = F.softmax(
            race_logits,
            dim=-1,
        )

        side_e = (
            self.side_embedding(
                self.task_side_ids
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        horizon_e = (
            self.horizon_embedding(
                self.task_horizon_ids
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        barrier_e = (
            self.barrier_embedding(
                self.task_barrier_ids
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        state = (
            market_state
            .unsqueeze(1)
            .expand(
                -1,
                18,
                -1,
            )
        )

        z = torch.cat(
            [
                state,
                side_e,
                horizon_e,
                barrier_e,
                race_prob,
            ],
            dim=-1,
        )

        out = (
            self.precision_decoder(
                z
            )
        )

        return {
            "win05_logit":
                out[
                    :,
                    :,
                    0
                ],

            "win10_logit":
                out[
                    :,
                    :,
                    1
                ],

            "net05_norm":
                out[
                    :,
                    :,
                    2
                ],

            "race_logits":
                race_logits,

            "market_state":
                market_state,
        }


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

def freeze_backbone(
    model,
):
    for p in model.base.parameters():
        p.requires_grad = False


def unfreeze_execution_layers(
    model,
):
    # Keep low-level experts frozen.
    for p in model.base.parameters():
        p.requires_grad = False

    modules = [
        model.base.cross_expert,
        model.base.market_state,
        model.base.side_towers,
        model.base.race_decoder,
        model.base.side_embedding,
        model.base.horizon_embedding,
        model.base.barrier_embedding,
    ]

    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def make_optimizer(
    model,
    backbone_active,
):
    head_parameters = []

    base_parameters = []

    for name, p in (
        model.named_parameters()
    ):
        if not p.requires_grad:
            continue

        if name.startswith(
            "base."
        ):
            base_parameters.append(
                p
            )
        else:
            head_parameters.append(
                p
            )

    groups = [
        {
            "params":
                head_parameters,

            "lr":
                HEAD_LR,
        }
    ]

    if (
        backbone_active
        and base_parameters
    ):
        groups.append(
            {
                "params":
                    base_parameters,

                "lr":
                    BACKBONE_LR,
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY,
    )


# ============================================================
# TARGET GATHER
# ============================================================

def gather_targets(
    row_ids,
    arrays,
    targets,
    device,
):
    source = arrays[
        "source"
    ][
        row_ids
    ]

    valid_np = targets[
        "valid"
    ][
        source
    ]

    win05_np = targets[
        "win05"
    ][
        source
    ]

    win10_np = targets[
        "win10"
    ][
        source
    ]

    gross_np = targets[
        "gross"
    ][
        source
    ]

    net05_np = (
        gross_np
        - 0.5
    )

    valid = torch.from_numpy(
        valid_np
    ).to(
        device=device,
        dtype=torch.bool,
        non_blocking=True,
    )

    win05 = torch.from_numpy(
        win05_np.astype(
            np.float32
        )
    ).to(
        device=device,
        non_blocking=True,
    )

    win10 = torch.from_numpy(
        win10_np.astype(
            np.float32
        )
    ).to(
        device=device,
        non_blocking=True,
    )

    net05_norm = torch.from_numpy(
        np.clip(
            net05_np
            / NET_SCALE,
            -2.0,
            2.0,
        ).astype(
            np.float32
        )
    ).to(
        device=device,
        non_blocking=True,
    )

    return (
        valid,
        win05,
        win10,
        net05_norm,
    )


# ============================================================
# LOSSES
# ============================================================

def hard_action_rank_loss(
    logits,
    y,
    valid,
):
    # Ranking loss is intentionally
    # evaluated in FP32 even when the
    # forward pass runs under AMP.
    #
    # FP16 cannot represent sentinel
    # values such as -1e9.
    rank_logits = logits.float()

    positive = (
        valid
        & (
            y > 0.5
        )
    )

    negative = (
        valid
        & (
            y <= 0.5
        )
    )

    has_pos = positive.any(
        dim=1
    )

    has_neg = negative.any(
        dim=1
    )

    usable = (
        has_pos
        & has_neg
    )

    if not usable.any():
        return (
            rank_logits.sum()
            * 0.0
        )

    sentinel = -1e9

    pos_score = (
        rank_logits
        .masked_fill(
            ~positive,
            sentinel,
        )
        .max(
            dim=1
        )
        .values
    )

    neg_score = (
        rank_logits
        .masked_fill(
            ~negative,
            sentinel,
        )
        .max(
            dim=1
        )
        .values
    )

    margin = 0.50

    return F.softplus(
        margin
        - (
            pos_score[
                usable
            ]
            - neg_score[
                usable
            ]
        )
    ).mean()


def listwise_profit_loss(
    logits,
    net_target,
    valid,
):
    # Keep ranking math in FP32.
    pred = (
        logits
        .float()
        .masked_fill(
            ~valid,
            -1e4,
        )
    )

    target_quality = (
        net_target.float()
        * NET_SCALE
        / 10.0
    )

    target_quality = (
        target_quality
        .masked_fill(
            ~valid,
            -1e4,
        )
    )

    target_prob = F.softmax(
        target_quality,
        dim=1,
    )

    pred_logprob = F.log_softmax(
        pred,
        dim=1,
    )

    usable = valid.any(
        dim=1
    )

    if not usable.any():
        return (
            pred.sum()
            * 0.0
        )

    loss = -(
        target_prob
        * pred_logprob
    ).sum(
        dim=1
    )

    return loss[
        usable
    ].mean()


def compute_loss(
    output,
    valid,
    win05,
    win10,
    net05_norm,
    race_true,
):
    logit05 = output[
        "win05_logit"
    ]

    logit10 = output[
        "win10_logit"
    ]

    net_pred = output[
        "net05_norm"
    ]

    race_logits = output[
        "race_logits"
    ]

    if not valid.any():
        raise RuntimeError(
            "Batch has no valid execution targets."
        )

    bce05 = F.binary_cross_entropy_with_logits(
        logit05[
            valid
        ],
        win05[
            valid
        ],
    )

    bce10 = F.binary_cross_entropy_with_logits(
        logit10[
            valid
        ],
        win10[
            valid
        ],
    )

    regression = F.smooth_l1_loss(
        net_pred[
            valid
        ],
        net05_norm[
            valid
        ],
        beta=0.25,
    )

    rank = hard_action_rank_loss(
        logit05,
        win05,
        valid,
    )

    listwise = listwise_profit_loss(
        logit05,
        net05_norm,
        valid,
    )

    p05 = torch.sigmoid(
        logit05
    )

    p10 = torch.sigmoid(
        logit10
    )

    # Higher transaction cost cannot
    # increase probability of a net win.
    monotonic = F.relu(
        p10 - p05
    )[
        valid
    ].mean()

    race_mask = (
        (race_true >= 0)
        & (race_true < 3)
    )

    if race_mask.any():
        race_aux = F.cross_entropy(
            race_logits[
                race_mask
            ],
            race_true[
                race_mask
            ].long(),
        )
    else:
        race_aux = (
            race_logits.sum()
            * 0.0
        )

    total = (
        1.00 * bce05
        + 0.60 * bce10
        + 0.35 * regression
        + 0.35 * rank
        + 0.20 * listwise
        + 0.10 * monotonic
        + 0.12 * race_aux
    )

    parts = {
        "total":
            total,

        "bce05":
            bce05,

        "bce10":
            bce10,

        "reg":
            regression,

        "rank":
            rank,

        "list":
            listwise,

        "mono":
            monotonic,

        "race":
            race_aux,
    }

    return (
        total,
        parts,
    )


# ============================================================
# TRAIN EPOCH
# ============================================================

def train_epoch(
    model,
    optimizer,
    rows,
    epoch,
    arrays,
    mean,
    std,
    targets,
    device,
    scaler,
    backbone_active,
):
    model.train()

    if not backbone_active:
        model.base.eval()

    rng = np.random.default_rng(
        SEED + epoch * 1009
    )

    order = rng.permutation(
        rows
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
        "bce05": 0.0,
        "bce10": 0.0,
        "reg": 0.0,
        "rank": 0.0,
        "list": 0.0,
        "mono": 0.0,
        "race": 0.0,
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

        batch_size = (
            x.shape[0]
        )

        batch_rows = order[
            cursor:
            cursor
            + batch_size
        ]

        cursor += batch_size

        (
            valid,
            win05,
            win10,
            net05_norm,
        ) = gather_targets(
            batch_rows,
            arrays,
            targets,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):
            output = model(
                x
            )

            loss, parts = compute_loss(
                output,
                valid,
                win05,
                win10,
                net05_norm,
                race_true,
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

        for key in sums:
            sums[
                key
            ] += float(
                parts[
                    key
                ].detach()
            )

        batches += 1

    if cursor != len(
        order
    ):
        raise RuntimeError(
            "Loader order alignment failure: "
            f"{cursor} != {len(order)}"
        )

    return {
        key:
            value
            / max(
                batches,
                1,
            )
        for key, value
        in sums.items()
    }


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_precision(
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

    p05 = []
    p10 = []
    net = []

    cursor = 0

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

        p05.append(
            torch.sigmoid(
                output[
                    "win05_logit"
                ]
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        p10.append(
            torch.sigmoid(
                output[
                    "win10_logit"
                ]
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        net.append(
            (
                output[
                    "net05_norm"
                ]
                * NET_SCALE
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        cursor += (
            x.shape[0]
        )

    if cursor != len(
        rows
    ):
        raise RuntimeError(
            "Prediction alignment failure."
        )

    return {
        "p05":
            np.concatenate(
                p05,
                axis=0,
            ),

        "p10":
            np.concatenate(
                p10,
                axis=0,
            ),

        "net":
            np.concatenate(
                net,
                axis=0,
            ),
    }


# ============================================================
# METRICS
# ============================================================

def binary_metrics(
    probability,
    truth,
    valid,
):
    p = probability[
        valid
    ]

    y = truth[
        valid
    ].astype(
        np.uint8
    )

    ap = float(
        average_precision_score(
            y,
            p,
        )
    )

    if (
        y.min()
        == y.max()
    ):
        auc = np.nan
    else:
        auc = float(
            roc_auc_score(
                y,
                p,
            )
        )

    return (
        ap,
        auc,
        float(
            y.mean()
        ),
    )


def profit_factor(
    pnl,
):
    pnl = np.asarray(
        pnl,
        dtype=np.float64,
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


def trading_day_ny(
    timestamps,
):
    ts = pd.Series(
        pd.to_datetime(
            timestamps,
            utc=True,
        )
    )

    ny = ts.dt.tz_convert(
        "America/New_York"
    )

    return (
        (
            ny
            + pd.Timedelta(
                hours=7
            )
        )
        .dt.floor(
            "D"
        )
        .dt.tz_localize(
            None
        )
        .to_numpy()
    )


def causal_daily_frontier(
    p05,
    valid,
    win05,
    gross,
    timestamps,
):
    masked = p05.copy()

    masked[
        ~valid
    ] = -np.inf

    best_task = np.argmax(
        masked,
        axis=1,
    )

    idx = np.arange(
        len(
            masked
        )
    )

    best_score = masked[
        idx,
        best_task,
    ]

    best_win = win05[
        idx,
        best_task,
    ]

    best_net = (
        gross[
            idx,
            best_task,
        ]
        - 0.5
    )

    day = trading_day_ny(
        timestamps
    )

    counts = (
        pd.Series(
            day
        )
        .value_counts()
    )

    eligible_days = set(
        counts[
            counts
            >= MIN_DAY_BARS
        ].index
    )

    rows = []

    for threshold in (
        DAILY_THRESHOLDS
    ):
        used = set()

        chosen_win = []
        chosen_net = []

        for i in range(
            len(
                best_score
            )
        ):
            d = pd.Timestamp(
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

            if not np.isfinite(
                best_score[
                    i
                ]
            ):
                continue

            if (
                best_score[
                    i
                ]
                < threshold
            ):
                continue

            used.add(
                d
            )

            chosen_win.append(
                int(
                    best_win[
                        i
                    ]
                )
            )

            chosen_net.append(
                float(
                    best_net[
                        i
                    ]
                )
            )

        n = len(
            chosen_win
        )

        coverage = (
            n
            / len(
                eligible_days
            )
            if eligible_days
            else np.nan
        )

        if n:
            win = float(
                np.mean(
                    chosen_win
                )
            )

            mean_net = float(
                np.mean(
                    chosen_net
                )
            )

            pf = profit_factor(
                chosen_net
            )

        else:
            win = np.nan
            mean_net = np.nan
            pf = np.nan

        rows.append(
            {
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
                    win,

                "mean_net":
                    mean_net,

                "pf":
                    pf,
            }
        )

    return pd.DataFrame(
        rows
    )


def frontier_selection(
    frontier,
):
    eligible = frontier[
        (
            frontier[
                "coverage"
            ]
            >= SELECTION_MIN_COVERAGE
        )
        & np.isfinite(
            frontier[
                "win"
            ]
        )
    ].copy()

    if len(
        eligible
    ) == 0:
        eligible = frontier[
            np.isfinite(
                frontier[
                    "win"
                ]
            )
        ].copy()

    if len(
        eligible
    ) == 0:
        return (
            -np.inf,
            None,
        )

    scores = (
        1.50
        * eligible[
            "win"
        ]
        +
        0.80
        * eligible[
            "coverage"
        ]
        +
        0.05
        * np.tanh(
            eligible[
                "mean_net"
            ]
            / 5.0
        )
    )

    best_idx = scores.idxmax()

    return (
        float(
            scores.loc[
                best_idx
            ]
        ),
        eligible.loc[
            best_idx
        ].to_dict(),
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
    targets,
    device,
    label,
):
    pred = predict_precision(
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

    valid = targets[
        "valid"
    ][
        source
    ]

    win05 = targets[
        "win05"
    ][
        source
    ]

    win10 = targets[
        "win10"
    ][
        source
    ]

    gross = targets[
        "gross"
    ][
        source
    ]

    timestamps = targets[
        "timestamp"
    ][
        source
    ]

    ap05, auc05, base05 = (
        binary_metrics(
            pred[
                "p05"
            ],
            win05,
            valid,
        )
    )

    ap10, auc10, base10 = (
        binary_metrics(
            pred[
                "p10"
            ],
            win10,
            valid,
        )
    )

    task_rows = []

    for j, meta in enumerate(
        TASK_META
    ):
        mask = valid[
            :,
            j
        ]

        ap, auc, base = (
            binary_metrics(
                pred[
                    "p05"
                ][
                    :,
                    j
                ],
                win05[
                    :,
                    j
                ],
                mask,
            )
        )

        task_rows.append(
            {
                "task":
                    j,

                "side":
                    meta[
                        "side"
                    ].upper(),

                "horizon":
                    meta[
                        "horizon"
                    ],

                "tp":
                    meta[
                        "tp"
                    ],

                "sl":
                    meta[
                        "sl"
                    ],

                "base":
                    base,

                "ap":
                    ap,

                "auc":
                    auc,

                "ap_gain":
                    ap - base,
            }
        )

    task_df = pd.DataFrame(
        task_rows
    )

    frontier = causal_daily_frontier(
        pred[
            "p05"
        ],
        valid,
        win05,
        gross,
        timestamps,
    )

    daily_score, daily_best = (
        frontier_selection(
            frontier
        )
    )

    selection = (
        0.40 * ap05
        + 0.20 * auc05
        + 0.10 * ap10
        + 0.30 * daily_score
    )

    print()
    print(
        f"{label} EXECUTION-PRECISION METRICS"
    )

    print(
        "-" * 120
    )

    print(
        f"WIN@0.5 "
        f"base={base05:.4f} "
        f"AP={ap05:.4f} "
        f"AUC={auc05:.4f} "
        f"AP-gain={ap05-base05:+.4f}"
    )

    print(
        f"WIN@1.0 "
        f"base={base10:.4f} "
        f"AP={ap10:.4f} "
        f"AUC={auc10:.4f} "
        f"AP-gain={ap10-base10:+.4f}"
    )

    print(
        f"Selection score: "
        f"{selection:.6f}"
    )

    print()
    print(
        f"{label} CAUSAL DAILY FRONTIER"
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

    if daily_best is not None:
        print()
        print(
            "Best selection frontier point:",
            daily_best,
        )

    print()
    print(
        f"{label} TOP TASKS BY AP GAIN"
    )

    print(
        task_df.sort_values(
            "ap_gain",
            ascending=False,
        )
        .head(
            18
        )
        .to_string(
            index=False,
        )
    )

    return {
        "selection":
            float(
                selection
            ),

        "ap05":
            ap05,

        "auc05":
            auc05,

        "base05":
            base05,

        "ap10":
            ap10,

        "auc10":
            auc10,

        "base10":
            base10,

        "daily_score":
            daily_score,

        "daily_best":
            daily_best,

        "frontier":
            frontier,

        "tasks":
            task_df,

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
        "TEN V6.7.1 "
        "EXECUTION-ALIGNED PRECISION BRAIN"
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

    targets = (
        load_execution_targets()
    )

    source_all = arrays[
        "source"
    ]

    if (
        source_all.min() < 0
        or source_all.max()
        >= len(
            targets[
                "year"
            ]
        )
    ):
        raise RuntimeError(
            "Feature source index outside "
            "V6.7.0 target range."
        )

    model_year = targets[
        "year"
    ][
        source_all
    ]

    train_rows = split[
        "train"
    ]

    val_all = split[
        "val"
    ]

    rows2023 = val_all[
        model_year[
            val_all
        ]
        == 2023
    ]

    rows2024 = val_all[
        model_year[
            val_all
        ]
        == 2024
    ]

    if not np.all(
        model_year[
            train_rows
        ]
        <= 2022
    ):
        raise RuntimeError(
            "Training split contains post-2022 rows."
        )

    print(
        "Features:",
        len(
            names
        ),
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
        "Benchmark 2024:",
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

    # ------------------------------------
    # Load V6.6.1 champion
    # ------------------------------------

    base = (
        brain.MultiSurfaceTechnicalBrain(
            groups
        )
        .to(
            device
        )
    )

    old_ckpt = torch.load(
        execmod.CKPT,
        map_location=device,
        weights_only=False,
    )

    base.load_state_dict(
        old_ckpt[
            "model"
        ]
    )

    print(
        "Initialized from V6.6.1 epoch:",
        old_ckpt[
            "epoch"
        ],
    )

    model = (
        ExecutionPrecisionBrainV671(
            base
        )
        .to(
            device
        )
    )

    freeze_backbone(
        model
    )

    optimizer = make_optimizer(
        model,
        backbone_active=False,
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                device.type
                == "cuda"
            )
        )
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_params:,}",
    )

    print(
        "Initial trainable parameters:",
        f"{trainable:,}",
    )

    best_selection = -np.inf
    best_epoch = None
    stale = 0

    history = []

    # ------------------------------------
    # TRAINING
    # ------------------------------------

    for epoch in range(
        1,
        TOTAL_EPOCHS + 1,
    ):

        backbone_active = (
            epoch
            > HEAD_ONLY_EPOCHS
        )

        if (
            epoch
            == HEAD_ONLY_EPOCHS + 1
        ):
            print()
            print(
                "=" * 130
            )

            print(
                "UNFREEZING EXECUTION-RELEVANT "
                "BACKBONE LAYERS"
            )

            print(
                "=" * 130
            )

            unfreeze_execution_layers(
                model
            )

            optimizer = make_optimizer(
                model,
                backbone_active=True,
            )

            trainable = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            print(
                "Trainable parameters:",
                f"{trainable:,}",
            )

        print()
        print(
            "=" * 130
        )

        print(
            f"EPOCH {epoch}/{TOTAL_EPOCHS} "
            f"| "
            + (
                "PARTIAL BACKBONE FINE-TUNE"
                if backbone_active
                else "PRECISION HEAD ONLY"
            )
        )

        print(
            "=" * 130
        )

        train_stats = train_epoch(
            model,
            optimizer,
            train_rows,
            epoch,
            arrays,
            mean,
            std,
            targets,
            device,
            scaler,
            backbone_active,
        )

        print(
            "TRAIN "
            + " ".join(
                [
                    f"{k}={v:.5f}"
                    for k, v
                    in train_stats.items()
                ]
            )
        )

        val = evaluate(
            model,
            rows2023,
            arrays,
            mean,
            std,
            targets,
            device,
            "2023 VALIDATION",
        )

        row = {
            "epoch":
                epoch,

            "backbone_active":
                backbone_active,

            "selection":
                val[
                    "selection"
                ],

            "ap05":
                val[
                    "ap05"
                ],

            "auc05":
                val[
                    "auc05"
                ],

            "ap10":
                val[
                    "ap10"
                ],

            "auc10":
                val[
                    "auc10"
                ],

            "daily_score":
                val[
                    "daily_score"
                ],
        }

        if (
            val[
                "daily_best"
            ]
            is not None
        ):
            for k, v in (
                val[
                    "daily_best"
                ].items()
            ):
                row[
                    f"daily_{k}"
                ] = v

        history.append(
            row
        )

        pd.DataFrame(
            history
        ).to_csv(
            OUT
            / "training_history_v671.csv",
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

                    "selection":
                        best_selection,

                    "model":
                        model.state_dict(),

                    "feature_count":
                        len(
                            names
                        ),

                    "tasks":
                        TASK_META,

                    "config":
                        {
                            "head_only_epochs":
                                HEAD_ONLY_EPOCHS,

                            "total_epochs":
                                TOTAL_EPOCHS,

                            "head_lr":
                                HEAD_LR,

                            "backbone_lr":
                                BACKBONE_LR,

                            "net_scale":
                                NET_SCALE,
                        },
                },
                CHAMPION,
            )

            val[
                "frontier"
            ].to_csv(
                OUT
                / "champion_2023_frontier.csv",
                index=False,
            )

            val[
                "tasks"
            ].to_csv(
                OUT
                / "champion_2023_tasks.csv",
                index=False,
            )

            print()
            print(
                "*** NEW V6.7.1 CHAMPION ***"
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
                "No improvement. "
                f"stale={stale}/{PATIENCE}"
            )

        # Do not early-stop during
        # head-only stage.
        if (
            epoch
            > HEAD_ONLY_EPOCHS
            and stale
            >= PATIENCE
        ):
            print(
                "Early stopping."
            )
            break

    if not CHAMPION.exists():
        raise RuntimeError(
            "No V6.7.1 champion was saved."
        )

    # ------------------------------------
    # FREEZE CHAMPION
    # ------------------------------------

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
        "FROZEN V6.7.1 CHAMPION"
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
        "2023 selection:",
        champion[
            "selection"
        ],
    )

    # ------------------------------------
    # Final 2023
    # ------------------------------------

    final23 = evaluate(
        model,
        rows2023,
        arrays,
        mean,
        std,
        targets,
        device,
        "FINAL 2023",
    )

    # ------------------------------------
    # 2024 benchmark ONLY AFTER freeze
    # ------------------------------------

    print()
    print(
        "=" * 130
    )

    print(
        "OPENING 2024 BENCHMARK "
        "AFTER CHAMPION FREEZE"
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
        targets,
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

    final24[
        "tasks"
    ].to_csv(
        OUT
        / "frozen_2024_tasks.csv",
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

        "final_2023_ap05":
            float(
                final23[
                    "ap05"
                ]
            ),

        "final_2023_auc05":
            float(
                final23[
                    "auc05"
                ]
            ),

        "final_2024_ap05":
            float(
                final24[
                    "ap05"
                ]
            ),

        "final_2024_auc05":
            float(
                final24[
                    "auc05"
                ]
            ),

        "2025_evaluated":
            False,

        "2026_evaluated":
            False,
    }

    with open(
        OUT
        / "summary_v671.json",
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
        "TEN V6.7.1 TRAINING COMPLETE"
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
