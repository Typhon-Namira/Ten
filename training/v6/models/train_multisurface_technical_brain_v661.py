from pathlib import Path
import json, random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

import training.v6.models.train_technical_moe_v652 as base


DATA_DIR = Path(
    "training/v6/data_lake/"
    "technical_state_v651"
)

TARGET_FILE = Path(
    "training/v6/data_lake/"
    "multihorizon_targets_v660/"
    "multihorizon_targets_v660.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "multisurface_technical_brain_v661"
)

SEQ = 24
BATCH = 384
EPOCHS = 12
PATIENCE = 3
WORKERS = 4
SEED = 661

HORIZONS = (
    30,
    60,
    120,
)

BARRIERS = (
    (30, 15),
    (40, 20),
    (60, 30),
)

SIDES = (
    "long",
    "short",
)


def seed_all(
    seed=SEED,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def safe_ap(
    y,
    score,
):
    try:
        return float(
            average_precision_score(
                y,
                score,
            )
        )
    except ValueError:
        return np.nan


def safe_auc(
    y,
    score,
):
    try:
        return float(
            roc_auc_score(
                y,
                score,
            )
        )
    except ValueError:
        return np.nan


def make_tasks():
    tasks = []

    for side_id, side in enumerate(
        SIDES
    ):
        for horizon_id, horizon in enumerate(
            HORIZONS
        ):
            for (
                barrier_id,
                (
                    tp,
                    sl,
                ),
            ) in enumerate(
                BARRIERS
            ):
                tasks.append(
                    {
                        "side_id":
                            side_id,

                        "side":
                            side,

                        "horizon_id":
                            horizon_id,

                        "horizon":
                            horizon,

                        "barrier_id":
                            barrier_id,

                        "tp":
                            tp,

                        "sl":
                            sl,
                    }
                )

    return tasks


TASKS = make_tasks()
N_TASKS = len(
    TASKS
)


def race_to_class(
    race,
):
    # 0 = TP
    # 1 = SL
    # 2 = TIMEOUT
    # -1 = ambiguous / ignore

    out = np.full(
        race.shape,
        -1,
        dtype=np.int8,
    )

    out[
        race == 1
    ] = 0

    out[
        race == 0
    ] = 1

    out[
        race == -1
    ] = 2

    return out


def load_data():
    features = np.load(
        DATA_DIR
        / "technical_features_v651.npy",
        mmap_mode="r",
    )

    valid = np.load(
        DATA_DIR
        / "technical_valid_v651.npy",
        mmap_mode="r",
    ).astype(
        bool
    )

    timestamps = np.load(
        DATA_DIR
        / "timestamps_ns.npy",
        mmap_mode="r",
    ).astype(
        np.int64
    )

    with open(
        DATA_DIR
        / "feature_names.json"
    ) as f:
        names = json.load(
            f
        )

    target = pd.read_parquet(
        TARGET_FILE
    )

    source = target[
        "source_row"
    ].to_numpy(
        np.int64
    )

    year = target[
        "year"
    ].to_numpy(
        np.int16
    )

    if len(features) != len(
        target
    ):
        raise RuntimeError(
            "Feature/target row mismatch"
        )

    race = np.empty(
        (
            len(target),
            N_TASKS,
        ),
        dtype=np.int8,
    )

    first_hit = np.zeros(
        (
            len(target),
            N_TASKS,
            2,
        ),
        dtype=np.int16,
    )

    for (
        task_id,
        meta,
    ) in enumerate(
        TASKS
    ):
        key = (
            f"h{meta['horizon']}_"
            f"tp{meta['tp']}_"
            f"sl{meta['sl']}"
        )

        side = meta[
            "side"
        ]

        race[
            :,
            task_id
        ] = target[
            f"{side}_race_{key}"
        ].to_numpy(
            np.int8
        )

        first_hit[
            :,
            task_id,
            0
        ] = target[
            f"{side}_first_tp_bar_{key}"
        ].to_numpy(
            np.int16
        )

        first_hit[
            :,
            task_id,
            1
        ] = target[
            f"{side}_first_sl_bar_{key}"
        ].to_numpy(
            np.int16
        )

    race_class = race_to_class(
        race
    )

    # Six side/horizon excursion tasks.
    #
    # LONG: 30 / 60 / 120
    # SHORT: 30 / 60 / 120
    #
    # channel 0 = MFE
    # channel 1 = MAE

    excursion = np.zeros(
        (
            len(target),
            6,
            2,
        ),
        dtype=np.float32,
    )

    k = 0

    for side in SIDES:
        for horizon in HORIZONS:
            mfe = target[
                f"{side}_mfe_bps_h{horizon}"
            ].to_numpy(
                np.float32
            )

            mae = target[
                f"{side}_mae_bps_h{horizon}"
            ].to_numpy(
                np.float32
            )

            excursion[
                :,
                k,
                0
            ] = np.log1p(
                np.maximum(
                    mfe,
                    0.0,
                )
            )

            excursion[
                :,
                k,
                1
            ] = np.log1p(
                np.maximum(
                    mae,
                    0.0,
                )
            )

            k += 1

    # Use only anchors where the
    # complete 120-minute surface exists.
    #
    # Because horizon nesting passed,
    # this guarantees 30m + 60m + 120m
    # all exist for the same sample.

    complete_surface = (
        target[
            "horizon_valid_h120"
        ].to_numpy(
            np.uint8
        )
        == 1
    )

    eligible = (
        complete_surface
        & valid[
            source
        ]
        & base.window_valid(
            valid,
            source,
            SEQ,
        )
        & base.contiguous_ok(
            timestamps,
            source,
            SEQ,
        )
    )

    split = {
        "train":
            np.flatnonzero(
                eligible
                & (
                    year <= 2022
                )
            ),

        "val":
            np.flatnonzero(
                eligible
                & (
                    year >= 2023
                )
                & (
                    year <= 2024
                )
            ),

        "test2025":
            np.flatnonzero(
                eligible
                & (
                    year == 2025
                )
            ),

        "reserved2026":
            np.flatnonzero(
                eligible
                & (
                    year == 2026
                )
            ),
    }

    mean, std = (
        base.chunked_norm(
            features,
            source[
                split[
                    "train"
                ]
            ],
        )
    )

    arrays = {
        "features":
            features,

        "source":
            source,

        "race":
            race,

        "race_class":
            race_class,

        "first_hit":
            first_hit,

        "excursion":
            excursion,
    }

    groups = base.build_groups(
        names
    )

    return (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    )


class SurfaceDataset(
    Dataset
):
    def __init__(
        self,
        rows,
        arrays,
        mean,
        std,
    ):
        self.rows = np.asarray(
            rows,
            np.int64,
        )

        self.a = arrays
        self.mean = mean
        self.std = std

    def __len__(
        self,
    ):
        return len(
            self.rows
        )

    def __getitem__(
        self,
        i,
    ):
        r = int(
            self.rows[i]
        )

        source = int(
            self.a[
                "source"
            ][r]
        )

        seq = np.asarray(
            self.a[
                "features"
            ][
                source
                - SEQ
                + 1:
                source
                + 1
            ],
            dtype=np.float32,
        )

        seq = (
            seq
            - self.mean
        ) / self.std

        return (
            torch.from_numpy(
                seq
            ),

            torch.from_numpy(
                self.a[
                    "race_class"
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    "first_hit"
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    "excursion"
                ][r].copy()
            ),
        )


def make_loader(
    rows,
    shuffle,
    arrays,
    mean,
    std,
):
    return DataLoader(
        SurfaceDataset(
            rows,
            arrays,
            mean,
            std,
        ),
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=(
            WORKERS > 0
        ),
        drop_last=shuffle,
    )


def class_weights(
    arrays,
    rows,
):
    y = arrays[
        "race_class"
    ][
        rows
    ]

    weights = np.ones(
        (
            N_TASKS,
            3,
        ),
        dtype=np.float32,
    )

    for task_id in range(
        N_TASKS
    ):
        z = y[
            :,
            task_id
        ]

        z = z[
            z >= 0
        ]

        count = np.bincount(
            z,
            minlength=3,
        ).astype(
            np.float64
        )

        total = count.sum()

        raw = np.sqrt(
            total
            / np.maximum(
                count,
                1.0,
            )
        )

        raw /= raw.mean()

        weights[
            task_id
        ] = np.clip(
            raw,
            0.35,
            6.0,
        ).astype(
            np.float32
        )

    return torch.from_numpy(
        weights
    )


class MultiSurfaceTechnicalBrain(
    nn.Module
):
    def __init__(
        self,
        groups,
    ):
        super().__init__()

        self.expert_names = list(
            groups
        )

        self.ids = [
            groups[k]
            for k in self.expert_names
        ]

        self.n_experts = len(
            self.ids
        )

        d = 64

        # Reuse the already-tested
        # V6.5.2 temporal expert encoder.

        self.experts = nn.ModuleList(
            [
                base.Expert(
                    len(ids),
                    d,
                )
                for ids
                in self.ids
            ]
        )

        self.market_token = nn.Parameter(
            torch.randn(
                1,
                1,
                d,
            )
            * 0.02
        )

        self.identity = nn.Parameter(
            torch.randn(
                1,
                self.n_experts + 1,
                d,
            )
            * 0.02
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=4,
                dim_feedforward=192,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
        )

        self.cross_expert = (
            nn.TransformerEncoder(
                layer,
                2,
            )
        )

        self.market_state = nn.Sequential(
            nn.LayerNorm(
                d
            ),

            nn.Linear(
                d,
                160,
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                160,
                128,
            ),

            nn.GELU(),
        )

        # Independent LONG / SHORT
        # directional specialists.

        self.side_towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        128,
                        128,
                    ),

                    nn.GELU(),

                    nn.Dropout(
                        0.10
                    ),

                    nn.Linear(
                        128,
                        128,
                    ),

                    nn.GELU(),
                )
                for _ in range(
                    2
                )
            ]
        )

        self.side_embedding = (
            nn.Embedding(
                2,
                24,
            )
        )

        self.horizon_embedding = (
            nn.Embedding(
                3,
                24,
            )
        )

        self.barrier_embedding = (
            nn.Embedding(
                3,
                24,
            )
        )

        self.register_buffer(
            "task_side",
            torch.tensor(
                [
                    m[
                        "side_id"
                    ]
                    for m in TASKS
                ],
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "task_horizon",
            torch.tensor(
                [
                    m[
                        "horizon_id"
                    ]
                    for m in TASKS
                ],
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "task_barrier",
            torch.tensor(
                [
                    m[
                        "barrier_id"
                    ]
                    for m in TASKS
                ],
                dtype=torch.long,
            ),
        )

        self.race_decoder = nn.Sequential(
            nn.Linear(
                128 + 72,
                192,
            ),

            nn.GELU(),

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
                3,
            ),
        )

        self.time_decoder = nn.Sequential(
            nn.Linear(
                128 + 72,
                160,
            ),

            nn.GELU(),

            nn.Linear(
                160,
                2,
            ),
        )

        excursion_side = []
        excursion_horizon = []

        for side_id in range(
            2
        ):
            for horizon_id in range(
                3
            ):
                excursion_side.append(
                    side_id
                )

                excursion_horizon.append(
                    horizon_id
                )

        self.register_buffer(
            "excursion_side",
            torch.tensor(
                excursion_side,
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "excursion_horizon",
            torch.tensor(
                excursion_horizon,
                dtype=torch.long,
            ),
        )

        self.excursion_decoder = (
            nn.Sequential(
                nn.Linear(
                    128 + 48,
                    160,
                ),

                nn.GELU(),

                nn.Linear(
                    160,
                    2,
                ),
            )
        )

    def forward(
        self,
        seq,
    ):
        expert_tokens = torch.stack(
            [
                expert(
                    seq[
                        :,
                        :,
                        ids
                    ]
                )
                for ids, expert
                in zip(
                    self.ids,
                    self.experts,
                )
            ],
            dim=1,
        )

        market = (
            self.market_token.expand(
                seq.shape[0],
                -1,
                -1,
            )
        )

        tokens = torch.cat(
            [
                market,
                expert_tokens,
            ],
            dim=1,
        )

        tokens = self.cross_expert(
            tokens
            + self.identity
        )

        state = self.market_state(
            tokens[
                :,
                0
            ]
        )

        side_state = torch.stack(
            [
                tower(
                    state
                )
                for tower
                in self.side_towers
            ],
            dim=1,
        )

        task_cond = torch.cat(
            [
                self.side_embedding(
                    self.task_side
                ),

                self.horizon_embedding(
                    self.task_horizon
                ),

                self.barrier_embedding(
                    self.task_barrier
                ),
            ],
            dim=1,
        )

        task_input = torch.cat(
            [
                side_state[
                    :,
                    self.task_side,
                    :
                ],

                task_cond[
                    None,
                    :,
                    :
                ].expand(
                    state.shape[0],
                    -1,
                    -1,
                ),
            ],
            dim=2,
        )

        excursion_cond = torch.cat(
            [
                self.side_embedding(
                    self.excursion_side
                ),

                self.horizon_embedding(
                    self.excursion_horizon
                ),
            ],
            dim=1,
        )

        excursion_input = torch.cat(
            [
                side_state[
                    :,
                    self.excursion_side,
                    :
                ],

                excursion_cond[
                    None,
                    :,
                    :
                ].expand(
                    state.shape[0],
                    -1,
                    -1,
                ),
            ],
            dim=2,
        )

        return {
            "race_logits":
                self.race_decoder(
                    task_input
                ),

            "time":
                torch.sigmoid(
                    self.time_decoder(
                        task_input
                    )
                ),

            "excursion_log":
                F.softplus(
                    self.excursion_decoder(
                        excursion_input
                    )
                ),
        }


def weighted_race_loss(
    logits,
    target,
    weights,
):
    valid = (
        target >= 0
    )

    safe = target.clamp(
        min=0
    ).long()

    logp = F.log_softmax(
        logits,
        dim=-1,
    )

    nll = -logp.gather(
        2,
        safe.unsqueeze(
            -1
        ),
    ).squeeze(
        -1
    )

    task_ids = torch.arange(
        target.shape[1],
        device=target.device,
    )[
        None,
        :
    ].expand_as(
        safe
    )

    sample_weight = (
        weights[
            task_ids,
            safe
        ]
        * valid.float()
    )

    return (
        (
            nll
            * sample_weight
        ).sum()
        / sample_weight.sum()
        .clamp_min(
            1.0
        )
    )


def hard_tp_sl_rank_loss(
    logits,
    target,
):
    # Score explicitly asks:
    #
    # "TP before SL"
    #
    # rather than:
    # "anything positive".

    score = (
        logits[
            :,
            :,
            0
        ]
        - logits[
            :,
            :,
            1
        ]
    )

    losses = []

    for task_id in range(
        N_TASKS
    ):
        tp = score[
            :,
            task_id
        ][
            target[
                :,
                task_id
            ]
            == 0
        ]

        sl = score[
            :,
            task_id
        ][
            target[
                :,
                task_id
            ]
            == 1
        ]

        if (
            len(tp) == 0
            or len(sl) == 0
        ):
            continue

        k = min(
            24,
            len(tp),
            len(sl),
        )

        hard_positive = torch.topk(
            tp,
            k,
            largest=False,
        ).values

        hard_negative = torch.topk(
            sl,
            k,
            largest=True,
        ).values

        losses.append(
            F.softplus(
                0.5
                - hard_positive[
                    :,
                    None
                ]
                + hard_negative[
                    None,
                    :
                ]
            ).mean()
        )

    if not losses:
        return (
            logits.sum()
            * 0.0
        )

    return (
        sum(
            losses
        )
        / len(
            losses
        )
    )


def probability_monotonicity(
    logits,
):
    # For the SAME side/barrier:
    #
    # P(TP first by 30m)
    # <= P(TP first by 60m)
    # <= P(TP first by 120m)
    #
    # Same for SL.
    #
    # Timeout moves in reverse.

    p = torch.softmax(
        logits,
        dim=-1,
    )

    losses = []

    for side_id in range(
        2
    ):
        base_id = (
            side_id
            * 9
        )

        for barrier_id in range(
            3
        ):
            i30 = (
                base_id
                + barrier_id
            )

            i60 = (
                base_id
                + 3
                + barrier_id
            )

            i120 = (
                base_id
                + 6
                + barrier_id
            )

            for cls in (
                0,
                1,
            ):
                losses.append(
                    F.relu(
                        p[
                            :,
                            i30,
                            cls
                        ]
                        - p[
                            :,
                            i60,
                            cls
                        ]
                    ).mean()
                )

                losses.append(
                    F.relu(
                        p[
                            :,
                            i60,
                            cls
                        ]
                        - p[
                            :,
                            i120,
                            cls
                        ]
                    ).mean()
                )

            losses.append(
                F.relu(
                    p[
                        :,
                        i60,
                        2
                    ]
                    - p[
                        :,
                        i30,
                        2
                    ]
                ).mean()
            )

            losses.append(
                F.relu(
                    p[
                        :,
                        i120,
                        2
                    ]
                    - p[
                        :,
                        i60,
                        2
                    ]
                ).mean()
            )

    return (
        sum(
            losses
        )
        / len(
            losses
        )
    )


def hit_time_loss(
    pred,
    first_hit,
):
    steps = torch.tensor(
        [
            m[
                "horizon"
            ]
            // 5
            for m in TASKS
        ],
        device=pred.device,
        dtype=torch.float32,
    )[
        None,
        :
    ]

    losses = []

    for hit_id in range(
        2
    ):
        target = first_hit[
            :,
            :,
            hit_id
        ].float()

        mask = (
            target > 0
        )

        if mask.any():
            normalized = (
                target
                / steps
            )

            losses.append(
                F.smooth_l1_loss(
                    pred[
                        :,
                        :,
                        hit_id
                    ][
                        mask
                    ],
                    normalized[
                        mask
                    ],
                )
            )

    if not losses:
        return (
            pred.sum()
            * 0.0
        )

    return (
        sum(
            losses
        )
        / len(
            losses
        )
    )


def excursion_monotonicity(
    pred,
):
    # Layout:
    #
    # LONG  H30,H60,H120
    # SHORT H30,H60,H120

    losses = []

    for side_id in range(
        2
    ):
        base_id = (
            side_id
            * 3
        )

        for metric_id in range(
            2
        ):
            x30 = pred[
                :,
                base_id,
                metric_id
            ]

            x60 = pred[
                :,
                base_id + 1,
                metric_id
            ]

            x120 = pred[
                :,
                base_id + 2,
                metric_id
            ]

            losses.append(
                F.relu(
                    x30 - x60
                ).mean()
            )

            losses.append(
                F.relu(
                    x60 - x120
                ).mean()
            )

    return (
        sum(
            losses
        )
        / len(
            losses
        )
    )


def train_epoch(
    model,
    dl,
    optimizer,
    scaler,
    device,
    race_weights,
):
    model.train()

    total = 0.0
    count = 0

    amp_enabled = (
        device.type
        == "cuda"
    )

    for (
        seq,
        race_class,
        first_hit,
        excursion,
    ) in dl:

        seq = seq.to(
            device,
            non_blocking=True,
        )

        race_class = (
            race_class.to(
                device,
                non_blocking=True,
            )
        )

        first_hit = (
            first_hit.to(
                device,
                non_blocking=True,
            )
        )

        excursion = (
            excursion.to(
                device,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            out = model(
                seq
            )

            loss_race = (
                weighted_race_loss(
                    out[
                        "race_logits"
                    ],
                    race_class,
                    race_weights,
                )
            )

            loss_rank = (
                hard_tp_sl_rank_loss(
                    out[
                        "race_logits"
                    ],
                    race_class,
                )
            )

            loss_prob_mono = (
                probability_monotonicity(
                    out[
                        "race_logits"
                    ]
                )
            )

            loss_time = (
                hit_time_loss(
                    out[
                        "time"
                    ],
                    first_hit,
                )
            )

            loss_exc = (
                F.smooth_l1_loss(
                    out[
                        "excursion_log"
                    ],
                    excursion,
                )
            )

            loss_exc_mono = (
                excursion_monotonicity(
                    out[
                        "excursion_log"
                    ]
                )
            )

            loss = (
                1.00
                * loss_race

                + 0.25
                * loss_rank

                + 0.15
                * loss_prob_mono

                + 0.12
                * loss_time

                + 0.18
                * loss_exc

                + 0.08
                * loss_exc_mono
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

        total += (
            float(
                loss.item()
            )
            * len(
                seq
            )
        )

        count += len(
            seq
        )

    return (
        total
        / max(
            count,
            1,
        )
    )


@torch.no_grad()
def predict(
    model,
    dl,
    device,
):
    model.eval()

    out = {
        "race_prob":
            [],

        "race_true":
            [],

        "excursion_log":
            [],

        "excursion_true":
            [],
    }

    for (
        seq,
        race_class,
        first_hit,
        excursion,
    ) in dl:

        pred = model(
            seq.to(
                device,
                non_blocking=True,
            )
        )

        out[
            "race_prob"
        ].append(
            torch.softmax(
                pred[
                    "race_logits"
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        out[
            "race_true"
        ].append(
            race_class.numpy()
        )

        out[
            "excursion_log"
        ].append(
            pred[
                "excursion_log"
            ]
            .cpu()
            .numpy()
        )

        out[
            "excursion_true"
        ].append(
            excursion.numpy()
        )

    return {
        key:
            np.concatenate(
                value,
                axis=0,
            )
        for key, value
        in out.items()
    }


def evaluate(
    name,
    pred,
    print_tasks=True,
):
    prob = pred[
        "race_prob"
    ]

    target = pred[
        "race_true"
    ]

    side_lift = [
        [],
        [],
    ]

    side_auc = [
        [],
        [],
    ]

    rows = []

    for (
        task_id,
        meta,
    ) in enumerate(
        TASKS
    ):
        valid = (
            target[
                :,
                task_id
            ]
            >= 0
        )

        y = target[
            valid,
            task_id
        ]

        p = prob[
            valid,
            task_id
        ]

        y_tp = (
            y == 0
        ).astype(
            np.int8
        )

        baseline = float(
            y_tp.mean()
        )

        ap = safe_ap(
            y_tp,
            p[
                :,
                0
            ],
        )

        auc = safe_auc(
            y_tp,
            p[
                :,
                0
            ],
        )

        resolved = (
            (y == 0)
            | (
                y == 1
            )
        )

        race_y = (
            y[
                resolved
            ]
            == 0
        ).astype(
            np.int8
        )

        race_score = (
            p[
                resolved,
                0
            ]
            / np.maximum(
                p[
                    resolved,
                    0
                ]
                + p[
                    resolved,
                    1
                ],
                1e-12,
            )
        )

        race_auc = safe_auc(
            race_y,
            race_score,
        )

        lift = (
            np.log(
                max(
                    ap,
                    1e-8,
                )
                / max(
                    baseline,
                    1e-8,
                )
            )
            if np.isfinite(
                ap
            )
            else np.nan
        )

        side_id = meta[
            "side_id"
        ]

        if np.isfinite(
            lift
        ):
            side_lift[
                side_id
            ].append(
                lift
            )

        if np.isfinite(
            race_auc
        ):
            side_auc[
                side_id
            ].append(
                race_auc
            )

        rows.append(
            (
                meta[
                    "side"
                ].upper(),

                meta[
                    "horizon"
                ],

                f"{meta['tp']}/{meta['sl']}",

                baseline,
                ap,
                auc,
                race_auc,
            )
        )

    lift_mean = [
        float(
            np.mean(
                x
            )
        )
        for x
        in side_lift
    ]

    auc_mean = [
        float(
            np.mean(
                x
            )
        )
        for x
        in side_auc
    ]

    # Champion cannot hide behind
    # one strong direction.
    balanced_lift = min(
        lift_mean
    )

    balanced_auc = min(
        auc_mean
    )

    selection = (
        0.60
        * balanced_lift

        + 0.40
        * (
            2.0
            * (
                balanced_auc
                - 0.5
            )
        )
    )

    exc_mae = float(
        np.mean(
            np.abs(
                np.expm1(
                    pred[
                        "excursion_log"
                    ]
                )
                - np.expm1(
                    pred[
                        "excursion_true"
                    ]
                )
            )
        )
    )

    print()
    print(
        name
    )

    print(
        "-" * 124
    )

    print(
        "SIDE BALANCE | "
        f"LONG log(AP/base)="
        f"{lift_mean[0]:+.4f} "
        f"raceAUC="
        f"{auc_mean[0]:.4f} | "
        f"SHORT log(AP/base)="
        f"{lift_mean[1]:+.4f} "
        f"raceAUC="
        f"{auc_mean[1]:.4f}"
    )

    print(
        f"SELECTION="
        f"{selection:+.5f} "
        f"| excursion_MAE="
        f"{exc_mae:.3f} bps"
    )

    if print_tasks:
        print(
            "SIDE  HORIZON BARRIER  "
            "BASE_TP    TP_AP   TP_AUC  RACE_AUC"
        )

        for (
            side,
            horizon,
            barrier,
            base_rate,
            ap,
            auc,
            race_auc,
        ) in rows:

            print(
                f"{side:<5} "
                f"{horizon:>7}m "
                f"{barrier:>7} "
                f"{base_rate:>8.3%} "
                f"{ap:>8.4f} "
                f"{auc:>8.4f} "
                f"{race_auc:>9.4f}"
            )

    return selection


def tail_report(
    name,
    pred,
):
    prob = pred[
        "race_prob"
    ]

    target = pred[
        "race_true"
    ]

    focus = (
        (
            30,
            30,
            15,
        ),
        (
            60,
            30,
            15,
        ),
        (
            60,
            40,
            20,
        ),
        (
            120,
            30,
            15,
        ),
        (
            120,
            40,
            20,
        ),
        (
            120,
            60,
            30,
        ),
    )

    print()
    print(
        name
    )

    print(
        "=" * 124
    )

    for (
        side_id,
        side,
    ) in enumerate(
        SIDES
    ):
        for (
            horizon,
            tp,
            sl,
        ) in focus:

            task_id = next(
                i
                for i, meta
                in enumerate(
                    TASKS
                )
                if (
                    meta[
                        "side_id"
                    ]
                    == side_id
                    and meta[
                        "horizon"
                    ]
                    == horizon
                    and meta[
                        "tp"
                    ]
                    == tp
                    and meta[
                        "sl"
                    ]
                    == sl
                )
            )

            valid = (
                target[
                    :,
                    task_id
                ]
                >= 0
            )

            y = target[
                valid,
                task_id
            ]

            score = prob[
                valid,
                task_id,
                0
            ]

            print(
                f"{side.upper():<5} "
                f"H{horizon:<3} "
                f"TP{tp}/SL{sl}"
            )

            for coverage in (
                2.0,
                1.0,
                0.5,
                0.2,
            ):
                n = max(
                    1,
                    round(
                        len(
                            score
                        )
                        * coverage
                        / 100.0
                    ),
                )

                idx = np.argpartition(
                    score,
                    -n,
                )[
                    -n:
                ]

                z = y[
                    idx
                ]

                tp_rate = float(
                    (
                        z == 0
                    ).mean()
                )

                sl_rate = float(
                    (
                        z == 1
                    ).mean()
                )

                timeout_rate = float(
                    (
                        z == 2
                    ).mean()
                )

                resolved = (
                    tp_rate
                    + sl_rate
                )

                tp_res = (
                    tp_rate
                    / resolved
                    if resolved > 0
                    else np.nan
                )

                print(
                    f"  {coverage:>4.1f}% "
                    f"n={n:>5} "
                    f"TP={tp_rate:>6.2%} "
                    f"SL={sl_rate:>6.2%} "
                    f"TO={timeout_rate:>6.2%} "
                    f"TP|RES={tp_res:>6.2%}"
                )


def main():
    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "TEN V6.6.1 "
        "MULTI-SURFACE TECHNICAL BRAIN"
    )

    print(
        "=" * 124
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    ) = load_data()

    for key in (
        "train",
        "val",
        "test2025",
        "reserved2026",
    ):
        print(
            f"{key.upper():<12}: "
            f"{len(split[key]):,}"
        )

    print(
        "Features:",
        len(
            names
        ),
    )

    print(
        "Tasks:",
        N_TASKS,
    )

    print(
        "Sequence:",
        SEQ,
        "M5 =",
        SEQ * 5,
        "minutes",
    )

    print(
        "Experts:",
        {
            k:
                len(vv)
            for k, vv
            in groups.items()
        },
    )

    np.savez(
        OUT
        / "normalization_v661.npz",
        mean=mean,
        std=std,
    )

    with open(
        OUT
        / "expert_groups_v661.json",
        "w",
    ) as f:
        json.dump(
            groups,
            f,
            indent=2,
        )

    with open(
        OUT
        / "task_meta_v661.json",
        "w",
    ) as f:
        json.dump(
            TASKS,
            f,
            indent=2,
        )

    train_dl = make_loader(
        split[
            "train"
        ],
        True,
        arrays,
        mean,
        std,
    )

    val_dl = make_loader(
        split[
            "val"
        ],
        False,
        arrays,
        mean,
        std,
    )

    test_dl = make_loader(
        split[
            "test2025"
        ],
        False,
        arrays,
        mean,
        std,
    )

    race_weights = class_weights(
        arrays,
        split[
            "train"
        ],
    ).to(
        device
    )

    print(
        "Race class weights range:",
        f"{race_weights.min().item():.3f}",
        "to",
        f"{race_weights.max().item():.3f}",
    )

    model = (
        MultiSurfaceTechnicalBrain(
            groups
        )
        .to(
            device
        )
    )

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-4,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    )

    best = -np.inf
    bad = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        loss = train_epoch(
            model,
            train_dl,
            optimizer,
            scaler,
            device,
            race_weights,
        )

        val_pred = predict(
            model,
            val_dl,
            device,
        )

        selection = evaluate(
            f"EPOCH {epoch} VALIDATION",
            val_pred,
            print_tasks=True,
        )

        scheduler.step(
            selection
        )

        lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        print(
            f"Epoch {epoch:02d} "
            f"| loss={loss:.5f} "
            f"| selection={selection:+.5f} "
            f"| lr={lr:.2e}"
        )

        if selection > (
            best
            + 1e-4
        ):
            best = selection
            bad = 0

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model":
                        model.state_dict(),

                    "val_selection":
                        selection,

                    "groups":
                        groups,

                    "feature_names":
                        names,

                    "tasks":
                        TASKS,

                    "seq":
                        SEQ,
                },
                OUT
                / "best_multisurface_technical_brain_v661.pt",
            )

            print(
                "NEW V6.6.1 "
                "MULTI-SURFACE CHAMPION"
            )

        else:
            bad += 1

            if bad >= PATIENCE:
                print(
                    "EARLY STOP"
                )

                break

    checkpoint = torch.load(
        OUT
        / "best_multisurface_technical_brain_v661.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    print()
    print(
        "=" * 124
    )

    print(
        "FROZEN V6.6.1 "
        "MULTI-SURFACE CHAMPION"
    )

    print(
        "=" * 124
    )

    print(
        "Epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "VAL selection:",
        f"{checkpoint['val_selection']:+.5f}",
    )

    final_val = predict(
        model,
        val_dl,
        device,
    )

    evaluate(
        "FINAL 2023-2024",
        final_val,
        print_tasks=True,
    )

    tail_report(
        "FINAL VALIDATION TAILS",
        final_val,
    )

    print()
    print(
        "=" * 124
    )

    print(
        "2025 OUT-OF-TIME TEST"
    )

    print(
        "=" * 124
    )

    # 2025 is only evaluated
    # after the champion is frozen.

    final_test = predict(
        model,
        test_dl,
        device,
    )

    evaluate(
        "2025 V6.6.1 MULTI-SURFACE",
        final_test,
        print_tasks=True,
    )

    tail_report(
        "2025 MULTI-SURFACE TAILS",
        final_test,
    )

    print(
        "2026 RESERVED: "
        "NOT EVALUATED BY V6.6.1 TRAINER"
    )


if __name__ == "__main__":
    main()
