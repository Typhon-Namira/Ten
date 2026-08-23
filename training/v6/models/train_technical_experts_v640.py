from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

TECH_DIR = Path(
    "training/v6/data_lake/technical_setup_v620"
)

TARGET_FILE = Path(
    "training/v6/data_lake/large_move_v60/"
    "large_move_targets_v60.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "technical_experts_v640"
)

SEQ = 24
BATCH = 512
EPOCHS = 12
PATIENCE = 3
WORKERS = 4
SEED = 640

STEP_NS = 300_000_000_000

TP_VALUE = 29.5
SL_VALUE = -15.5
COST = 0.5


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def realized_from_race(
    race,
    terminal,
):
    out = np.full(
        len(race),
        np.nan,
        np.float32,
    )

    out[
        race == 1
    ] = TP_VALUE

    out[
        race == 0
    ] = SL_VALUE

    m = race == -1

    out[m] = (
        terminal[m]
        - COST
    )

    return out


def contiguous_ok(
    timestamps,
    source,
    seq,
):
    bad = np.zeros(
        len(timestamps),
        np.int32,
    )

    bad[1:] = (
        np.diff(timestamps)
        != STEP_NS
    ).astype(
        np.int32
    )

    pref = np.concatenate(
        [
            [0],
            np.cumsum(
                bad,
                dtype=np.int64,
            ),
        ]
    )

    start = source - seq + 1

    safe = np.clip(
        start,
        0,
        None,
    )

    edges = (
        pref[source + 1]
        - pref[safe + 1]
    )

    return (
        (start >= 0)
        & (edges == 0)
    )


def window_valid(
    valid,
    source,
    seq,
):
    bad = (
        ~valid
    ).astype(
        np.int32
    )

    pref = np.concatenate(
        [
            [0],
            np.cumsum(
                bad,
                dtype=np.int64,
            ),
        ]
    )

    start = (
        source
        - seq
        + 1
    )

    safe = np.clip(
        start,
        0,
        None,
    )

    count = (
        pref[source + 1]
        - pref[safe]
    )

    return (
        (start >= 0)
        & (count == 0)
    )


def chunked_norm(
    features,
    rows,
    chunk=50000,
):
    s = np.zeros(
        features.shape[1],
        np.float64,
    )

    s2 = np.zeros(
        features.shape[1],
        np.float64,
    )

    n = 0

    for i in range(
        0,
        len(rows),
        chunk,
    ):
        r = rows[
            i:
            i + chunk
        ]

        x = np.asarray(
            features[r],
            np.float64,
        )

        s += x.sum(
            axis=0
        )

        s2 += np.square(
            x
        ).sum(
            axis=0
        )

        n += len(x)

    mean = (
        s
        / max(
            n,
            1,
        )
    )

    var = (
        s2
        / max(
            n,
            1,
        )
        - mean ** 2
    )

    std = np.sqrt(
        np.maximum(
            var,
            1e-8,
        )
    )

    std[
        std < 1e-4
    ] = 1.0

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def build_expert_groups(
    names,
):
    rules = {
        "trend": (
            "trend",
            "ema",
            "structure_score",
            "return_3",
            "return_6",
            "return_12",
            "return_24",
            "return_48",
            "return_96",
        ),

        "structure": (
            "structure",
            "prior_high",
            "prior_low",
            "range_position",
            "distance_high",
            "distance_low",
        ),

        "breakout": (
            "breakout",
            "retest",
            "false_breakout",
        ),

        "liquidity": (
            "sweep",
            "rejection",
            "wick",
            "support_rejection",
            "resistance_rejection",
        ),

        "momentum": (
            "momentum",
            "macd",
            "rsi",
            "accel",
        ),

        "volatility": (
            "atr",
            "volatility",
            "compression",
            "expansion",
            "range_bps",
            "rv_",
        ),

        "meanrev": (
            "bb_",
            "rsi_reversal",
            "support_rejection",
            "resistance_rejection",
        ),

        "session": (
            "hour_",
            "dow_",
            "session",
            "asia",
            "london",
            "ny_",
        ),

        "setup": (
            "setup_",
            "context_",
        ),

        "multitf": (
            "m15_",
            "h1_",
            "h4_",
        ),
    }

    groups = {}

    lower = [
        n.lower()
        for n in names
    ]

    for group, keys in rules.items():
        ids = [
            i
            for i, name
            in enumerate(lower)
            if any(
                key in name
                for key in keys
            )
        ]

        if ids:
            groups[group] = sorted(
                set(ids)
            )

    groups[
        "global"
    ] = list(
        range(
            len(names)
        )
    )

    if len(groups) < 6:
        raise RuntimeError(
            f"Not enough expert groups: "
            f"{groups.keys()}"
        )

    return groups


def load_data():
    x = np.load(
        TECH_DIR
        / "technical_setup_features.npy",
        mmap_mode="r",
    )

    valid = np.load(
        TECH_DIR
        / "technical_setup_valid.npy",
        mmap_mode="r",
    ).astype(
        bool
    )

    timestamps = np.load(
        TECH_DIR
        / "timestamps_ns.npy",
        mmap_mode="r",
    ).astype(
        np.int64
    )

    with open(
        TECH_DIR
        / "feature_names.json"
    ) as f:
        names = json.load(f)

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

    horizon = (
        target[
            "horizon_valid"
        ].to_numpy(
            np.int8
        )
        == 1
    )

    long_race = target[
        "long_race_tp30_sl15"
    ].to_numpy(
        np.int8
    )

    short_race = target[
        "short_race_tp30_sl15"
    ].to_numpy(
        np.int8
    )

    long_terminal = target[
        "long_terminal_bps"
    ].to_numpy(
        np.float32
    )

    short_terminal = target[
        "short_terminal_bps"
    ].to_numpy(
        np.float32
    )

    pnl = np.column_stack(
        [
            realized_from_race(
                long_race,
                long_terminal,
            ),

            realized_from_race(
                short_race,
                short_terminal,
            ),
        ]
    ).astype(
        np.float32
    )

    race = np.column_stack(
        [
            long_race,
            short_race,
        ]
    ).astype(
        np.int8
    )

    eligible = (
        horizon
        & np.isin(
            race,
            [-1, 0, 1],
        ).all(
            axis=1
        )
        & np.isfinite(
            pnl
        ).all(
            axis=1
        )
    )

    eligible &= (
        valid[source]
        & window_valid(
            valid,
            source,
            SEQ,
        )
        & contiguous_ok(
            timestamps,
            source,
            SEQ,
        )
    )

    # 1 = decisive event:
    # TP or SL happened before horizon.
    decisive = (
        race != -1
    ).astype(
        np.float32
    )

    # Conditional race target:
    # only meaningful where decisive == 1.
    tp_vs_sl = (
        race == 1
    ).astype(
        np.float32
    )

    # Auxiliary only for timeout rows.
    timeout_target = np.clip(
        pnl / 10.0,
        -3.0,
        3.0,
    ).astype(
        np.float32
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

    mean, std = chunked_norm(
        x,
        source[
            split["train"]
        ],
    )

    groups = build_expert_groups(
        names
    )

    arrays = {
        "features":
            x,

        "source":
            source,

        "decisive":
            decisive,

        "tp_vs_sl":
            tp_vs_sl,

        "timeout_target":
            timeout_target,

        "pnl":
            pnl,

        "race":
            race,
    }

    return (
        arrays,
        split,
        groups,
        names,
        mean,
        std,
    )


class TechnicalDataset(
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
            np.float32,
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
                seq[-1].copy()
            ),

            torch.from_numpy(
                self.a[
                    "decisive"
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    "tp_vs_sl"
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    "timeout_target"
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    "pnl"
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    "race"
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
        TechnicalDataset(
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


class Expert(
    nn.Module
):
    def __init__(
        self,
        dim,
        out_dim=64,
    ):
        super().__init__()

        hidden = max(
            64,
            min(
                256,
                dim * 2,
            ),
        )

        self.net = nn.Sequential(
            nn.LayerNorm(
                dim
            ),

            nn.Linear(
                dim,
                hidden,
            ),

            nn.GELU(),

            nn.Dropout(
                0.08
            ),

            nn.Linear(
                hidden,
                out_dim,
            ),

            nn.GELU(),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


class TemporalBlock(
    nn.Module
):
    def __init__(
        self,
        channels,
        dilation,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            1,
        )

        self.norm = nn.BatchNorm1d(
            channels
        )

    def forward(
        self,
        x,
    ):
        z = F.gelu(
            self.conv1(
                x
            )
        )

        z = self.conv2(
            z
        )

        return F.gelu(
            self.norm(
                x + z
            )
        )


class TechnicalExpertsV640(
    nn.Module
):
    def __init__(
        self,
        n_features,
        groups,
    ):
        super().__init__()

        self.names = list(
            groups
        )

        self.ids = [
            groups[k]
            for k in self.names
        ]

        dim = 64

        self.experts = nn.ModuleList(
            [
                Expert(
                    len(ids),
                    dim,
                )
                for ids
                in self.ids
            ]
        )

        self.regime = nn.Sequential(
            nn.Linear(
                dim * 2,
                128,
            ),

            nn.GELU(),

            nn.Linear(
                128,
                len(
                    self.names
                ),
            ),
        )

        self.q = nn.Linear(
            dim,
            dim,
        )

        self.k = nn.Linear(
            dim,
            dim,
        )

        self.v = nn.Linear(
            dim,
            dim,
        )

        self.o = nn.Linear(
            dim,
            dim,
        )

        self.temporal_in = nn.Linear(
            n_features,
            96,
        )

        self.temporal = nn.Sequential(
            TemporalBlock(
                96,
                1,
            ),

            TemporalBlock(
                96,
                2,
            ),

            TemporalBlock(
                96,
                4,
            ),
        )

        self.fuse = nn.Sequential(
            nn.Linear(
                dim
                + 96 * 2,
                256,
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                256,
                160,
            ),

            nn.GELU(),
        )

        # Is the next race decisive?
        self.event_head = nn.Linear(
            160,
            2,
        )

        # Conditional on TP/SL occurring:
        # which one wins?
        self.race_head = nn.Linear(
            160,
            2,
        )

        # Auxiliary terminal return
        # for timeout cases.
        self.timeout_head = nn.Linear(
            160,
            2,
        )

        # Independent ranking score.
        self.utility_head = nn.Linear(
            160,
            2,
        )

    def forward(
        self,
        seq,
        current,
    ):
        tokens = torch.stack(
            [
                expert(
                    current[
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

        # Global expert is inserted last.
        global_token = tokens[
            :,
            -1
        ]

        mean_token = tokens.mean(
            dim=1
        )

        gates = torch.softmax(
            self.regime(
                torch.cat(
                    [
                        global_token,
                        mean_token,
                    ],
                    dim=1,
                )
            ),
            dim=1,
        )

        q = self.q(
            tokens
        )

        k = self.k(
            tokens
        )

        v = self.v(
            tokens
        )

        attention = torch.softmax(
            torch.matmul(
                q,
                k.transpose(
                    1,
                    2,
                ),
            )
            / np.sqrt(
                q.shape[-1]
            ),
            dim=-1,
        )

        mixed = (
            tokens
            + self.o(
                torch.matmul(
                    attention,
                    v,
                )
            )
        )

        current_context = (
            mixed
            * gates.unsqueeze(
                -1
            )
        ).sum(
            dim=1
        )

        temporal = F.gelu(
            self.temporal_in(
                seq
            )
        )

        temporal = temporal.transpose(
            1,
            2,
        )

        temporal = self.temporal(
            temporal
        )

        z = self.fuse(
            torch.cat(
                [
                    current_context,
                    temporal[
                        :,
                        :,
                        -1
                    ],
                    temporal.mean(
                        dim=2
                    ),
                ],
                dim=1,
            )
        )

        return {
            "event":
                self.event_head(
                    z
                ),

            "race":
                self.race_head(
                    z
                ),

            "timeout":
                self.timeout_head(
                    z
                ),

            "utility":
                self.utility_head(
                    z
                ),

            "gates":
                gates,

            "attention":
                attention,
        }


def hard_rank_loss(
    score,
    race,
):
    losses = []

    for side in range(2):
        positive = score[
            :,
            side
        ][
            race[
                :,
                side
            ] == 1
        ]

        negative = score[
            :,
            side
        ][
            race[
                :,
                side
            ] == 0
        ]

        if (
            len(positive) == 0
            or len(negative) == 0
        ):
            continue

        k = min(
            64,
            len(positive),
            len(negative),
        )

        hard_negative = torch.topk(
            negative,
            k,
        ).values

        positive_pick = positive[
            torch.randperm(
                len(positive),
                device=positive.device,
            )[:k]
        ]

        losses.append(
            F.softplus(
                1.0
                - positive_pick[
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
            score.sum()
            * 0.0
        )

    return (
        sum(losses)
        / len(losses)
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total = 0.0
    count = 0

    for (
        seq,
        current,
        decisive,
        tp_vs_sl,
        timeout_target,
        pnl,
        race,
    ) in loader:

        seq = seq.to(
            device
        )

        current = current.to(
            device
        )

        decisive = decisive.to(
            device
        )

        tp_vs_sl = tp_vs_sl.to(
            device
        )

        timeout_target = (
            timeout_target.to(
                device
            )
        )

        race = race.to(
            device
        )

        out = model(
            seq,
            current,
        )

        loss_event = (
            F.binary_cross_entropy_with_logits(
                out["event"],
                decisive,
            )
        )

        decisive_mask = (
            decisive.bool()
        )

        if decisive_mask.any():
            loss_race = (
                F.binary_cross_entropy_with_logits(
                    out["race"][
                        decisive_mask
                    ],
                    tp_vs_sl[
                        decisive_mask
                    ],
                )
            )

        else:
            loss_race = (
                out["race"].sum()
                * 0.0
            )

        timeout_mask = (
            ~decisive_mask
        )

        if timeout_mask.any():
            loss_timeout = (
                F.smooth_l1_loss(
                    out["timeout"][
                        timeout_mask
                    ],
                    timeout_target[
                        timeout_mask
                    ],
                )
            )

        else:
            loss_timeout = (
                out["timeout"].sum()
                * 0.0
            )

        loss_rank = hard_rank_loss(
            out["utility"],
            race,
        )

        if decisive_mask.any():
            loss_utility = (
                F.binary_cross_entropy_with_logits(
                    out["utility"][
                        decisive_mask
                    ],
                    tp_vs_sl[
                        decisive_mask
                    ],
                )
            )

        else:
            loss_utility = (
                out["utility"].sum()
                * 0.0
            )

        loss = (
            0.35 * loss_event
            + 1.00 * loss_race
            + 0.10 * loss_timeout
            + 0.65 * loss_rank
            + 0.20 * loss_utility
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            2.0,
        )

        optimizer.step()

        total += (
            float(
                loss.item()
            )
            * len(seq)
        )

        count += len(seq)

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
    loader,
    device,
):
    model.eval()

    out = {
        key: []
        for key in (
            "event",
            "race",
            "timeout",
            "utility",
            "pnl",
            "race_true",
            "gates",
            "attention",
        )
    }

    for (
        seq,
        current,
        decisive,
        tp_vs_sl,
        timeout_target,
        pnl,
        race,
    ) in loader:

        pred = model(
            seq.to(
                device
            ),
            current.to(
                device
            ),
        )

        for key in (
            "event",
            "race",
            "timeout",
            "utility",
            "gates",
            "attention",
        ):
            out[key].append(
                pred[key]
                .cpu()
                .numpy()
            )

        out[
            "pnl"
        ].append(
            pnl.numpy()
        )

        out[
            "race_true"
        ].append(
            race.numpy()
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


def profit_factor(
    x,
):
    gains = x[
        x > 0
    ].sum()

    losses = -x[
        x < 0
    ].sum()

    if losses <= 0:
        return np.inf

    return float(
        gains / losses
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


def scores_from_prediction(
    pred,
):
    p_event = (
        1.0
        / (
            1.0
            + np.exp(
                -pred[
                    "event"
                ]
            )
        )
    )

    p_race = (
        1.0
        / (
            1.0
            + np.exp(
                -pred[
                    "race"
                ]
            )
        )
    )

    # Two-stage semantics:
    #
    # P(TP)
    # =
    # P(decisive)
    # *
    # P(TP | decisive)
    p_tp = (
        p_event
        * p_race
    )

    p_sl = (
        p_event
        * (
            1.0
            - p_race
        )
    )

    # Conservative score.
    # Timeout contribution is ignored here.
    race_edge = (
        TP_VALUE
        * p_tp
        + SL_VALUE
        * p_sl
    )

    return {
        "p_event":
            p_event,

        "p_race":
            p_race,

        "p_tp":
            p_tp,

        "p_sl":
            p_sl,

        "race_edge":
            race_edge,

        "utility":
            pred[
                "utility"
            ],
    }


def evaluate(
    name,
    pred,
    expert_names,
):
    score = scores_from_prediction(
        pred
    )

    race_true = pred[
        "race_true"
    ]

    pnl = pred[
        "pnl"
    ]

    print()
    print(name)
    print("-" * 120)

    for side, label in enumerate(
        (
            "LONG",
            "SHORT",
        )
    ):
        y_tp = (
            race_true[
                :,
                side
            ]
            == 1
        ).astype(
            np.int8
        )

        decisive = (
            race_true[
                :,
                side
            ]
            != -1
        )

        race_y = (
            race_true[
                decisive,
                side
            ]
            == 1
        ).astype(
            np.int8
        )

        print(
            f"{label:<5} "
            f"PTP_AP="
            f"{safe_ap(y_tp, score['p_tp'][:, side]):.4f} "
            f"PTP_AUC="
            f"{safe_auc(y_tp, score['p_tp'][:, side]):.4f} "
            f"RACE_AP="
            f"{safe_ap(race_y, score['p_race'][decisive, side]):.4f}"
        )

    policies = {
        "RACE_EDGE":
            score[
                "race_edge"
            ],

        "RACE_PROB":
            score[
                "p_race"
            ],

        "UTILITY":
            score[
                "utility"
            ],
    }

    utilities = {}

    for policy_name, policy in policies.items():
        long_side = (
            policy[:, 0]
            >= policy[:, 1]
        )

        rank_score = np.maximum(
            policy[:, 0],
            policy[:, 1],
        )

        selected_pnl = np.where(
            long_side,
            pnl[:, 0],
            pnl[:, 1],
        )

        selected_race = np.where(
            long_side,
            race_true[:, 0],
            race_true[:, 1],
        )

        means = []

        print(
            f"  [{policy_name}]"
        )

        for coverage in (
            5.0,
            2.0,
            1.0,
            0.5,
        ):
            n = max(
                1,
                round(
                    len(rank_score)
                    * coverage
                    / 100.0
                ),
            )

            idx = np.argpartition(
                rank_score,
                -n,
            )[-n:]

            x = selected_pnl[
                idx
            ]

            means.append(
                float(
                    x.mean()
                )
            )

            print(
                f"   {coverage:>4.1f}% "
                f"n={n:>5} "
                f"TP={(selected_race[idx] == 1).mean():>6.2%} "
                f"SL={(selected_race[idx] == 0).mean():>6.2%} "
                f"WIN={(x > 0).mean():>6.2%} "
                f"mean={x.mean():>+7.3f} "
                f"PF={profit_factor(x):>6.3f} "
                f"LONG={long_side[idx].mean():>6.2%}"
            )

        utilities[
            policy_name
        ] = (
            0.40
            * means[0]

            + 0.35
            * means[1]

            + 0.25
            * means[2]
        )

        print(
            f"   utility="
            f"{utilities[policy_name]:+.4f}"
        )

    gates = pred[
        "gates"
    ].mean(
        axis=0
    )

    order = np.argsort(
        gates
    )[
        ::-1
    ]

    print(
        "Expert gates:",
        ", ".join(
            f"{expert_names[i]}"
            f"={gates[i]:.3f}"
            for i in order
        ),
    )

    # Champion selection uses only
    # conservative RACE_EDGE.
    return utilities[
        "RACE_EDGE"
    ]


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
        "TEN V6.4 "
        "REGIME-CONDITIONED "
        "TECHNICAL EXPERT BRAIN"
    )

    print(
        "=" * 120
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
        feature_names,
        mean,
        std,
    ) = load_data()

    for name in (
        "train",
        "val",
        "test2025",
        "reserved2026",
    ):
        print(
            f"{name.upper():<12}: "
            f"{len(split[name]):,}"
        )

    print(
        "Features:",
        len(
            feature_names
        ),
    )

    print(
        "Experts:",
        {
            name:
                len(ids)
            for name, ids
            in groups.items()
        },
    )

    print(
        "Sequence:",
        SEQ,
        "M5 =",
        SEQ * 5,
        "minutes",
    )

    np.savez(
        OUT
        / "normalization_v640.npz",
        mean=mean,
        std=std,
    )

    with open(
        OUT
        / "expert_groups_v640.json",
        "w",
    ) as f:
        json.dump(
            groups,
            f,
            indent=2,
        )

    train_loader = make_loader(
        split["train"],
        True,
        arrays,
        mean,
        std,
    )

    val_loader = make_loader(
        split["val"],
        False,
        arrays,
        mean,
        std,
    )

    test_loader = make_loader(
        split["test2025"],
        False,
        arrays,
        mean,
        std,
    )

    model = (
        TechnicalExpertsV640(
            len(
                feature_names
            ),
            groups,
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
        lr=1.5e-4,
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

    best = -np.inf
    bad_epochs = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_prediction = predict(
            model,
            val_loader,
            device,
        )

        utility = evaluate(
            f"EPOCH {epoch} VALIDATION",
            val_prediction,
            model.names,
        )

        scheduler.step(
            utility
        )

        lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]

        print(
            f"Epoch {epoch:02d} "
            f"| loss={loss:.5f} "
            f"| utility={utility:+.4f} "
            f"| lr={lr:.2e}"
        )

        if utility > (
            best
            + 1e-4
        ):
            best = utility
            bad_epochs = 0

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model":
                        model.state_dict(),

                    "val_utility":
                        utility,

                    "groups":
                        groups,

                    "feature_names":
                        feature_names,

                    "seq":
                        SEQ,
                },
                OUT
                / "best_technical_experts_v640.pt",
            )

            print(
                "NEW V6.4 "
                "TECHNICAL CHAMPION"
            )

        else:
            bad_epochs += 1

            if (
                bad_epochs
                >= PATIENCE
            ):
                print(
                    "EARLY STOP"
                )

                break

    checkpoint = torch.load(
        OUT
        / "best_technical_experts_v640.pt",
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
        "=" * 120
    )

    print(
        "FROZEN V6.4 "
        "TECHNICAL CHAMPION"
    )

    print(
        "=" * 120
    )

    print(
        "Epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "VAL utility:",
        f"{checkpoint['val_utility']:+.4f}",
    )

    evaluate(
        "FINAL 2023-2024",
        predict(
            model,
            val_loader,
            device,
        ),
        model.names,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "2025 OUT-OF-TIME TEST"
    )

    print(
        "=" * 120
    )

    evaluate(
        "2025 V6.4 TECHNICAL EXPERTS",
        predict(
            model,
            test_loader,
            device,
        ),
        model.names,
    )

    print(
        "2026 RESERVED: "
        "NOT EVALUATED BY V6.4 TRAINER"
    )


if __name__ == "__main__":
    main()
