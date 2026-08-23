from pathlib import Path
import json, random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


DATA_DIR = Path(
    'training/v6/data_lake/'
    'technical_state_v651'
)

TARGET_FILE = Path(
    'training/v6/data_lake/'
    'large_move_v60/'
    'large_move_targets_v60.parquet'
)

OUT = Path(
    'training/artifacts/v6/'
    'technical_moe_v652'
)


SEQ = 24
BATCH = 384
EPOCHS = 12
PATIENCE = 3
WORKERS = 4

SEED = 652

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


def realized(
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

    pref = np.r_[
        0,
        np.cumsum(
            bad,
            dtype=np.int64,
        ),
    ]

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

    return (
        (start >= 0)
        & (
            (
                pref[source + 1]
                - pref[safe]
            )
            == 0
        )
    )


def contiguous_ok(
    ts,
    source,
    seq,
):
    bad = np.zeros(
        len(ts),
        np.int32,
    )

    bad[1:] = (
        np.diff(ts)
        != STEP_NS
    ).astype(
        np.int32
    )

    pref = np.r_[
        0,
        np.cumsum(
            bad,
            dtype=np.int64,
        ),
    ]

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

    return (
        (start >= 0)
        & (
            (
                pref[source + 1]
                - pref[safe + 1]
            )
            == 0
        )
    )


def chunked_norm(
    features,
    rows,
    chunk=40000,
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
        x = np.asarray(
            features[
                rows[
                    i:
                    i + chunk
                ]
            ],
            np.float64,
        )

        s += x.sum(0)
        s2 += np.square(
            x
        ).sum(0)

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
        - np.square(
            mean
        )
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


def assign_expert(name):
    n = name.lower()

    rules = [
        (
            'session',
            (
                'session',
                'hour_',
                'dow_',
                'asia',
                'london',
                'new_york',
                'ny_',
                'overlap',
            ),
        ),

        (
            'breakout_retest',
            (
                'breakout',
                'retest',
                'acceptance',
                'failure',
                '_bos_',
                '_choch_',
                'bos_up',
                'bos_down',
                'choch_up',
                'choch_down',
            ),
        ),

        (
            'liquidity',
            (
                'sweep',
                'equal_high',
                'equal_low',
                'liquidity',
                'rejection',
                'wick_imbalance',
                'upper_wick',
                'lower_wick',
            ),
        ),

        (
            'support_resistance',
            (
                'support',
                'resistance',
                'swing_high',
                'swing_low',
                'prior_high',
                'prior_low',
                'distance_high',
                'distance_low',
                'range_position',
            ),
        ),

        (
            'mean_reversion',
            (
                'bb_',
                'reversion',
                'meanrev',
                'rsi_reversal',
            ),
        ),

        (
            'volatility',
            (
                'atr',
                'volatility',
                'vol_of_vol',
                'compression',
                'expansion',
                'rv_',
                'range_bps',
                'body_expansion',
            ),
        ),

        (
            'momentum',
            (
                'momentum',
                'macd',
                'rsi',
                'return_',
                'impulse',
                'efficiency',
                'accel',
                'roc_',
            ),
        ),

        (
            'trend_structure',
            (
                'trend',
                'ema',
                'structure',
                '_hh',
                '_hl',
                '_lh',
                '_ll',
                'higher_tf',
                'multitf',
                'mtf_',
            ),
        ),

        (
            'price_action',
            (
                'inside_bar',
                'outside_bar',
                'candle_run',
                'body_',
                'close_pos',
                'close_position',
                'wick',
            ),
        ),

        (
            'execution_micro',
            (
                'spread',
                'micro',
                'bps',
            ),
        ),
    ]

    for group, keys in rules:
        if any(
            key in n
            for key in keys
        ):
            return group

    return 'residual_context'


def build_groups(names):
    groups = {}

    for i, name in enumerate(
        names
    ):
        groups.setdefault(
            assign_expert(
                name
            ),
            [],
        ).append(i)

    groups = {
        key: value
        for key, value
        in groups.items()
        if value
    }

    flat = [
        i
        for ids in groups.values()
        for i in ids
    ]

    if sorted(flat) != list(
        range(
            len(names)
        )
    ):
        raise RuntimeError(
            'Expert partition incomplete'
        )

    if len(
        set(flat)
    ) != len(flat):
        raise RuntimeError(
            'Expert partition overlaps'
        )

    if len(groups) < 8:
        raise RuntimeError(
            f'Too few expert groups: '
            f'{list(groups)}'
        )

    return groups


def load_data():
    x = np.load(
        DATA_DIR
        / 'technical_features_v651.npy',
        mmap_mode='r',
    )

    valid = np.load(
        DATA_DIR
        / 'technical_valid_v651.npy',
        mmap_mode='r',
    ).astype(
        bool
    )

    ts = np.load(
        DATA_DIR
        / 'timestamps_ns.npy',
        mmap_mode='r',
    ).astype(
        np.int64
    )

    with open(
        DATA_DIR
        / 'feature_names.json'
    ) as f:
        names = json.load(f)

    if x.shape[1] != len(
        names
    ):
        raise RuntimeError(
            'Feature-name mismatch'
        )

    t = pd.read_parquet(
        TARGET_FILE
    )

    source = t[
        'source_row'
    ].to_numpy(
        np.int64
    )

    year = t[
        'year'
    ].to_numpy(
        np.int16
    )

    horizon = (
        t[
            'horizon_valid'
        ].to_numpy(
            np.int8
        )
        == 1
    )

    lr = t[
        'long_race_tp30_sl15'
    ].to_numpy(
        np.int8
    )

    sr = t[
        'short_race_tp30_sl15'
    ].to_numpy(
        np.int8
    )

    race = np.column_stack(
        [
            lr,
            sr,
        ]
    ).astype(
        np.int8
    )

    pnl = np.column_stack(
        [
            realized(
                lr,
                t[
                    'long_terminal_bps'
                ].to_numpy(
                    np.float32
                ),
            ),

            realized(
                sr,
                t[
                    'short_terminal_bps'
                ].to_numpy(
                    np.float32
                ),
            ),
        ]
    ).astype(
        np.float32
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
        & valid[
            source
        ]
        & window_valid(
            valid,
            source,
            SEQ,
        )
        & contiguous_ok(
            ts,
            source,
            SEQ,
        )
    )

    decisive = (
        race != -1
    ).astype(
        np.float32
    )

    tp_vs_sl = (
        race == 1
    ).astype(
        np.float32
    )

    # Ordinal quality:
    # TP = +1
    # SL = -1
    # timeout = continuous middle region
    ordinal = np.clip(
        pnl / 10.0,
        -0.8,
        0.8,
    ).astype(
        np.float32
    )

    ordinal[
        race == 1
    ] = 1.0

    ordinal[
        race == 0
    ] = -1.0

    split = {
        'train':
            np.flatnonzero(
                eligible
                & (
                    year <= 2022
                )
            ),

        'val':
            np.flatnonzero(
                eligible
                & (
                    year >= 2023
                )
                & (
                    year <= 2024
                )
            ),

        'test2025':
            np.flatnonzero(
                eligible
                & (
                    year == 2025
                )
            ),

        'reserved2026':
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
            split[
                'train'
            ]
        ],
    )

    arrays = {
        'features':
            x,

        'source':
            source,

        'race':
            race,

        'pnl':
            pnl,

        'decisive':
            decisive,

        'tp_vs_sl':
            tp_vs_sl,

        'ordinal':
            ordinal,
    }

    return (
        arrays,
        split,
        build_groups(
            names
        ),
        names,
        mean,
        std,
    )


class DS(Dataset):
    def __init__(
        self,
        rows,
        a,
        mean,
        std,
    ):
        self.rows = np.asarray(
            rows,
            np.int64,
        )

        self.a = a
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(
            self.rows
        )

    def __getitem__(self, i):
        r = int(
            self.rows[i]
        )

        source = int(
            self.a[
                'source'
            ][r]
        )

        seq = np.asarray(
            self.a[
                'features'
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
                self.a[
                    'decisive'
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    'tp_vs_sl'
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    'ordinal'
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    'pnl'
                ][r].copy()
            ),

            torch.from_numpy(
                self.a[
                    'race'
                ][r].copy()
            ),
        )


def loader(
    rows,
    shuffle,
    a,
    mean,
    std,
):
    return DataLoader(
        DS(
            rows,
            a,
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


class TBlock(nn.Module):
    def __init__(
        self,
        channels,
        dilation,
    ):
        super().__init__()

        self.c1 = nn.Conv1d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

        self.c2 = nn.Conv1d(
            channels,
            channels,
            1,
        )

        self.bn = nn.BatchNorm1d(
            channels
        )

        self.drop = nn.Dropout(
            0.08
        )

    def forward(self, x):
        z = F.gelu(
            self.c1(
                x
            )
        )

        z = self.drop(
            self.c2(
                z
            )
        )

        return F.gelu(
            self.bn(
                x + z
            )
        )


class Expert(nn.Module):
    def __init__(
        self,
        dim,
        token=64,
    ):
        super().__init__()

        hidden = max(
            64,
            min(
                192,
                dim * 2,
            ),
        )

        self.cur = nn.Sequential(
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
                token,
            ),
            nn.GELU(),
        )

        self.tin = nn.Sequential(
            nn.LayerNorm(
                dim
            ),
            nn.Linear(
                dim,
                32,
            ),
            nn.GELU(),
        )

        self.tcn = nn.Sequential(
            TBlock(
                32,
                1,
            ),
            TBlock(
                32,
                2,
            ),
            TBlock(
                32,
                4,
            ),
        )

        self.tout = nn.Sequential(
            nn.Linear(
                64,
                token,
            ),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(
                token * 2,
                token,
            ),
            nn.GELU(),
            nn.LayerNorm(
                token
            ),
        )

    def forward(
        self,
        seq,
    ):
        current = self.cur(
            seq[
                :,
                -1
            ]
        )

        temporal = self.tin(
            seq
        ).transpose(
            1,
            2,
        )

        temporal = self.tcn(
            temporal
        )

        temporal = self.tout(
            torch.cat(
                [
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

        return self.fuse(
            torch.cat(
                [
                    current,
                    temporal,
                ],
                dim=1,
            )
        )


class TechnicalMoE(nn.Module):
    def __init__(
        self,
        groups,
    ):
        super().__init__()

        self.expert_names = list(
            groups
        )

        self.ids = [
            groups[key]
            for key in self.expert_names
        ]

        self.n = len(
            self.ids
        )

        d = 64

        self.experts = nn.ModuleList(
            [
                Expert(
                    len(ids),
                    d,
                )
                for ids
                in self.ids
            ]
        )

        self.identity = nn.Parameter(
            torch.randn(
                1,
                self.n,
                d,
            )
            * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=192,
            dropout=0.10,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.cross = nn.TransformerEncoder(
            layer,
            2,
        )

        self.router = nn.Sequential(
            nn.LayerNorm(
                d
            ),
            nn.Linear(
                d,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                self.n,
            ),
        )

        self.shared = nn.Sequential(
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

        self.long = nn.Sequential(
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
                64,
            ),
            nn.GELU(),
        )

        self.short = nn.Sequential(
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
                64,
            ),
            nn.GELU(),
        )

        self.le = nn.Linear(
            64,
            1,
        )

        self.se = nn.Linear(
            64,
            1,
        )

        self.lr = nn.Linear(
            64,
            1,
        )

        self.sr = nn.Linear(
            64,
            1,
        )

        self.lt = nn.Linear(
            64,
            1,
        )

        self.st = nn.Linear(
            64,
            1,
        )

        self.lu = nn.Linear(
            64,
            1,
        )

        self.su = nn.Linear(
            64,
            1,
        )

    def forward(
        self,
        seq,
    ):
        tokens = torch.stack(
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

        mixed = self.cross(
            tokens
            + self.identity
        )

        gates = torch.softmax(
            self.router(
                mixed.mean(
                    dim=1
                )
            ),
            dim=1,
        )

        context = (
            mixed
            * gates.unsqueeze(
                -1
            )
        ).sum(
            dim=1
        )

        shared = self.shared(
            context
        )

        long_z = self.long(
            shared
        )

        short_z = self.short(
            shared
        )

        def pair(
            long_head,
            short_head,
        ):
            return torch.cat(
                [
                    long_head(
                        long_z
                    ),

                    short_head(
                        short_z
                    ),
                ],
                dim=1,
            )

        return {
            'event':
                pair(
                    self.le,
                    self.se,
                ),

            'race':
                pair(
                    self.lr,
                    self.sr,
                ),

            'timeout':
                pair(
                    self.lt,
                    self.st,
                ),

            'utility':
                pair(
                    self.lu,
                    self.su,
                ),

            'gates':
                gates,
        }


def pos_weights(
    arrays,
    rows,
):
    decisive = arrays[
        'decisive'
    ][
        rows
    ]

    target = arrays[
        'tp_vs_sl'
    ][
        rows
    ]

    event_pos = decisive.sum(
        axis=0
    ).astype(
        np.float64
    )

    event_weight = (
        len(decisive)
        - event_pos
    ) / np.maximum(
        event_pos,
        1.0,
    )

    race_weight = []

    for side in range(
        2
    ):
        mask = (
            decisive[
                :,
                side
            ]
            > 0.5
        )

        pos = target[
            mask,
            side
        ].sum()

        neg = (
            mask.sum()
            - pos
        )

        race_weight.append(
            neg
            / max(
                float(pos),
                1.0,
            )
        )

    return (
        torch.tensor(
            event_weight,
            dtype=torch.float32,
        ),

        torch.tensor(
            race_weight,
            dtype=torch.float32,
        ),
    )


def ordinal_rank_loss(
    score,
    target,
    race,
):
    losses = []

    for side in range(
        2
    ):
        s = score[
            :,
            side
        ]

        t = target[
            :,
            side
        ]

        r = race[
            :,
            side
        ]

        tp = s[
            r == 1
        ]

        sl = s[
            r == 0
        ]

        # Hard TP vs hard SL.
        if (
            len(tp)
            and len(sl)
        ):
            k = min(
                48,
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
            ).values

            losses.append(
                F.softplus(
                    0.60
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

        # General ordinal ordering:
        # TP > good timeout >
        # neutral > bad timeout > SL.
        perm = torch.randperm(
            len(s),
            device=s.device,
        )

        diff = (
            t
            - t[
                perm
            ]
        )

        mask = (
            torch.abs(
                diff
            )
            >= 0.35
        )

        if mask.any():
            losses.append(
                F.softplus(
                    0.25
                    - torch.sign(
                        diff[
                            mask
                        ]
                    )
                    * (
                        s[
                            mask
                        ]
                        - s[
                            perm
                        ][
                            mask
                        ]
                    )
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


def side_loss(
    utility,
    pnl,
):
    diff = (
        pnl[:, 0]
        - pnl[:, 1]
    )

    mask = (
        torch.abs(
            diff
        )
        >= 3.0
    )

    if not mask.any():
        return (
            utility.sum()
            * 0.0
        )

    return (
        F.binary_cross_entropy_with_logits(
            utility[
                mask,
                0
            ]
            - utility[
                mask,
                1
            ],
            (
                diff[
                    mask
                ]
                > 0
            ).float(),
        )
    )


def balance_loss(
    gates,
):
    mean = gates.mean(
        dim=0
    )

    target = torch.full_like(
        mean,
        1.0
        / gates.shape[1],
    )

    return torch.square(
        mean
        - target
    ).mean()


def train_epoch(
    model,
    dl,
    optimizer,
    scaler,
    device,
    event_weight,
    race_weight,
):
    model.train()

    total = 0.0
    count = 0

    for (
        seq,
        decisive,
        tp_vs_sl,
        ordinal,
        pnl,
        race,
    ) in dl:

        seq = seq.to(
            device,
            non_blocking=True,
        )

        decisive = decisive.to(
            device
        )

        tp_vs_sl = tp_vs_sl.to(
            device
        )

        ordinal = ordinal.to(
            device
        )

        pnl = pnl.to(
            device
        )

        race = race.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=scaler.is_enabled()
        ):
            out = model(
                seq
            )

            loss_event = (
                F.binary_cross_entropy_with_logits(
                    out[
                        'event'
                    ],
                    decisive,
                    pos_weight=event_weight,
                )
            )

            race_losses = []
            timeout_losses = []

            for side in range(
                2
            ):
                decisive_mask = (
                    decisive[
                        :,
                        side
                    ].bool()
                )

                if decisive_mask.any():
                    race_losses.append(
                        F.binary_cross_entropy_with_logits(
                            out[
                                'race'
                            ][
                                decisive_mask,
                                side
                            ],
                            tp_vs_sl[
                                decisive_mask,
                                side
                            ],
                            pos_weight=race_weight[
                                side
                            ],
                        )
                    )

                timeout_mask = (
                    ~decisive_mask
                )

                if timeout_mask.any():
                    timeout_target = torch.clamp(
                        pnl[
                            timeout_mask,
                            side
                        ]
                        / 10.0,
                        -2.0,
                        2.0,
                    )

                    timeout_losses.append(
                        F.smooth_l1_loss(
                            out[
                                'timeout'
                            ][
                                timeout_mask,
                                side
                            ],
                            timeout_target,
                        )
                    )

            loss_race = (
                sum(
                    race_losses
                )
                / len(
                    race_losses
                )
            )

            loss_timeout = (
                sum(
                    timeout_losses
                )
                / len(
                    timeout_losses
                )
            )

            loss_utility = (
                F.smooth_l1_loss(
                    out[
                        'utility'
                    ],
                    ordinal,
                )
            )

            loss_rank = ordinal_rank_loss(
                out[
                    'utility'
                ],
                ordinal,
                race,
            )

            loss_side = side_loss(
                out[
                    'utility'
                ],
                pnl,
            )

            loss_balance = balance_loss(
                out[
                    'gates'
                ]
            )

            loss = (
                0.35
                * loss_event

                + 1.00
                * loss_race

                + 0.10
                * loss_timeout

                + 0.30
                * loss_utility

                + 0.70
                * loss_rank

                + 0.35
                * loss_side

                + 0.05
                * loss_balance
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
            * len(seq)
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
        key: []
        for key in (
            'event',
            'race',
            'timeout',
            'utility',
            'gates',
            'pnl',
            'race_true',
        )
    }

    for (
        seq,
        decisive,
        tp_vs_sl,
        ordinal,
        pnl,
        race,
    ) in dl:

        pred = model(
            seq.to(
                device,
                non_blocking=True,
            )
        )

        for key in (
            'event',
            'race',
            'timeout',
            'utility',
            'gates',
        ):
            out[
                key
            ].append(
                pred[
                    key
                ].cpu().numpy()
            )

        out[
            'pnl'
        ].append(
            pnl.numpy()
        )

        out[
            'race_true'
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


def sigmoid(x):
    return (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    x,
                    -30.0,
                    30.0,
                )
            )
        )
    )


def pf(x):
    gains = x[
        x > 0
    ].sum()

    losses = -x[
        x < 0
    ].sum()

    if losses <= 0:
        return np.inf

    return float(
        gains
        / losses
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


def scores(
    pred,
):
    p_event = sigmoid(
        pred[
            'event'
        ]
    )

    p_race = sigmoid(
        pred[
            'race'
        ]
    )

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

    timeout = (
        10.0
        * np.tanh(
            pred[
                'timeout'
            ]
        )
    )

    ev = (
        p_event
        * (
            p_race
            * TP_VALUE

            + (
                1.0
                - p_race
            )
            * SL_VALUE
        )

        + (
            1.0
            - p_event
        )
        * timeout
    )

    return {
        'p_event':
            p_event,

        'p_race':
            p_race,

        'p_tp':
            p_tp,

        'p_sl':
            p_sl,

        'ev':
            ev,

        'utility':
            pred[
                'utility'
            ],
    }


def side_metric(
    score,
    pnl,
):
    values = []

    for side in range(
        2
    ):
        value = 0.0

        for coverage, weight in (
            (
                2.0,
                0.30,
            ),
            (
                1.0,
                0.40,
            ),
            (
                0.5,
                0.30,
            ),
        ):
            n = max(
                1,
                round(
                    len(score)
                    * coverage
                    / 100.0
                ),
            )

            idx = np.argpartition(
                score[
                    :,
                    side
                ],
                -n,
            )[
                -n:
            ]

            value += (
                weight
                * float(
                    pnl[
                        idx,
                        side
                    ].mean()
                )
            )

        values.append(
            value
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        values,
    )


def report_policy(
    name,
    policy,
    pred,
):
    long_side = (
        policy[:, 0]
        >= policy[:, 1]
    )

    rank = np.maximum(
        policy[:, 0],
        policy[:, 1],
    )

    pnl = np.where(
        long_side,
        pred[
            'pnl'
        ][:, 0],
        pred[
            'pnl'
        ][:, 1],
    )

    race = np.where(
        long_side,
        pred[
            'race_true'
        ][:, 0],
        pred[
            'race_true'
        ][:, 1],
    )

    print(
        f'  [{name}]'
    )

    for coverage in (
        5.0,
        2.0,
        1.0,
        0.5,
        0.2,
    ):
        n = max(
            1,
            round(
                len(rank)
                * coverage
                / 100.0
            ),
        )

        idx = np.argpartition(
            rank,
            -n,
        )[
            -n:
        ]

        x = pnl[
            idx
        ]

        print(
            f'   {coverage:>4.1f}% '
            f'n={n:>5} '
            f'TP={(race[idx] == 1).mean():>6.2%} '
            f'SL={(race[idx] == 0).mean():>6.2%} '
            f'WIN={(x > 0).mean():>6.2%} '
            f'mean={x.mean():>+7.3f} '
            f'PF={pf(x):>6.3f} '
            f'LONG={long_side[idx].mean():>6.2%}'
        )


def evaluate(
    name,
    pred,
    expert_names,
):
    s = scores(
        pred
    )

    race = pred[
        'race_true'
    ]

    print()
    print(
        name
    )

    print(
        '-' * 122
    )

    for side, label in enumerate(
        (
            'LONG',
            'SHORT',
        )
    ):
        y = (
            race[
                :,
                side
            ]
            == 1
        ).astype(
            np.int8
        )

        decisive = (
            race[
                :,
                side
            ]
            != -1
        )

        race_y = (
            race[
                decisive,
                side
            ]
            == 1
        ).astype(
            np.int8
        )

        print(
            f'{label:<5} '
            f'PTP_AP='
            f'{safe_ap(y, s["p_tp"][:, side]):.4f} '
            f'PTP_AUC='
            f'{safe_auc(y, s["p_tp"][:, side]):.4f} '
            f'RACE_AP='
            f'{safe_ap(race_y, s["p_race"][decisive, side]):.4f} '
            f'RACE_AUC='
            f'{safe_auc(race_y, s["p_race"][decisive, side]):.4f}'
        )

    selection, values = side_metric(
        s[
            'utility'
        ],
        pred[
            'pnl'
        ],
    )

    print(
        'SIDE-BALANCED UTILITY:',
        f'{selection:+.4f}',
        f'(LONG={values[0]:+.4f}, '
        f'SHORT={values[1]:+.4f})',
    )

    report_policy(
        'UTILITY',
        s[
            'utility'
        ],
        pred,
    )

    report_policy(
        'EXPECTED_VALUE',
        s[
            'ev'
        ],
        pred,
    )

    gates = pred[
        'gates'
    ].mean(
        axis=0
    )

    order = np.argsort(
        gates
    )[
        ::-1
    ]

    entropy = float(
        (
            -pred[
                'gates'
            ]
            * np.log(
                np.maximum(
                    pred[
                        'gates'
                    ],
                    1e-12,
                )
            )
        )
        .sum(
            axis=1
        )
        .mean()
    )

    print(
        'Expert gates:',
        ', '.join(
            f'{expert_names[i]}'
            f'={gates[i]:.3f}'
            for i in order
        ),
    )

    print(
        'Router entropy:',
        f'{entropy:.4f}',
    )

    return selection


def main():
    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print(
        'TEN V6.5.2 '
        'NON-OVERLAPPING TECHNICAL '
        'MIXTURE-OF-EXPERTS'
    )

    print(
        '=' * 122
    )

    print(
        'Device:',
        device,
    )

    if torch.cuda.is_available():
        print(
            'GPU:',
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
        'train',
        'val',
        'test2025',
        'reserved2026',
    ):
        print(
            f'{key.upper():<12}: '
            f'{len(split[key]):,}'
        )

    print(
        'Features:',
        len(names),
    )

    print(
        'Sequence:',
        SEQ,
        'M5 =',
        SEQ * 5,
        'minutes',
    )

    print(
        'Experts:',
        {
            key:
                len(value)
            for key, value
            in groups.items()
        },
    )

    largest = max(
        groups,
        key=lambda key:
            len(
                groups[key]
            ),
    )

    fraction = (
        len(
            groups[
                largest
            ]
        )
        / len(names)
    )

    print(
        'Largest expert:',
        largest,
        f'{fraction:.2%}',
    )

    if fraction > 0.40:
        print(
            'WARNING: largest expert '
            'exceeds 40% of features.'
        )

    with open(
        OUT
        / 'expert_groups_v652.json',
        'w',
    ) as f:
        json.dump(
            groups,
            f,
            indent=2,
        )

    residual = [
        names[i]
        for i in groups.get(
            'residual_context',
            [],
        )
    ]

    with open(
        OUT
        / 'residual_features_v652.json',
        'w',
    ) as f:
        json.dump(
            residual,
            f,
            indent=2,
        )

    np.savez(
        OUT
        / 'normalization_v652.npz',
        mean=mean,
        std=std,
    )

    train_dl = loader(
        split[
            'train'
        ],
        True,
        arrays,
        mean,
        std,
    )

    val_dl = loader(
        split[
            'val'
        ],
        False,
        arrays,
        mean,
        std,
    )

    test_dl = loader(
        split[
            'test2025'
        ],
        False,
        arrays,
        mean,
        std,
    )

    event_weight, race_weight = (
        pos_weights(
            arrays,
            split[
                'train'
            ],
        )
    )

    event_weight = (
        event_weight.to(
            device
        )
    )

    race_weight = (
        race_weight.to(
            device
        )
    )

    print(
        'Event pos weights:',
        event_weight.cpu().numpy(),
    )

    print(
        'Race TP pos weights:',
        race_weight.cpu().numpy(),
    )

    model = TechnicalMoE(
        groups
    ).to(
        device
    )

    print(
        'Parameters:',
        f'{sum(p.numel() for p in model.parameters()):,}',
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
            mode='max',
            factor=0.5,
            patience=1,
            min_lr=1e-5,
        )
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            device.type
            == 'cuda'
        )
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
            event_weight,
            race_weight,
        )

        val_pred = predict(
            model,
            val_dl,
            device,
        )

        selection = evaluate(
            f'EPOCH {epoch} VALIDATION',
            val_pred,
            model.expert_names,
        )

        scheduler.step(
            selection
        )

        lr = optimizer.param_groups[
            0
        ][
            'lr'
        ]

        print(
            f'Epoch {epoch:02d} '
            f'| loss={loss:.5f} '
            f'| selection={selection:+.4f} '
            f'| lr={lr:.2e}'
        )

        if selection > (
            best + 1e-4
        ):
            best = selection
            bad = 0

            torch.save(
                {
                    'epoch':
                        epoch,

                    'model':
                        model.state_dict(),

                    'val_selection':
                        selection,

                    'groups':
                        groups,

                    'feature_names':
                        names,

                    'seq':
                        SEQ,
                },
                OUT
                / 'best_technical_moe_v652.pt',
            )

            print(
                'NEW V6.5.2 '
                'TECHNICAL CHAMPION'
            )

        else:
            bad += 1

            if bad >= PATIENCE:
                print(
                    'EARLY STOP'
                )
                break

    checkpoint = torch.load(
        OUT
        / 'best_technical_moe_v652.pt',
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            'model'
        ]
    )

    print()
    print(
        '=' * 122
    )

    print(
        'FROZEN V6.5.2 '
        'TECHNICAL CHAMPION'
    )

    print(
        '=' * 122
    )

    print(
        'Epoch:',
        checkpoint[
            'epoch'
        ],
    )

    print(
        'VAL selection:',
        f'{checkpoint["val_selection"]:+.4f}',
    )

    evaluate(
        'FINAL 2023-2024',
        predict(
            model,
            val_dl,
            device,
        ),
        model.expert_names,
    )

    print()
    print(
        '=' * 122
    )

    print(
        '2025 OUT-OF-TIME TEST'
    )

    print(
        '=' * 122
    )

    evaluate(
        '2025 V6.5.2 TECHNICAL MOE',
        predict(
            model,
            test_dl,
            device,
        ),
        model.expert_names,
    )

    print(
        '2026 RESERVED: '
        'NOT EVALUATED BY V6.5.2 TRAINER'
    )


if __name__ == '__main__':
    main()
