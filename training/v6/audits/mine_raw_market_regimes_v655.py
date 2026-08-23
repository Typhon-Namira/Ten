from pathlib import Path
import json
from itertools import combinations

import numpy as np
import pandas as pd
import torch

import training.v6.models.train_technical_moe_v652 as v


OUT = Path(
    "training/artifacts/v6/"
    "raw_market_regime_miner_v655"
)

CKPT = Path(
    "training/artifacts/v6/"
    "technical_moe_v652/"
    "best_technical_moe_v652.pt"
)

SCORE_QS = (
    0.95,
    0.98,
    0.99,
    0.995,
    0.998,
)

POLICIES = (
    "p_tp",
    "p_race",
)

MIN_TRAIN = 150
MIN_VAL = 50

BE_TP_RES = (
    -v.SL_VALUE
    / (
        v.TP_VALUE
        - v.SL_VALUE
    )
)


def pf(x):
    gain = x[
        x > 0
    ].sum()

    loss = -x[
        x < 0
    ].sum()

    if loss <= 0:
        return np.inf

    return float(
        gain / loss
    )


def metrics(
    mask,
    pnl,
    race,
):
    idx = np.flatnonzero(
        mask
    )

    if len(idx) == 0:
        return {
            "n": 0,
            "tp": np.nan,
            "sl": np.nan,
            "timeout": np.nan,
            "tp_res": np.nan,
            "win": np.nan,
            "mean": np.nan,
            "pf": np.nan,
        }

    x = pnl[
        idx
    ]

    r = race[
        idx
    ]

    tp = float(
        (
            r == 1
        ).mean()
    )

    sl = float(
        (
            r == 0
        ).mean()
    )

    timeout = float(
        (
            r == -1
        ).mean()
    )

    resolved = (
        tp + sl
    )

    tp_res = (
        tp / resolved
        if resolved > 0
        else np.nan
    )

    return {
        "n":
            int(
                len(idx)
            ),

        "tp":
            tp,

        "sl":
            sl,

        "timeout":
            timeout,

        "tp_res":
            tp_res,

        "win":
            float(
                (
                    x > 0
                ).mean()
            ),

        "mean":
            float(
                x.mean()
            ),

        "pf":
            pf(
                x
            ),
    }


def robust_scale_triplet(
    train,
    val,
    test,
):
    med = np.nanmedian(
        train
    )

    q25, q75 = np.nanquantile(
        train,
        [
            0.25,
            0.75,
        ],
    )

    scale = max(
        float(
            q75 - q25
        ),
        1e-6,
    )

    return (
        (
            train - med
        ) / scale,

        (
            val - med
        ) / scale,

        (
            test - med
        ) / scale,
    )


def composite(
    features,
    names,
):
    train_x = features[
        "train"
    ]

    val_x = features[
        "val"
    ]

    test_x = features[
        "test"
    ]

    index = features[
        "index"
    ]

    tr = []
    va = []
    te = []

    for name in names:
        i = index[
            name
        ]

        a, b, c = (
            robust_scale_triplet(
                train_x[
                    :,
                    i
                ],
                val_x[
                    :,
                    i
                ],
                test_x[
                    :,
                    i
                ],
            )
        )

        tr.append(a)
        va.append(b)
        te.append(c)

    return (
        np.nanmean(
            np.column_stack(
                tr
            ),
            axis=1,
        ),

        np.nanmean(
            np.column_stack(
                va
            ),
            axis=1,
        ),

        np.nanmean(
            np.column_stack(
                te
            ),
            axis=1,
        ),
    )


def raw_triplet(
    features,
    name,
):
    i = features[
        "index"
    ][
        name
    ]

    return (
        features[
            "train"
        ][
            :,
            i
        ],

        features[
            "val"
        ][
            :,
            i
        ],

        features[
            "test"
        ][
            :,
            i
        ],
    )


def max_triplet(
    features,
    names,
):
    arrays = [
        raw_triplet(
            features,
            name,
        )
        for name in names
    ]

    return tuple(
        np.nanmax(
            np.column_stack(
                [
                    x[k]
                    for x in arrays
                ]
            ),
            axis=1,
        )
        for k in range(
            3
        )
    )


def min_abs_triplet(
    features,
    names,
):
    arrays = [
        raw_triplet(
            features,
            name,
        )
        for name in names
    ]

    return tuple(
        np.nanmin(
            np.abs(
                np.column_stack(
                    [
                        x[k]
                        for x in arrays
                    ]
                )
            ),
            axis=1,
        )
        for k in range(
            3
        )
    )


class RegimeBook:
    def __init__(self):
        self.items = []

    def add(
        self,
        name,
        dimension,
        train,
        val,
        test,
    ):
        self.items.append(
            {
                "name":
                    name,

                "dimension":
                    dimension,

                "train":
                    np.asarray(
                        train,
                        dtype=bool,
                    ),

                "val":
                    np.asarray(
                        val,
                        dtype=bool,
                    ),

                "test":
                    np.asarray(
                        test,
                        dtype=bool,
                    ),
            }
        )

    def tertiles(
        self,
        prefix,
        dimension,
        train,
        val,
        test,
    ):
        low, high = np.nanquantile(
            train,
            [
                0.33,
                0.67,
            ],
        )

        self.add(
            prefix + "_LOW",
            dimension,
            train <= low,
            val <= low,
            test <= low,
        )

        self.add(
            prefix + "_MID",
            dimension,
            (
                (train > low)
                & (
                    train < high
                )
            ),
            (
                (val > low)
                & (
                    val < high
                )
            ),
            (
                (test > low)
                & (
                    test < high
                )
            ),
        )

        self.add(
            prefix + "_HIGH",
            dimension,
            train >= high,
            val >= high,
            test >= high,
        )

        return {
            "low":
                float(low),

            "high":
                float(high),
        }

    def high(
        self,
        name,
        dimension,
        train,
        val,
        test,
        q=0.67,
    ):
        threshold = float(
            np.nanquantile(
                train,
                q,
            )
        )

        self.add(
            name,
            dimension,
            train >= threshold,
            val >= threshold,
            test >= threshold,
        )

        return threshold

    def positive_event(
        self,
        name,
        dimension,
        train,
        val,
        test,
        q=0.50,
    ):
        positive = train[
            train > 0
        ]

        threshold = (
            float(
                np.nanquantile(
                    positive,
                    q,
                )
            )
            if len(
                positive
            )
            else 0.0
        )

        threshold = max(
            threshold,
            1e-12,
        )

        self.add(
            name,
            dimension,
            train >= threshold,
            val >= threshold,
            test >= threshold,
        )

        return threshold


def required_features():
    names = [
        # HTF / MTF structure
        "m15v65_structure_score_v65",
        "h1v65_structure_score_v65",
        "h4v65_structure_score_v65",

        "mtf_structure_up_agreement",
        "mtf_structure_down_agreement",
        "mtf_structure_conflict",

        # Trend efficiency
        "m15v65_trend_efficiency_24",
        "h1v65_trend_efficiency_24",
        "h4v65_trend_efficiency_24",

        # Impulse
        "m15v65_impulse_balance_12",
        "h1v65_impulse_balance_12",
        "h4v65_impulse_balance_12",

        # Volatility
        "m5v65_volatility_ratio_v65",
        "m15v65_volatility_ratio_v65",
        "h1v65_volatility_ratio_v65",

        "m5v65_vol_of_vol_24",
        "m15v65_vol_of_vol_24",
        "h1v65_vol_of_vol_24",

        # Compression / expansion
        "m5_compression_score",
        "m15_compression_score",
        "h1_compression_score",

        "m5_expansion_score",
        "m15_expansion_score",
        "h1_expansion_score",

        "m5v65_body_expansion",
        "m15v65_body_expansion",

        # S/R distance
        "m5v651_support_distance_atr",
        "m15v651_support_distance_atr",
        "h1v651_support_distance_atr",

        "m5v651_resistance_distance_atr",
        "m15v651_resistance_distance_atr",
        "h1v651_resistance_distance_atr",

        # S/R strength
        "m5v651_support_strength",
        "m15v651_support_strength",
        "h1v651_support_strength",

        "m5v651_resistance_strength",
        "m15v651_resistance_strength",
        "h1v651_resistance_strength",

        # Breakout
        "m5v651_recent_breakout_up_strength",
        "m15v651_recent_breakout_up_strength",
        "h1v651_recent_breakout_up_strength",

        "m5v651_recent_breakout_down_strength",
        "m15v651_recent_breakout_down_strength",
        "h1v651_recent_breakout_down_strength",

        # Retest
        "m5v651_retest_long_quality",
        "m15v651_retest_long_quality",
        "h1v651_retest_long_quality",

        "m5v651_retest_short_quality",
        "m15v651_retest_short_quality",
        "h1v651_retest_short_quality",

        # Acceptance / failure
        "m5v651_breakout_long_acceptance",
        "m15v651_breakout_long_acceptance",
        "h1v651_breakout_long_acceptance",

        "m5v651_breakout_short_acceptance",
        "m15v651_breakout_short_acceptance",
        "h1v651_breakout_short_acceptance",

        "m5v651_breakout_long_failure",
        "m15v651_breakout_long_failure",
        "h1v651_breakout_long_failure",

        "m5v651_breakout_short_failure",
        "m15v651_breakout_short_failure",
        "h1v651_breakout_short_failure",

        # Swing sweeps
        "m5v65_sweep_swing_high",
        "m15v65_sweep_swing_high",
        "h1v65_sweep_swing_high",

        "m5v65_sweep_swing_low",
        "m15v65_sweep_swing_low",
        "h1v65_sweep_swing_low",

        # Session state from V6.5A
        "session_is_london",
        "session_is_new_york",
        "session_is_overlap",

        # Session dynamics
        "session_v651_london_vs_asia_range",
        "session_v651_ny_vs_london_range",

        "session_v651_sweep_asia_high",
        "session_v651_sweep_asia_low",

        "session_v651_ny_sweep_london_high",
        "session_v651_ny_sweep_london_low",

        "session_v651_break_asia_high",
        "session_v651_break_asia_low",

        "session_v651_ny_break_london_high",
        "session_v651_ny_break_london_low",
    ]

    return sorted(
        set(
            names
        )
    )


def extract_features(
    arrays,
    split,
    all_names,
):
    needed = required_features()

    name_to_full = {
        name: i
        for i, name
        in enumerate(
            all_names
        )
    }

    missing = [
        name
        for name in needed
        if name not in name_to_full
    ]

    if missing:
        print(
            "MISSING FEATURES:"
        )

        for name in missing:
            print(
                " ",
                name,
            )

        raise RuntimeError(
            "Required V6.5.5 features missing"
        )

    full_ids = np.array(
        [
            name_to_full[
                name
            ]
            for name in needed
        ],
        dtype=np.int64,
    )

    local_index = {
        name: i
        for i, name
        in enumerate(
            needed
        )
    }

    out = {
        "index":
            local_index,

        "names":
            needed,
    }

    for source_name, rows in (
        (
            "train",
            split[
                "train"
            ],
        ),
        (
            "val",
            split[
                "val"
            ],
        ),
        (
            "test",
            split[
                "test2025"
            ],
        ),
    ):
        source = arrays[
            "source"
        ][
            rows
        ]

        out[
            source_name
        ] = np.asarray(
            arrays[
                "features"
            ][
                source
            ][
                :,
                full_ids
            ],
            dtype=np.float32,
        )

    return out


def build_regimes(
    features,
):
    book = RegimeBook()

    thresholds = {}

    # --------------------------------------------------
    # H1/H4 directional structure
    # --------------------------------------------------

    h1 = raw_triplet(
        features,
        "h1v65_structure_score_v65",
    )

    h4 = raw_triplet(
        features,
        "h4v65_structure_score_v65",
    )

    for k, split_name in enumerate(
        (
            "train",
            "val",
            "test",
        )
    ):
        pass

    book.add(
        "HTF_BULL",
        "htf_direction",
        (
            (h1[0] > 0)
            & (
                h4[0] > 0
            )
        ),
        (
            (h1[1] > 0)
            & (
                h4[1] > 0
            )
        ),
        (
            (h1[2] > 0)
            & (
                h4[2] > 0
            )
        ),
    )

    book.add(
        "HTF_BEAR",
        "htf_direction",
        (
            (h1[0] < 0)
            & (
                h4[0] < 0
            )
        ),
        (
            (h1[1] < 0)
            & (
                h4[1] < 0
            )
        ),
        (
            (h1[2] < 0)
            & (
                h4[2] < 0
            )
        ),
    )

    book.add(
        "HTF_MIXED",
        "htf_direction",
        ~(
            (h1[0] > 0)
            & (
                h4[0] > 0
            )
        )
        & ~(
            (h1[0] < 0)
            & (
                h4[0] < 0
            )
        ),
        ~(
            (h1[1] > 0)
            & (
                h4[1] > 0
            )
        )
        & ~(
            (h1[1] < 0)
            & (
                h4[1] < 0
            )
        ),
        ~(
            (h1[2] > 0)
            & (
                h4[2] > 0
            )
        )
        & ~(
            (h1[2] < 0)
            & (
                h4[2] < 0
            )
        ),
    )

    # --------------------------------------------------
    # MTF alignment
    # --------------------------------------------------

    for name, label in (
        (
            "mtf_structure_up_agreement",
            "MTF_UP_STRONG",
        ),
        (
            "mtf_structure_down_agreement",
            "MTF_DOWN_STRONG",
        ),
        (
            "mtf_structure_conflict",
            "MTF_CONFLICT_HIGH",
        ),
    ):
        x = raw_triplet(
            features,
            name,
        )

        thresholds[
            label
        ] = book.high(
            label,
            "mtf_structure",
            *x,
            q=0.67,
        )

    # --------------------------------------------------
    # Volatility regime
    # --------------------------------------------------

    volatility = composite(
        features,
        [
            "m5v65_volatility_ratio_v65",
            "m15v65_volatility_ratio_v65",
            "h1v65_volatility_ratio_v65",
            "m5v65_vol_of_vol_24",
            "m15v65_vol_of_vol_24",
            "h1v65_vol_of_vol_24",
        ],
    )

    thresholds[
        "VOLATILITY"
    ] = book.tertiles(
        "VOL",
        "volatility",
        *volatility,
    )

    # --------------------------------------------------
    # Trend efficiency
    # --------------------------------------------------

    efficiency = composite(
        features,
        [
            "m15v65_trend_efficiency_24",
            "h1v65_trend_efficiency_24",
            "h4v65_trend_efficiency_24",
        ],
    )

    thresholds[
        "TREND_EFF"
    ] = book.tertiles(
        "TREND_EFF",
        "trend_efficiency",
        *efficiency,
    )

    # --------------------------------------------------
    # Signed impulse regime
    # --------------------------------------------------

    impulse = composite(
        features,
        [
            "m15v65_impulse_balance_12",
            "h1v65_impulse_balance_12",
            "h4v65_impulse_balance_12",
        ],
    )

    thresholds[
        "IMPULSE"
    ] = book.tertiles(
        "IMPULSE",
        "impulse",
        *impulse,
    )

    # --------------------------------------------------
    # Compression / expansion
    # --------------------------------------------------

    compression = composite(
        features,
        [
            "m5_compression_score",
            "m15_compression_score",
            "h1_compression_score",
        ],
    )

    thresholds[
        "COMPRESSION_HIGH"
    ] = book.high(
        "COMPRESSION_HIGH",
        "compression",
        *compression,
        q=0.67,
    )

    expansion = composite(
        features,
        [
            "m5_expansion_score",
            "m15_expansion_score",
            "h1_expansion_score",
            "m5v65_body_expansion",
            "m15v65_body_expansion",
        ],
    )

    thresholds[
        "EXPANSION_HIGH"
    ] = book.high(
        "EXPANSION_HIGH",
        "expansion",
        *expansion,
        q=0.67,
    )


    # --------------------------------------------------
    # S/R location and strength
    # --------------------------------------------------

    near_support = min_abs_triplet(
        features,
        [
            "m5v651_support_distance_atr",
            "m15v651_support_distance_atr",
            "h1v651_support_distance_atr",
        ],
    )

    support_cut = float(
        np.nanquantile(
            near_support[0],
            0.33,
        )
    )

    book.add(
        "NEAR_SUPPORT",
        "sr_location",
        near_support[0] <= support_cut,
        near_support[1] <= support_cut,
        near_support[2] <= support_cut,
    )

    thresholds[
        "NEAR_SUPPORT"
    ] = support_cut

    near_resistance = min_abs_triplet(
        features,
        [
            "m5v651_resistance_distance_atr",
            "m15v651_resistance_distance_atr",
            "h1v651_resistance_distance_atr",
        ],
    )

    resistance_cut = float(
        np.nanquantile(
            near_resistance[0],
            0.33,
        )
    )

    book.add(
        "NEAR_RESISTANCE",
        "sr_location",
        near_resistance[0]
        <= resistance_cut,
        near_resistance[1]
        <= resistance_cut,
        near_resistance[2]
        <= resistance_cut,
    )

    thresholds[
        "NEAR_RESISTANCE"
    ] = resistance_cut

    strong_support = max_triplet(
        features,
        [
            "m5v651_support_strength",
            "m15v651_support_strength",
            "h1v651_support_strength",
        ],
    )

    thresholds[
        "STRONG_SUPPORT"
    ] = book.high(
        "STRONG_SUPPORT",
        "sr_strength",
        *strong_support,
        q=0.67,
    )

    strong_resistance = max_triplet(
        features,
        [
            "m5v651_resistance_strength",
            "m15v651_resistance_strength",
            "h1v651_resistance_strength",
        ],
    )

    thresholds[
        "STRONG_RESISTANCE"
    ] = book.high(
        "STRONG_RESISTANCE",
        "sr_strength",
        *strong_resistance,
        q=0.67,
    )

    # --------------------------------------------------
    # Breakout / retest
    # --------------------------------------------------

    breakout_long = max_triplet(
        features,
        [
            "m5v651_recent_breakout_up_strength",
            "m15v651_recent_breakout_up_strength",
            "h1v651_recent_breakout_up_strength",
        ],
    )

    thresholds[
        "BREAKOUT_LONG_STRONG"
    ] = book.positive_event(
        "BREAKOUT_LONG_STRONG",
        "breakout",
        *breakout_long,
    )

    breakout_short = max_triplet(
        features,
        [
            "m5v651_recent_breakout_down_strength",
            "m15v651_recent_breakout_down_strength",
            "h1v651_recent_breakout_down_strength",
        ],
    )

    thresholds[
        "BREAKOUT_SHORT_STRONG"
    ] = book.positive_event(
        "BREAKOUT_SHORT_STRONG",
        "breakout",
        *breakout_short,
    )

    retest_long = max_triplet(
        features,
        [
            "m5v651_retest_long_quality",
            "m15v651_retest_long_quality",
            "h1v651_retest_long_quality",
        ],
    )

    thresholds[
        "RETEST_LONG_STRONG"
    ] = book.positive_event(
        "RETEST_LONG_STRONG",
        "retest",
        *retest_long,
    )

    retest_short = max_triplet(
        features,
        [
            "m5v651_retest_short_quality",
            "m15v651_retest_short_quality",
            "h1v651_retest_short_quality",
        ],
    )

    thresholds[
        "RETEST_SHORT_STRONG"
    ] = book.positive_event(
        "RETEST_SHORT_STRONG",
        "retest",
        *retest_short,
    )

    acceptance_long = max_triplet(
        features,
        [
            "m5v651_breakout_long_acceptance",
            "m15v651_breakout_long_acceptance",
            "h1v651_breakout_long_acceptance",
        ],
    )

    thresholds[
        "LONG_ACCEPTANCE"
    ] = book.high(
        "LONG_ACCEPTANCE",
        "acceptance",
        *acceptance_long,
        q=0.80,
    )

    acceptance_short = max_triplet(
        features,
        [
            "m5v651_breakout_short_acceptance",
            "m15v651_breakout_short_acceptance",
            "h1v651_breakout_short_acceptance",
        ],
    )

    thresholds[
        "SHORT_ACCEPTANCE"
    ] = book.high(
        "SHORT_ACCEPTANCE",
        "acceptance",
        *acceptance_short,
        q=0.80,
    )

    failure_long = max_triplet(
        features,
        [
            "m5v651_breakout_long_failure",
            "m15v651_breakout_long_failure",
            "h1v651_breakout_long_failure",
        ],
    )

    thresholds[
        "LONG_BREAK_FAILURE"
    ] = book.positive_event(
        "LONG_BREAK_FAILURE",
        "failure",
        *failure_long,
    )

    failure_short = max_triplet(
        features,
        [
            "m5v651_breakout_short_failure",
            "m15v651_breakout_short_failure",
            "h1v651_breakout_short_failure",
        ],
    )

    thresholds[
        "SHORT_BREAK_FAILURE"
    ] = book.positive_event(
        "SHORT_BREAK_FAILURE",
        "failure",
        *failure_short,
    )

    # --------------------------------------------------
    # Liquidity sweep
    # --------------------------------------------------

    sweep_high = max_triplet(
        features,
        [
            "m5v65_sweep_swing_high",
            "m15v65_sweep_swing_high",
            "h1v65_sweep_swing_high",
            "session_v651_sweep_asia_high",
            "session_v651_ny_sweep_london_high",
        ],
    )

    thresholds[
        "SWEEP_HIGH"
    ] = book.positive_event(
        "SWEEP_HIGH",
        "liquidity",
        *sweep_high,
    )

    sweep_low = max_triplet(
        features,
        [
            "m5v65_sweep_swing_low",
            "m15v65_sweep_swing_low",
            "h1v65_sweep_swing_low",
            "session_v651_sweep_asia_low",
            "session_v651_ny_sweep_london_low",
        ],
    )

    thresholds[
        "SWEEP_LOW"
    ] = book.positive_event(
        "SWEEP_LOW",
        "liquidity",
        *sweep_low,
    )

    # --------------------------------------------------
    # Sessions
    # --------------------------------------------------

    london = raw_triplet(
        features,
        "session_is_london",
    )

    ny = raw_triplet(
        features,
        "session_is_new_york",
    )

    overlap = raw_triplet(
        features,
        "session_is_overlap",
    )

    book.add(
        "SESSION_LONDON",
        "session",
        (
            (london[0] > 0.5)
            & (
                overlap[0] < 0.5
            )
        ),
        (
            (london[1] > 0.5)
            & (
                overlap[1] < 0.5
            )
        ),
        (
            (london[2] > 0.5)
            & (
                overlap[2] < 0.5
            )
        ),
    )

    book.add(
        "SESSION_NY",
        "session",
        (
            (ny[0] > 0.5)
            & (
                overlap[0] < 0.5
            )
        ),
        (
            (ny[1] > 0.5)
            & (
                overlap[1] < 0.5
            )
        ),
        (
            (ny[2] > 0.5)
            & (
                overlap[2] < 0.5
            )
        ),
    )

    book.add(
        "SESSION_OVERLAP",
        "session",
        overlap[0] > 0.5,
        overlap[1] > 0.5,
        overlap[2] > 0.5,
    )

    book.add(
        "SESSION_OTHER",
        "session",
        (
            (london[0] < 0.5)
            & (
                ny[0] < 0.5
            )
        ),
        (
            (london[1] < 0.5)
            & (
                ny[1] < 0.5
            )
        ),
        (
            (london[2] < 0.5)
            & (
                ny[2] < 0.5
            )
        ),
    )

    london_expansion = raw_triplet(
        features,
        "session_v651_london_vs_asia_range",
    )

    positive = london_expansion[0][
        london_expansion[0] > 0
    ]

    london_cut = (
        float(
            np.nanquantile(
                positive,
                0.67,
            )
        )
        if len(positive)
        else np.inf
    )

    book.add(
        "LONDON_RANGE_EXPANDED",
        "session_expansion",
        london_expansion[0]
        >= london_cut,
        london_expansion[1]
        >= london_cut,
        london_expansion[2]
        >= london_cut,
    )

    thresholds[
        "LONDON_RANGE_EXPANDED"
    ] = london_cut

    ny_expansion = raw_triplet(
        features,
        "session_v651_ny_vs_london_range",
    )

    positive = ny_expansion[0][
        ny_expansion[0] > 0
    ]

    ny_cut = (
        float(
            np.nanquantile(
                positive,
                0.67,
            )
        )
        if len(positive)
        else np.inf
    )

    book.add(
        "NY_RANGE_EXPANDED",
        "session_expansion",
        ny_expansion[0]
        >= ny_cut,
        ny_expansion[1]
        >= ny_cut,
        ny_expansion[2]
        >= ny_cut,
    )

    thresholds[
        "NY_RANGE_EXPANDED"
    ] = ny_cut

    return (
        book,
        thresholds,
    )


def side_regimes(
    book,
    side,
):
    if side == 0:
        forbidden = (
            "BREAKOUT_SHORT",
            "RETEST_SHORT",
            "SHORT_ACCEPTANCE",
            "SHORT_BREAK_FAILURE",
            "NEAR_RESISTANCE",
            "STRONG_RESISTANCE",
            "SWEEP_HIGH",
        )

    else:
        forbidden = (
            "BREAKOUT_LONG",
            "RETEST_LONG",
            "LONG_ACCEPTANCE",
            "LONG_BREAK_FAILURE",
            "NEAR_SUPPORT",
            "STRONG_SUPPORT",
            "SWEEP_LOW",
        )

    return [
        item
        for item in book.items
        if not any(
            key in item[
                "name"
            ]
            for key in forbidden
        )
    ]


def candidate_regimes(
    book,
    side,
):
    atoms = side_regimes(
        book,
        side,
    )

    out = [
        (
            item[
                "name"
            ],
            1,
            item[
                "train"
            ],
            item[
                "val"
            ],
            item[
                "test"
            ],
        )
        for item in atoms
    ]

    for a, b in combinations(
        atoms,
        2,
    ):
        if (
            a[
                "dimension"
            ]
            == b[
                "dimension"
            ]
        ):
            continue

        out.append(
            (
                a[
                    "name"
                ]
                + " & "
                + b[
                    "name"
                ],

                2,

                a[
                    "train"
                ]
                & b[
                    "train"
                ],

                a[
                    "val"
                ]
                & b[
                    "val"
                ],

                a[
                    "test"
                ]
                & b[
                    "test"
                ],
            )
        )

    return out


def main():
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
        "TEN V6.5.5 "
        "RAW MARKET REGIME MINER"
    )

    print("=" * 124)

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
    ) = v.load_data()

    model = v.TechnicalMoE(
        groups
    ).to(
        device
    )

    checkpoint = torch.load(
        CKPT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    print(
        "Champion epoch:",
        checkpoint[
            "epoch"
        ],
    )

    def predict_rows(rows):
        dl = v.loader(
            rows,
            False,
            arrays,
            mean,
            std,
        )

        pred = v.predict(
            model,
            dl,
            device,
        )

        return (
            pred,
            v.scores(
                pred
            ),
        )

    print(
        "Predicting TRAIN ..."
    )

    train_pred, train_score = (
        predict_rows(
            split[
                "train"
            ]
        )
    )

    print(
        "Predicting VAL ..."
    )

    val_pred, val_score = (
        predict_rows(
            split[
                "val"
            ]
        )
    )

    print(
        "Predicting 2025 ..."
    )

    test_pred, test_score = (
        predict_rows(
            split[
                "test2025"
            ]
        )
    )

    print(
        "Extracting raw market context ..."
    )

    features = extract_features(
        arrays,
        split,
        names,
    )

    print(
        "Raw context features:",
        len(
            features[
                "names"
            ]
        ),
    )

    book, regime_thresholds = (
        build_regimes(
            features
        )
    )

    print(
        "Atomic regimes:",
        len(
            book.items
        ),
    )

    with open(
        OUT
        / "regime_thresholds_v655.json",
        "w",
    ) as f:
        json.dump(
            regime_thresholds,
            f,
            indent=2,
        )

    target = pd.read_parquet(
        v.TARGET_FILE,
        columns=[
            "year",
        ],
    )

    years = {
        "train":
            target[
                "year"
            ].to_numpy(
                np.int16
            )[
                split[
                    "train"
                ]
            ],

        "val":
            target[
                "year"
            ].to_numpy(
                np.int16
            )[
                split[
                    "val"
                ]
            ],

        "test":
            target[
                "year"
            ].to_numpy(
                np.int16
            )[
                split[
                    "test2025"
                ]
            ],
    }

    all_rows = []
    score_thresholds = {}

    for policy_name in POLICIES:
        score_thresholds[
            policy_name
        ] = {}

        for side, side_name in (
            (
                0,
                "LONG",
            ),
            (
                1,
                "SHORT",
            ),
        ):
            regimes = candidate_regimes(
                book,
                side,
            )

            print()
            print(
                policy_name.upper(),
                side_name,
                "regime candidates:",
                len(
                    regimes
                ),
            )

            train_side = train_score[
                policy_name
            ][
                :,
                side
            ]

            val_side = val_score[
                policy_name
            ][
                :,
                side
            ]

            test_side = test_score[
                policy_name
            ][
                :,
                side
            ]

            score_thresholds[
                policy_name
            ][
                side_name
            ] = {}

            for q in SCORE_QS:
                threshold = float(
                    np.quantile(
                        train_side,
                        q,
                    )
                )

                q_name = (
                    f"Q{q * 100:g}"
                )

                score_thresholds[
                    policy_name
                ][
                    side_name
                ][
                    q_name
                ] = threshold

                train_tail = (
                    train_side
                    >= threshold
                )

                val_tail = (
                    val_side
                    >= threshold
                )

                test_tail = (
                    test_side
                    >= threshold
                )

                baseline_train = metrics(
                    train_tail,
                    train_pred[
                        "pnl"
                    ][
                        :,
                        side
                    ],
                    train_pred[
                        "race_true"
                    ][
                        :,
                        side
                    ],
                )

                baseline_val = metrics(
                    val_tail,
                    val_pred[
                        "pnl"
                    ][
                        :,
                        side
                    ],
                    val_pred[
                        "race_true"
                    ][
                        :,
                        side
                    ],
                )

                baseline_test = metrics(
                    test_tail,
                    test_pred[
                        "pnl"
                    ][
                        :,
                        side
                    ],
                    test_pred[
                        "race_true"
                    ][
                        :,
                        side
                    ],
                )

                for (
                    regime_name,
                    order,
                    regime_train,
                    regime_val,
                    regime_test,
                ) in regimes:

                    train_mask = (
                        train_tail
                        & regime_train
                    )

                    val_mask = (
                        val_tail
                        & regime_val
                    )

                    test_mask = (
                        test_tail
                        & regime_test
                    )

                    train_m = metrics(
                        train_mask,
                        train_pred[
                            "pnl"
                        ][
                            :,
                            side
                        ],
                        train_pred[
                            "race_true"
                        ][
                            :,
                            side
                        ],
                    )

                    val_m = metrics(
                        val_mask,
                        val_pred[
                            "pnl"
                        ][
                            :,
                            side
                        ],
                        val_pred[
                            "race_true"
                        ][
                            :,
                            side
                        ],
                    )

                    test_m = metrics(
                        test_mask,
                        test_pred[
                            "pnl"
                        ][
                            :,
                            side
                        ],
                        test_pred[
                            "race_true"
                        ][
                            :,
                            side
                        ],
                    )

                    if (
                        train_m[
                            "n"
                        ]
                        < MIN_TRAIN
                        or val_m[
                            "n"
                        ]
                        < MIN_VAL
                    ):
                        qualified = False
                    else:
                        qualified = True

                    train_lift = (
                        train_m[
                            "mean"
                        ]
                        - baseline_train[
                            "mean"
                        ]
                    )

                    val_lift = (
                        val_m[
                            "mean"
                        ]
                        - baseline_val[
                            "mean"
                        ]
                    )

                    test_lift = (
                        test_m[
                            "mean"
                        ]
                        - baseline_test[
                            "mean"
                        ]
                    )

                    stable_mean = min(
                        train_m[
                            "mean"
                        ],
                        val_m[
                            "mean"
                        ],
                    )

                    stable_lift = min(
                        train_lift,
                        val_lift,
                    )

                    gap = abs(
                        train_m[
                            "mean"
                        ]
                        - val_m[
                            "mean"
                        ]
                    )

                    stable_score = (
                        stable_mean
                        + 0.50
                        * stable_lift
                        - 0.15
                        * gap
                    )

                    strict = (
                        qualified
                        and train_m[
                            "mean"
                        ] > 0
                        and val_m[
                            "mean"
                        ] > 0
                        and train_m[
                            "pf"
                        ] > 1.0
                        and val_m[
                            "pf"
                        ] > 1.0
                        and train_m[
                            "tp_res"
                        ] > BE_TP_RES
                        and val_m[
                            "tp_res"
                        ] > BE_TP_RES
                    )

                    row = {
                        "policy":
                            policy_name,

                        "side":
                            side_name,

                        "score_q":
                            q,

                        "score_threshold":
                            threshold,

                        "regime_order":
                            order,

                        "regime":
                            regime_name,

                        "qualified":
                            qualified,

                        "strict_train_val":
                            strict,

                        "stable_score":
                            stable_score,

                        "stable_mean":
                            stable_mean,

                        "stable_lift":
                            stable_lift,

                        "train_val_gap":
                            gap,
                    }

                    for prefix, result in (
                        (
                            "train",
                            train_m,
                        ),
                        (
                            "val",
                            val_m,
                        ),
                        (
                            "test2025",
                            test_m,
                        ),
                    ):
                        for key, value in result.items():
                            row[
                                f"{prefix}_{key}"
                            ] = value

                    row[
                        "train_lift"
                    ] = train_lift

                    row[
                        "val_lift"
                    ] = val_lift

                    row[
                        "test2025_lift"
                    ] = test_lift

                    row[
                        "train_baseline_mean"
                    ] = baseline_train[
                        "mean"
                    ]

                    row[
                        "val_baseline_mean"
                    ] = baseline_val[
                        "mean"
                    ]

                    row[
                        "test2025_baseline_mean"
                    ] = baseline_test[
                        "mean"
                    ]

                    all_rows.append(
                        row
                    )

    with open(
        OUT
        / "score_thresholds_v655.json",
        "w",
    ) as f:
        json.dump(
            score_thresholds,
            f,
            indent=2,
        )

    result = pd.DataFrame(
        all_rows
    )

    result.to_csv(
        OUT
        / "all_raw_regime_candidates_v655.csv",
        index=False,
    )

    qualified = (
        result[
            result[
                "qualified"
            ]
        ]
        .sort_values(
            [
                "stable_score",
                "stable_mean",
                "stable_lift",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    qualified.to_csv(
        OUT
        / "qualified_raw_regimes_v655.csv",
        index=False,
    )

    strict = (
        qualified[
            qualified[
                "strict_train_val"
            ]
        ]
        .copy()
    )

    strict.to_csv(
        OUT
        / "strict_train_val_survivors_v655.csv",
        index=False,
    )

    print()
    print("=" * 124)

    print(
        "TOTAL CANDIDATES:",
        f"{len(result):,}",
    )

    print(
        "QUALIFIED:",
        f"{len(qualified):,}",
    )

    print(
        "STRICT TRAIN+VAL SURVIVORS:",
        f"{len(strict):,}",
    )

    print(
        "BREAK-EVEN TP|RES:",
        f"{BE_TP_RES:.2%}",
    )

    print()
    print(
        "TOP 30 STABLE RAW REGIMES"
    )

    print("-" * 124)

    cols = [
        "policy",
        "side",
        "score_q",
        "regime",
        "train_n",
        "train_mean",
        "train_pf",
        "train_tp_res",
        "val_n",
        "val_mean",
        "val_pf",
        "val_tp_res",
        "test2025_n",
        "test2025_mean",
        "test2025_pf",
        "test2025_tp_res",
        "train_lift",
        "val_lift",
        "test2025_lift",
    ]

    with pd.option_context(
        "display.max_colwidth",
        90,
        "display.width",
        240,
        "display.max_columns",
        None,
    ):
        print(
            qualified[
                cols
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    if len(strict):
        print()
        print(
            "STRICT TRAIN+VAL SURVIVORS"
        )

        print("-" * 124)

        with pd.option_context(
            "display.max_colwidth",
            90,
            "display.width",
            240,
            "display.max_columns",
            None,
        ):
            print(
                strict[
                    cols
                ]
                .head(
                    30
                )
                .to_string(
                    index=False
                )
            )

    else:
        print()
        print(
            "NO STRICT TRAIN+VAL "
            "POSITIVE REGIME SURVIVED."
        )

    print()
    print(
        "Saved:",
        OUT,
    )

    print(
        "2025 was NOT used "
        "for ranking or selection."
    )

    print(
        "2026 RESERVED: "
        "NOT EVALUATED."
    )


if __name__ == "__main__":
    main()
