from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression

import training.v6.models.train_multisurface_technical_brain_v661 as brain


CKPT = Path(
    "training/artifacts/v6/"
    "multisurface_technical_brain_v661/"
    "best_multisurface_technical_brain_v661.pt"
)

M5_FILE = Path(
    "training/v2/data_lake/xau/"
    "xauusd_m5_bid_ask_2016_2026-06.parquet"
)

OUT = Path(
    "training/artifacts/v6/"
    "surface_execution_policy_v662"
)

STEP_NS = 300_000_000_000

ENTRY_QS = (
    0.990,
    0.995,
    0.998,
)

MIN_AGREEMENTS = (
    3,
    4,
    5,
)

PERSISTENCES = (
    1,
    2,
)

RESET_Q = 0.95
COOLDOWN_BARS = 1

MIN_SELECTION_TRADES = 30

# Spread is already embedded
# in executable BID/ASK prices.
#
# This is EXTRA round-trip friction
# used for policy selection.
SELECTION_EXTRA_COST_BPS = 0.50

STRESS_COSTS = (
    0.0,
    0.5,
    1.0,
    2.0,
)


TRADEABLE = (
    (30, 30, 15),
    (60, 30, 15),
    (60, 40, 20),
    (120, 30, 15),
    (120, 40, 20),
    (120, 60, 30),
)


def sigmoid(x):
    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -x
            )
        )
    )


def logit(x):
    x = np.clip(
        x,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        x
        / (
            1.0 - x
        )
    )


def race_log_odds(
    p_tp,
    p_sl,
):
    return np.log(
        np.maximum(
            p_tp,
            1e-8,
        )
        / np.maximum(
            p_sl,
            1e-8,
        )
    )


def fit_binary_calibrator(
    x,
    y,
):
    x = np.asarray(
        x,
        np.float64,
    )

    y = np.asarray(
        y,
        np.int8,
    )

    if (
        len(y) < 100
        or len(
            np.unique(
                y
            )
        ) < 2
    ):
        return {
            "coef": 1.0,
            "intercept": 0.0,
            "fallback": True,
        }

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=500,
    )

    model.fit(
        x.reshape(
            -1,
            1,
        ),
        y,
    )

    return {
        "coef":
            float(
                model.coef_[
                    0,
                    0
                ]
            ),

        "intercept":
            float(
                model.intercept_[
                    0
                ]
            ),

        "fallback":
            False,
    }


def apply_calibrator(
    x,
    model,
):
    return sigmoid(
        model[
            "coef"
        ]
        * x
        + model[
            "intercept"
        ]
    )


def fit_calibrators(
    pred,
):
    prob = pred[
        "race_prob"
    ]

    true = pred[
        "race_true"
    ]

    models = []

    for task_id in range(
        brain.N_TASKS
    ):
        p = prob[
            :,
            task_id
        ]

        y = true[
            :,
            task_id
        ]

        valid = (
            y >= 0
        )

        event_target = (
            y[
                valid
            ]
            != 2
        ).astype(
            np.int8
        )

        raw_event = logit(
            p[
                valid,
                0
            ]
            + p[
                valid,
                1
            ]
        )

        event_model = (
            fit_binary_calibrator(
                raw_event,
                event_target,
            )
        )

        resolved = (
            (y == 0)
            | (
                y == 1
            )
        )

        race_target = (
            y[
                resolved
            ]
            == 0
        ).astype(
            np.int8
        )

        raw_race = race_log_odds(
            p[
                resolved,
                0
            ],
            p[
                resolved,
                1
            ],
        )

        race_model = (
            fit_binary_calibrator(
                raw_race,
                race_target,
            )
        )

        models.append(
            {
                "event":
                    event_model,

                "race":
                    race_model,
            }
        )

    return models


def calibrate(
    pred,
    models,
):
    raw = pred[
        "race_prob"
    ]

    out = np.zeros_like(
        raw,
        dtype=np.float64,
    )

    for task_id in range(
        brain.N_TASKS
    ):
        p = raw[
            :,
            task_id
        ]

        raw_event = logit(
            p[
                :,
                0
            ]
            + p[
                :,
                1
            ]
        )

        event = apply_calibrator(
            raw_event,
            models[
                task_id
            ][
                "event"
            ],
        )

        raw_race = race_log_odds(
            p[
                :,
                0
            ],
            p[
                :,
                1
            ],
        )

        q_tp = apply_calibrator(
            raw_race,
            models[
                task_id
            ][
                "race"
            ],
        )

        out[
            :,
            task_id,
            0
        ] = (
            event
            * q_tp
        )

        out[
            :,
            task_id,
            1
        ] = (
            event
            * (
                1.0
                - q_tp
            )
        )

        out[
            :,
            task_id,
            2
        ] = (
            1.0
            - event
        )

    return out


def task_lookup():
    lookup = {}

    for task_id, meta in enumerate(
        brain.TASKS
    ):
        lookup[
            (
                meta[
                    "side_id"
                ],
                meta[
                    "horizon"
                ],
                meta[
                    "tp"
                ],
                meta[
                    "sl"
                ],
            )
        ] = task_id

    return lookup


TASK_LOOKUP = task_lookup()


def surface_state(
    calibrated,
):
    n = calibrated.shape[
        0
    ]

    task_score = np.zeros(
        (
            n,
            brain.N_TASKS,
        ),
        dtype=np.float64,
    )

    for task_id, meta in enumerate(
        brain.TASKS
    ):
        p_tp = calibrated[
            :,
            task_id,
            0
        ]

        p_sl = calibrated[
            :,
            task_id,
            1
        ]

        resolved = (
            p_tp
            + p_sl
        )

        q_tp = (
            p_tp
            / np.maximum(
                resolved,
                1e-12,
            )
        )

        tp = float(
            meta[
                "tp"
            ]
        )

        sl = float(
            meta[
                "sl"
            ]
        )

        # Conditional expected R
        # among TP/SL resolved events.
        edge_r = (
            q_tp
            * tp
            - (
                1.0
                - q_tp
            )
            * sl
        ) / sl

        # Downweight a task when
        # model expects mostly timeout.
        task_score[
            :,
            task_id
        ] = (
            np.sqrt(
                resolved
            )
            * edge_r
        )

    side_score = np.zeros(
        (
            n,
            2,
        ),
        dtype=np.float64,
    )

    agreement = np.zeros(
        (
            n,
            2,
        ),
        dtype=np.int16,
    )

    best_task = np.full(
        (
            n,
            2,
        ),
        -1,
        dtype=np.int16,
    )

    best_task_score = np.full(
        (
            n,
            2,
        ),
        -np.inf,
        dtype=np.float64,
    )

    for side_id in range(
        2
    ):
        all_ids = [
            task_id
            for task_id, meta
            in enumerate(
                brain.TASKS
            )
            if meta[
                "side_id"
            ]
            == side_id
        ]

        x = task_score[
            :,
            all_ids
        ]

        sorted_x = np.sort(
            x,
            axis=1,
        )

        top3 = sorted_x[
            :,
            -3:
        ].mean(
            axis=1
        )

        median = np.median(
            x,
            axis=1,
        )

        # Robust consensus:
        # strong top tasks matter,
        # but broad disagreement
        # is penalized by median.
        side_score[
            :,
            side_id
        ] = (
            0.65
            * top3
            + 0.35
            * median
        )

        agreement[
            :,
            side_id
        ] = (
            x > 0
        ).sum(
            axis=1
        )

        trade_ids = [
            TASK_LOOKUP[
                (
                    side_id,
                    horizon,
                    tp,
                    sl,
                )
            ]
            for (
                horizon,
                tp,
                sl,
            ) in TRADEABLE
        ]

        tx = task_score[
            :,
            trade_ids
        ]

        local = np.argmax(
            tx,
            axis=1,
        )

        trade_ids_np = np.asarray(
            trade_ids,
            dtype=np.int16,
        )

        best_task[
            :,
            side_id
        ] = trade_ids_np[
            local
        ]

        best_task_score[
            :,
            side_id
        ] = tx[
            np.arange(
                n
            ),
            local,
        ]

    return {
        "task_score":
            task_score,

        "side_score":
            side_score,

        "agreement":
            agreement,

        "best_task":
            best_task,

        "best_task_score":
            best_task_score,
    }


def build_signals(
    source,
    timestamps_ns,
    state,
    thresholds,
    min_agreement,
    persistence,
):
    n = len(
        source
    )

    raw = np.full(
        n,
        -1,
        dtype=np.int8,
    )

    score = state[
        "side_score"
    ]

    agree = state[
        "agreement"
    ]

    best_score = state[
        "best_task_score"
    ]

    for i in range(
        n
    ):
        candidates = []

        for side in range(
            2
        ):
            if (
                score[
                    i,
                    side
                ]
                >= thresholds[
                    side
                ]
                and agree[
                    i,
                    side
                ]
                >= min_agreement
                and best_score[
                    i,
                    side
                ]
                > 0
            ):
                candidates.append(
                    side
                )

        if not candidates:
            continue

        if len(
            candidates
        ) == 1:
            raw[i] = candidates[
                0
            ]

        else:
            raw[i] = int(
                candidates[
                    np.argmax(
                        [
                            score[
                                i,
                                side
                            ]
                            for side
                            in candidates
                        ]
                    )
                ]
            )

    if persistence <= 1:
        return raw

    signal = np.full_like(
        raw,
        -1,
    )

    for i in range(
        persistence - 1,
        n,
    ):
        side = raw[i]

        if side < 0:
            continue

        good = True

        for lag in range(
            1,
            persistence
        ):
            j = (
                i - lag
            )

            if raw[j] != side:
                good = False
                break

            if (
                source[
                    j + 1
                ]
                != source[
                    j
                ]
                + 1
            ):
                good = False
                break

            if (
                timestamps_ns[
                    j + 1
                ]
                - timestamps_ns[
                    j
                ]
                != STEP_NS
            ):
                good = False
                break

        if good:
            signal[i] = side

    return signal


def execute_trade(
    m5,
    timestamps_ns,
    signal_source,
    side_id,
    task_id,
):
    meta = brain.TASKS[
        task_id
    ]

    horizon = int(
        meta[
            "horizon"
        ]
    )

    tp_bps = float(
        meta[
            "tp"
        ]
    )

    sl_bps = float(
        meta[
            "sl"
        ]
    )

    steps = (
        horizon
        // 5
    )

    entry_row = (
        signal_source
        + 1
    )

    end_row = (
        signal_source
        + steps
    )

    n_rows = len(
        timestamps_ns
    )

    if (
        entry_row >= n_rows
        or end_row >= n_rows
    ):
        return None

    check = timestamps_ns[
        signal_source:
        end_row + 1
    ]

    if (
        len(check)
        != steps + 1
        or not np.all(
            np.diff(
                check
            )
            == STEP_NS
        )
    ):
        return None

    if side_id == 0:
        entry = float(
            m5[
                "ask_open"
            ][
                entry_row
            ]
        )

        tp_price = (
            entry
            * (
                1.0
                + tp_bps
                / 10000.0
            )
        )

        sl_price = (
            entry
            * (
                1.0
                - sl_bps
                / 10000.0
            )
        )

    else:
        entry = float(
            m5[
                "bid_open"
            ][
                entry_row
            ]
        )

        tp_price = (
            entry
            * (
                1.0
                - tp_bps
                / 10000.0
            )
        )

        sl_price = (
            entry
            * (
                1.0
                + sl_bps
                / 10000.0
            )
        )

    for row in range(
        entry_row,
        end_row + 1,
    ):
        if side_id == 0:
            open_px = float(
                m5[
                    "bid_open"
                ][row]
            )

            # Gap through stop.
            if open_px <= sl_price:
                gross = (
                    (
                        open_px
                        - entry
                    )
                    / entry
                    * 10000.0
                )

                return {
                    "outcome":
                        "SL_GAP",

                    "gross_bps":
                        gross,

                    "exit_row":
                        row,

                    "hold_bars":
                        row
                        - entry_row
                        + 1,

                    "entry_price":
                        entry,

                    "exit_price":
                        open_px,
                }

            # Favor no positive gap
            # improvement at TP:
            # fill at target price.
            if open_px >= tp_price:
                return {
                    "outcome":
                        "TP",

                    "gross_bps":
                        tp_bps,

                    "exit_row":
                        row,

                    "hold_bars":
                        row
                        - entry_row
                        + 1,

                    "entry_price":
                        entry,

                    "exit_price":
                        tp_price,
                }

            tp_hit = (
                float(
                    m5[
                        "bid_high"
                    ][row]
                )
                >= tp_price
            )

            sl_hit = (
                float(
                    m5[
                        "bid_low"
                    ][row]
                )
                <= sl_price
            )

        else:
            open_px = float(
                m5[
                    "ask_open"
                ][row]
            )

            if open_px >= sl_price:
                gross = (
                    (
                        entry
                        - open_px
                    )
                    / entry
                    * 10000.0
                )

                return {
                    "outcome":
                        "SL_GAP",

                    "gross_bps":
                        gross,

                    "exit_row":
                        row,

                    "hold_bars":
                        row
                        - entry_row
                        + 1,

                    "entry_price":
                        entry,

                    "exit_price":
                        open_px,
                }

            if open_px <= tp_price:
                return {
                    "outcome":
                        "TP",

                    "gross_bps":
                        tp_bps,

                    "exit_row":
                        row,

                    "hold_bars":
                        row
                        - entry_row
                        + 1,

                    "entry_price":
                        entry,

                    "exit_price":
                        tp_price,
                }

            tp_hit = (
                float(
                    m5[
                        "ask_low"
                    ][row]
                )
                <= tp_price
            )

            sl_hit = (
                float(
                    m5[
                        "ask_high"
                    ][row]
                )
                >= sl_price
            )

        # Conservative intrabar assumption:
        # if both touch in same M5 bar,
        # SL is assumed first.
        if tp_hit and sl_hit:
            return {
                "outcome":
                    "SL_AMBIGUOUS",

                "gross_bps":
                    -sl_bps,

                "exit_row":
                    row,

                "hold_bars":
                    row
                    - entry_row
                    + 1,

                "entry_price":
                    entry,

                "exit_price":
                    sl_price,
            }

        if sl_hit:
            return {
                "outcome":
                    "SL",

                "gross_bps":
                    -sl_bps,

                "exit_row":
                    row,

                "hold_bars":
                    row
                    - entry_row
                    + 1,

                "entry_price":
                    entry,

                "exit_price":
                    sl_price,
            }

        if tp_hit:
            return {
                "outcome":
                    "TP",

                "gross_bps":
                    tp_bps,

                "exit_row":
                    row,

                "hold_bars":
                    row
                    - entry_row
                    + 1,

                "entry_price":
                    entry,

                "exit_price":
                    tp_price,
            }

    if side_id == 0:
        exit_price = float(
            m5[
                "bid_close"
            ][
                end_row
            ]
        )

        gross = (
            (
                exit_price
                - entry
            )
            / entry
            * 10000.0
        )

    else:
        exit_price = float(
            m5[
                "ask_close"
            ][
                end_row
            ]
        )

        gross = (
            (
                entry
                - exit_price
            )
            / entry
            * 10000.0
        )

    return {
        "outcome":
            "TIMEOUT",

        "gross_bps":
            gross,

        "exit_row":
            end_row,

        "hold_bars":
            steps,

        "entry_price":
            entry,

        "exit_price":
            exit_price,
    }


def simulate(
    rows,
    source,
    signal,
    state,
    m5,
    timestamps_ns,
    reset_thresholds,
):
    trades = []

    occupied_until = -1

    blocked = [
        False,
        False,
    ]

    stats = {
        "persistent_signals":
            int(
                (
                    signal >= 0
                ).sum()
            ),

        "suppressed_open":
            0,

        "suppressed_rearm":
            0,

        "invalid_execution":
            0,
    }

    score = state[
        "side_score"
    ]

    for i in range(
        len(
            source
        )
    ):
        s = int(
            source[i]
        )

        # Hysteresis re-arm.
        #
        # After a trade exits,
        # same side must cool back
        # below a lower threshold
        # before a new entry is allowed.
        if s > occupied_until:
            for side in range(
                2
            ):
                if (
                    blocked[
                        side
                    ]
                    and score[
                        i,
                        side
                    ]
                    < reset_thresholds[
                        side
                    ]
                ):
                    blocked[
                        side
                    ] = False

        side = int(
            signal[i]
        )

        if side < 0:
            continue

        if s <= occupied_until:
            stats[
                "suppressed_open"
            ] += 1
            continue

        if blocked[
            side
        ]:
            stats[
                "suppressed_rearm"
            ] += 1
            continue

        task_id = int(
            state[
                "best_task"
            ][
                i,
                side
            ]
        )

        result = execute_trade(
            m5,
            timestamps_ns,
            s,
            side,
            task_id,
        )

        if result is None:
            stats[
                "invalid_execution"
            ] += 1
            continue

        meta = brain.TASKS[
            task_id
        ]

        entry_row = (
            s + 1
        )

        trade = {
            "signal_source":
                s,

            "signal_timestamp":
                m5[
                    "timestamp"
                ][s],

            "entry_row":
                entry_row,

            "entry_timestamp":
                m5[
                    "timestamp"
                ][
                    entry_row
                ],

            "exit_row":
                result[
                    "exit_row"
                ],

            "exit_timestamp":
                m5[
                    "timestamp"
                ][
                    result[
                        "exit_row"
                    ]
                ],

            "side":
                (
                    "LONG"
                    if side == 0
                    else "SHORT"
                ),

            "task_id":
                task_id,

            "horizon":
                meta[
                    "horizon"
                ],

            "tp_bps":
                meta[
                    "tp"
                ],

            "sl_bps":
                meta[
                    "sl"
                ],

            "surface_score":
                float(
                    score[
                        i,
                        side
                    ]
                ),

            "opposite_score":
                float(
                    score[
                        i,
                        1 - side
                    ]
                ),

            "agreement":
                int(
                    state[
                        "agreement"
                    ][
                        i,
                        side
                    ]
                ),

            "task_score":
                float(
                    state[
                        "best_task_score"
                    ][
                        i,
                        side
                    ]
                ),

            "outcome":
                result[
                    "outcome"
                ],

            "gross_bps":
                float(
                    result[
                        "gross_bps"
                    ]
                ),

            "hold_bars":
                int(
                    result[
                        "hold_bars"
                    ]
                ),

            "entry_price":
                float(
                    result[
                        "entry_price"
                    ]
                ),

            "exit_price":
                float(
                    result[
                        "exit_price"
                    ]
                ),
        }

        trades.append(
            trade
        )

        occupied_until = (
            int(
                result[
                    "exit_row"
                ]
            )
            + COOLDOWN_BARS
        )

        blocked[
            side
        ] = True

    return (
        pd.DataFrame(
            trades
        ),
        stats,
    )


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


def max_drawdown(
    pnl,
):
    pnl = np.asarray(
        pnl,
        np.float64,
    )

    if not len(
        pnl
    ):
        return 0.0

    equity = np.cumsum(
        pnl
    )

    peak = np.maximum.accumulate(
        np.r_[
            0.0,
            equity
        ]
    )[
        1:
    ]

    dd = (
        equity
        - peak
    )

    return float(
        -dd.min()
    )


def longest_loss_streak(
    pnl,
):
    best = 0
    current = 0

    for value in pnl:
        if value < 0:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 0

    return best


def trade_metrics(
    trades,
    extra_cost_bps=0.0,
):
    if len(
        trades
    ) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "sum": 0.0,
            "pf": np.nan,
            "win": np.nan,
            "max_dd": np.nan,
            "loss_streak": 0,
            "avg_hold": np.nan,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "timeout_rate": np.nan,
        }

    pnl = (
        trades[
            "gross_bps"
        ].to_numpy(
            np.float64
        )
        - extra_cost_bps
    )

    outcome = trades[
        "outcome"
    ].astype(
        str
    )

    tp = (
        outcome == "TP"
    )

    sl = (
        outcome.str.startswith(
            "SL"
        )
    )

    timeout = (
        outcome == "TIMEOUT"
    )

    return {
        "n":
            int(
                len(
                    pnl
                )
            ),

        "mean":
            float(
                pnl.mean()
            ),

        "median":
            float(
                np.median(
                    pnl
                )
            ),

        "sum":
            float(
                pnl.sum()
            ),

        "pf":
            profit_factor(
                pnl
            ),

        "win":
            float(
                (
                    pnl > 0
                ).mean()
            ),

        "max_dd":
            max_drawdown(
                pnl
            ),

        "loss_streak":
            longest_loss_streak(
                pnl
            ),

        "avg_hold":
            float(
                trades[
                    "hold_bars"
                ].mean()
            ),

        "tp_rate":
            float(
                tp.mean()
            ),

        "sl_rate":
            float(
                sl.mean()
            ),

        "timeout_rate":
            float(
                timeout.mean()
            ),
    }


def half_metrics(
    trades,
    extra_cost,
):
    if not len(
        trades
    ):
        return (
            trade_metrics(
                trades,
                extra_cost,
            ),
            trade_metrics(
                trades,
                extra_cost,
            ),
        )

    month = pd.to_datetime(
        trades[
            "entry_timestamp"
        ],
        utc=True,
    ).dt.month

    h1 = trades[
        month <= 6
    ]

    h2 = trades[
        month >= 7
    ]

    return (
        trade_metrics(
            h1,
            extra_cost,
        ),
        trade_metrics(
            h2,
            extra_cost,
        ),
    )


def selection_score(
    metrics,
    h1,
    h2,
):
    if (
        metrics[
            "n"
        ]
        < MIN_SELECTION_TRADES
        or h1[
            "n"
        ] < 10
        or h2[
            "n"
        ] < 10
    ):
        return -np.inf

    robust_mean = min(
        h1[
            "mean"
        ],
        h2[
            "mean"
        ],
    )

    robust_pf = min(
        h1[
            "pf"
        ],
        h2[
            "pf"
        ],
    )

    pf_term = math.log(
        np.clip(
            robust_pf,
            0.20,
            5.0,
        )
    )

    total_pf_term = math.log(
        np.clip(
            metrics[
                "pf"
            ],
            0.20,
            5.0,
        )
    )

    return float(
        robust_mean
        + 0.20
        * pf_term
        + 0.10
        * total_pf_term
        - 0.0015
        * metrics[
            "max_dd"
        ]
    )


def print_metrics(
    title,
    trades,
    stats,
):
    print()
    print(
        title
    )

    print(
        "=" * 126
    )

    print(
        "Persistent signals:",
        stats[
            "persistent_signals"
        ],
    )

    print(
        "Suppressed while OPEN:",
        stats[
            "suppressed_open"
        ],
    )

    print(
        "Suppressed waiting RE-ARM:",
        stats[
            "suppressed_rearm"
        ],
    )

    print(
        "Invalid execution:",
        stats[
            "invalid_execution"
        ],
    )

    for cost in STRESS_COSTS:
        m = trade_metrics(
            trades,
            cost,
        )

        print(
            f"EXTRA COST {cost:>4.1f}bps | "
            f"trades={m['n']:>4} "
            f"mean={m['mean']:>+7.3f} "
            f"net={m['sum']:>+9.2f} "
            f"PF={m['pf']:>6.3f} "
            f"WIN={m['win']:>6.2%} "
            f"DD={m['max_dd']:>7.2f} "
            f"loss_streak={m['loss_streak']:>3} "
            f"hold={m['avg_hold']:>5.2f} bars"
        )

    if len(
        trades
    ):
        print()
        print(
            "Outcome:",
            trades[
                "outcome"
            ]
            .value_counts()
            .to_dict()
        )

        print(
            "Sides:",
            trades[
                "side"
            ]
            .value_counts()
            .to_dict()
        )

        task_counts = (
            trades.groupby(
                [
                    "side",
                    "horizon",
                    "tp_bps",
                    "sl_bps",
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        print(
            "Dynamic task usage:"
        )

        print(
            task_counts.to_string()
        )


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
        "TEN V6.6.2 "
        "SURFACE CONSENSUS + "
        "STATEFUL EXECUTION ENGINE"
    )

    print(
        "=" * 126
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

    model = (
        brain.MultiSurfaceTechnicalBrain(
            groups
        )
        .to(
            device
        )
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
        "Frozen Technical Brain epoch:",
        checkpoint[
            "epoch"
        ],
    )

    target_meta = pd.read_parquet(
        brain.TARGET_FILE,
        columns=[
            "year",
            "source_row",
        ],
    )

    year = target_meta[
        "year"
    ].to_numpy(
        np.int16
    )

    val_rows = split[
        "val"
    ]

    rows2023 = val_rows[
        year[
            val_rows
        ]
        == 2023
    ]

    rows2024 = val_rows[
        year[
            val_rows
        ]
        == 2024
    ]

    rows2025 = split[
        "test2025"
    ]

    print(
        "2023 calibration:",
        f"{len(rows2023):,}",
    )

    print(
        "2024 policy selection:",
        f"{len(rows2024):,}",
    )

    print(
        "2025 OOT benchmark:",
        f"{len(rows2025):,}",
    )

    print(
        "2026 RESERVED:",
        f"{len(split['reserved2026']):,}",
        "(NOT EVALUATED)",
    )

    def predict_rows(
        rows,
    ):
        dl = brain.make_loader(
            rows,
            False,
            arrays,
            mean,
            std,
        )

        return brain.predict(
            model,
            dl,
            device,
        )

    print()
    print(
        "Predicting 2023 ..."
    )

    pred2023 = predict_rows(
        rows2023
    )

    print(
        "Predicting 2024 ..."
    )

    pred2024 = predict_rows(
        rows2024
    )

    print(
        "Predicting 2025 ..."
    )

    pred2025 = predict_rows(
        rows2025
    )

    print(
        "Fitting per-task calibration "
        "on 2023 only ..."
    )

    calibrators = fit_calibrators(
        pred2023
    )

    with open(
        OUT
        / "calibrators_v662.json",
        "w",
    ) as f:
        json.dump(
            calibrators,
            f,
            indent=2,
        )

    cal2023 = calibrate(
        pred2023,
        calibrators,
    )

    cal2024 = calibrate(
        pred2024,
        calibrators,
    )

    cal2025 = calibrate(
        pred2025,
        calibrators,
    )

    state2023 = surface_state(
        cal2023
    )

    state2024 = surface_state(
        cal2024
    )

    state2025 = surface_state(
        cal2025
    )

    print(
        "Loading executable M5 stream ..."
    )

    m5_df = pd.read_parquet(
        M5_FILE,
        columns=[
            "timestamp",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        ],
    ).reset_index(
        drop=True
    )

    m5_df[
        "timestamp"
    ] = pd.to_datetime(
        m5_df[
            "timestamp"
        ],
        utc=True,
    )

    m5 = {
        col:
            m5_df[
                col
            ].to_numpy()
        for col in m5_df.columns
    }

    timestamps_ns = (
        m5_df[
            "timestamp"
        ]
        .astype(
            "int64"
        )
        .to_numpy(
            np.int64
        )
    )

    source2023 = arrays[
        "source"
    ][
        rows2023
    ]

    source2024 = arrays[
        "source"
    ][
        rows2024
    ]

    source2025 = arrays[
        "source"
    ][
        rows2025
    ]

    ts2023 = timestamps_ns[
        source2023
    ]

    ts2024 = timestamps_ns[
        source2024
    ]

    ts2025 = timestamps_ns[
        source2025
    ]

    reset_thresholds = [
        float(
            np.quantile(
                state2023[
                    "side_score"
                ][
                    :,
                    side
                ],
                RESET_Q,
            )
        )
        for side in range(
            2
        )
    ]

    print()
    print(
        "Reset thresholds:",
        reset_thresholds,
    )

    grid_rows = []

    best = None
    best_score = -np.inf

    print()
    print(
        "Selecting policy on 2024 only ..."
    )

    for q in ENTRY_QS:
        thresholds = [
            float(
                np.quantile(
                    state2023[
                        "side_score"
                    ][
                        :,
                        side
                    ],
                    q,
                )
            )
            for side in range(
                2
            )
        ]

        for min_agreement in (
            MIN_AGREEMENTS
        ):
            for persistence in (
                PERSISTENCES
            ):
                signal = build_signals(
                    source2024,
                    ts2024,
                    state2024,
                    thresholds,
                    min_agreement,
                    persistence,
                )

                trades, stats = simulate(
                    rows2024,
                    source2024,
                    signal,
                    state2024,
                    m5,
                    timestamps_ns,
                    reset_thresholds,
                )

                m = trade_metrics(
                    trades,
                    SELECTION_EXTRA_COST_BPS,
                )

                h1, h2 = half_metrics(
                    trades,
                    SELECTION_EXTRA_COST_BPS,
                )

                score = selection_score(
                    m,
                    h1,
                    h2,
                )

                grid_rows.append(
                    {
                        "entry_q":
                            q,

                        "min_agreement":
                            min_agreement,

                        "persistence":
                            persistence,

                        "long_threshold":
                            thresholds[
                                0
                            ],

                        "short_threshold":
                            thresholds[
                                1
                            ],

                        "selection_score":
                            score,

                        "trades":
                            m[
                                "n"
                            ],

                        "mean_bps":
                            m[
                                "mean"
                            ],

                        "net_bps":
                            m[
                                "sum"
                            ],

                        "pf":
                            m[
                                "pf"
                            ],

                        "win":
                            m[
                                "win"
                            ],

                        "max_dd":
                            m[
                                "max_dd"
                            ],

                        "h1_n":
                            h1[
                                "n"
                            ],

                        "h1_mean":
                            h1[
                                "mean"
                            ],

                        "h1_pf":
                            h1[
                                "pf"
                            ],

                        "h2_n":
                            h2[
                                "n"
                            ],

                        "h2_mean":
                            h2[
                                "mean"
                            ],

                        "h2_pf":
                            h2[
                                "pf"
                            ],

                        "persistent_signals":
                            stats[
                                "persistent_signals"
                            ],

                        "suppressed_open":
                            stats[
                                "suppressed_open"
                            ],

                        "suppressed_rearm":
                            stats[
                                "suppressed_rearm"
                            ],
                    }
                )

                print(
                    f"Q={q:.3f} "
                    f"agree={min_agreement} "
                    f"persist={persistence} "
                    f"trades={m['n']:>4} "
                    f"mean={m['mean']:>+7.3f} "
                    f"PF={m['pf']:>6.3f} "
                    f"DD={m['max_dd']:>7.2f} "
                    f"score={score:>+8.4f}"
                )

                if score > best_score:
                    best_score = score

                    best = {
                        "entry_q":
                            q,

                        "min_agreement":
                            min_agreement,

                        "persistence":
                            persistence,

                        "thresholds":
                            thresholds,
                    }

    grid = pd.DataFrame(
        grid_rows
    ).sort_values(
        "selection_score",
        ascending=False,
    )

    grid.to_csv(
        OUT
        / "policy_grid_2024_v662.csv",
        index=False,
    )

    if (
        best is None
        or not np.isfinite(
            best_score
        )
    ):
        raise RuntimeError(
            "No eligible V6.6.2 policy "
            "survived minimum trade rules."
        )

    policy = {
        **best,

        "reset_q":
            RESET_Q,

        "reset_thresholds":
            reset_thresholds,

        "cooldown_bars":
            COOLDOWN_BARS,

        "selection_extra_cost_bps":
            SELECTION_EXTRA_COST_BPS,

        "selection_score":
            float(
                best_score
            ),

        "tradeable_tasks":
            [
                list(x)
                for x in TRADEABLE
            ],
    }

    with open(
        OUT
        / "frozen_policy_v662.json",
        "w",
    ) as f:
        json.dump(
            policy,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 126
    )

    print(
        "FROZEN V6.6.2 POLICY"
    )

    print(
        "=" * 126
    )

    print(
        json.dumps(
            policy,
            indent=2,
        )
    )

    signal2024 = build_signals(
        source2024,
        ts2024,
        state2024,
        best[
            "thresholds"
        ],
        best[
            "min_agreement"
        ],
        best[
            "persistence"
        ],
    )

    trades2024, stats2024 = (
        simulate(
            rows2024,
            source2024,
            signal2024,
            state2024,
            m5,
            timestamps_ns,
            reset_thresholds,
        )
    )

    print_metrics(
        "2024 POLICY-SELECTION BACKTEST",
        trades2024,
        stats2024,
    )

    trades2024.to_csv(
        OUT
        / "trades_2024_v662.csv",
        index=False,
    )

    print()
    print(
        "=" * 126
    )

    print(
        "2025 FROZEN OUT-OF-TIME "
        "EXECUTION BENCHMARK"
    )

    print(
        "=" * 126
    )

    signal2025 = build_signals(
        source2025,
        ts2025,
        state2025,
        best[
            "thresholds"
        ],
        best[
            "min_agreement"
        ],
        best[
            "persistence"
        ],
    )

    trades2025, stats2025 = (
        simulate(
            rows2025,
            source2025,
            signal2025,
            state2025,
            m5,
            timestamps_ns,
            reset_thresholds,
        )
    )

    print_metrics(
        "2025 V6.6.2 STATEFUL EXECUTION",
        trades2025,
        stats2025,
    )

    trades2025.to_csv(
        OUT
        / "trades_2025_v662.csv",
        index=False,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Spread is embedded through "
        "BID/ASK execution."
    )

    print(
        "Extra-cost rows represent "
        "additional round-trip "
        "commission/slippage stress."
    )

    print(
        "Entries use NEXT M5 OPEN, "
        "not the signal candle close."
    )

    print(
        "Only one position may exist "
        "at a time."
    )

    print(
        "Signals while OPEN are ignored."
    )

    print(
        "Same-side re-entry requires "
        "a reset below hysteresis threshold."
    )

    print(
        "Same-bar TP+SL is treated "
        "conservatively as SL."
    )

    print(
        "Gap-through-stop uses actual "
        "executable opening price."
    )

    print(
        "2025 was NOT used "
        "for V6.6.2 policy selection."
    )

    print(
        "2026 RESERVED: NOT EVALUATED."
    )

    print(
        "Saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
