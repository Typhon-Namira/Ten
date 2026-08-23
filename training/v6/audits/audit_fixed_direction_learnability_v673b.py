from pathlib import Path
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
import training.v6.models.train_directional_utility_brain_v673 as v673


OUT = Path(
    "training/artifacts/v6/"
    "fixed_direction_learnability_v673b"
)

EPOCHS = 6
BATCH = 4096
LR = 5e-4
WEIGHT_DECAY = 1e-4

SEED = 20260823


def seed_all():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class DirectionProbe(
    nn.Module
):
    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                256,
            ),
            nn.GELU(),
            nn.LayerNorm(
                256
            ),
            nn.Dropout(
                0.10
            ),

            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.LayerNorm(
                128
            ),

            nn.Linear(
                128,
                9,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


def make_pair_targets(
    execution,
):
    n = execution[
        "gross"
    ].shape[0]

    y = np.zeros(
        (
            n,
            9,
        ),
        dtype=np.float32,
    )

    valid = np.zeros(
        (
            n,
            9,
        ),
        dtype=bool,
    )

    gap = np.full(
        (
            n,
            9,
        ),
        np.nan,
        dtype=np.float32,
    )

    for j in range(9):

        sj = j + 9

        long_net = (
            execution[
                "gross"
            ][
                :,
                j
            ]
            - 0.5
        )

        short_net = (
            execution[
                "gross"
            ][
                :,
                sj
            ]
            - 0.5
        )

        ok = (
            execution[
                "valid"
            ][
                :,
                j
            ]
            & execution[
                "valid"
            ][
                :,
                sj
            ]
            & np.isfinite(
                long_net
            )
            & np.isfinite(
                short_net
            )
        )

        g = (
            short_net
            - long_net
        )

        ok &= (
            np.abs(g)
            > 1e-6
        )

        valid[
            :,
            j
        ] = ok

        gap[
            :,
            j
        ] = g

        y[
            ok,
            j
        ] = (
            g[
                ok
            ]
            > 0
        ).astype(
            np.float32
        )

    return (
        y,
        valid,
        gap,
    )


@torch.no_grad()
def extract_representation(
    model,
    rows,
    arrays,
    mean,
    std,
    device,
    label,
):
    print(
        f"Extracting {label}:",
        f"{len(rows):,}",
    )

    model.eval()

    loader = brain.make_loader(
        rows,
        False,
        arrays,
        mean,
        std,
    )

    chunks = []

    count = 0

    for batch in loader:

        x = batch[
            0
        ].to(
            device,
            non_blocking=True,
        )

        d = model(x)

        e = d[
            "exec"
        ]

        market = e[
            "market_state"
        ].float()

        p05 = torch.sigmoid(
            e[
                "win05_logit"
            ].float()
        )

        net = e[
            "net05_norm"
        ].float()

        race = F.softmax(
            e[
                "race_logits"
            ].float(),
            dim=-1,
        ).flatten(
            start_dim=1
        )

        ordinal = torch.sigmoid(
            d[
                "ordinal_logits"
            ].float()
        )

        rank = torch.sigmoid(
            d[
                "rank_logit"
            ].float()
        ).unsqueeze(
            1
        )

        best_net = d[
            "best_net_norm"
        ].float().unsqueeze(
            1
        )

        count_score = torch.sigmoid(
            d[
                "count_norm"
            ].float()
        ).unsqueeze(
            1
        )

        old_side = torch.sigmoid(
            d[
                "side_logit"
            ].float()
        ).unsqueeze(
            1
        )

        z = torch.cat(
            [
                market,
                p05,
                net,
                race,
                ordinal,
                rank,
                best_net,
                count_score,
                old_side,
            ],
            dim=1,
        )

        chunks.append(
            z.cpu().numpy().astype(
                np.float32
            )
        )

        count += (
            x.shape[0]
        )

    if count != len(rows):
        raise RuntimeError(
            "Representation alignment failure."
        )

    result = np.concatenate(
        chunks,
        axis=0,
    )

    print(
        f"{label} representation:",
        result.shape,
    )

    return result


def normalize(
    train,
    *others,
):
    mu = train.mean(
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    sd = train.std(
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    sd = np.where(
        sd < 1e-5,
        1.0,
        sd,
    ).astype(
        np.float32
    )

    train_n = (
        (train - mu)
        / sd
    ).astype(
        np.float32
    )

    output = [
        train_n
    ]

    for x in others:
        output.append(
            (
                (x - mu)
                / sd
            ).astype(
                np.float32
            )
        )

    return (
        *output,
        mu,
        sd,
    )


def task_name(
    j,
):
    meta = brain.TASKS[j]

    return (
        f"H{meta['horizon']}_"
        f"TP{meta['tp']}_"
        f"SL{meta['sl']}"
    )


def evaluate_probe(
    model,
    x,
    y,
    valid,
    gap,
    device,
    label,
):
    model.eval()

    probs = []

    with torch.no_grad():

        for start in range(
            0,
            len(x),
            BATCH,
        ):

            xb = torch.from_numpy(
                x[
                    start:
                    start + BATCH
                ]
            ).to(
                device,
                non_blocking=True,
            )

            p = torch.sigmoid(
                model(xb)
            )

            probs.append(
                p.cpu().numpy().astype(
                    np.float32
                )
            )

    p = np.concatenate(
        probs,
        axis=0,
    )

    rows = []

    print()
    print(
        f"{label} FIXED-DIRECTION PROBE"
    )

    print(
        "-" * 125
    )

    for j in range(9):

        for min_gap in (
            0.0,
            3.0,
            10.0,
        ):

            mask = (
                valid[
                    :,
                    j
                ]
                & (
                    np.abs(
                        gap[
                            :,
                            j
                        ]
                    )
                    >= min_gap
                )
            )

            yy = y[
                mask,
                j
            ].astype(
                np.uint8
            )

            pp = p[
                mask,
                j
            ]

            if (
                len(yy) == 0
                or yy.min() == yy.max()
            ):
                auc = np.nan
                acc = np.nan

            else:
                auc = float(
                    roc_auc_score(
                        yy,
                        pp,
                    )
                )

                acc = float(
                    (
                        (
                            pp >= 0.5
                        )
                        == yy
                    ).mean()
                )

            rows.append(
                {
                    "task":
                        j,

                    "name":
                        task_name(j),

                    "gap":
                        min_gap,

                    "n":
                        int(
                            mask.sum()
                        ),

                    "acc":
                        acc,

                    "auc":
                        auc,
                }
            )

    df = pd.DataFrame(
        rows
    )

    wide = df[
        df[
            "gap"
        ]
        == 0.0
    ].copy()

    print(
        wide[
            [
                "name",
                "n",
                "acc",
                "auc",
            ]
        ].to_string(
            index=False,
            formatters={
                "acc":
                    lambda x:
                        f"{x:.2%}",

                "auc":
                    lambda x:
                        f"{x:.4f}",
            },
        )
    )

    print()
    print(
        "MEAN AUC BY HORIZON"
    )

    for h in (
        30,
        60,
        120,
    ):

        task_ids = [
            j
            for j in range(9)
            if brain.TASKS[j][
                "horizon"
            ] == h
        ]

        for min_gap in (
            0.0,
            3.0,
            10.0,
        ):

            sub = df[
                (
                    df[
                        "task"
                    ].isin(
                        task_ids
                    )
                )
                & (
                    df[
                        "gap"
                    ]
                    == min_gap
                )
            ]

            print(
                f"H{h:<3} "
                f"gap>={min_gap:>4.0f} "
                f"AUC={sub['auc'].mean():.4f}"
            )

    return (
        df,
        p,
    )


def main():
    started = time.time()

    seed_all()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TEN V6.7.3B "
        "FIXED DIRECTION LEARNABILITY PROBE"
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

    daily_targets = (
        v672.load_daily_targets()
    )

    y_all, valid_all, gap_all = (
        make_pair_targets(
            execution
        )
    )

    years = execution[
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
        "Train:",
        f"{len(train_rows):,}",
    )

    print(
        "2023:",
        f"{len(rows23):,}",
    )

    print(
        "2024:",
        f"{len(rows24):,}",
    )

    print(
        "2025 NOT USED."
    )

    print(
        "2026 NOT USED."
    )

    daily_model, c672 = (
        v673.build_v672(
            groups,
            device,
        )
    )

    print(
        "Frozen representation from "
        f"V6.7.2 epoch {c672['epoch']}"
    )

    # No gradient through V6.7.2.
    for p in daily_model.parameters():
        p.requires_grad = False

    x_train = extract_representation(
        daily_model,
        train_rows,
        arrays,
        mean,
        std,
        device,
        "TRAIN",
    )

    x23 = extract_representation(
        daily_model,
        rows23,
        arrays,
        mean,
        std,
        device,
        "2023",
    )

    x24 = extract_representation(
        daily_model,
        rows24,
        arrays,
        mean,
        std,
        device,
        "2024",
    )

    (
        x_train,
        x23,
        x24,
        feature_mu,
        feature_sd,
    ) = normalize(
        x_train,
        x23,
        x24,
    )

    train_source = arrays[
        "source"
    ][
        train_rows
    ]

    s23 = arrays[
        "source"
    ][
        rows23
    ]

    s24 = arrays[
        "source"
    ][
        rows24
    ]

    y_train = y_all[
        train_source
    ]

    valid_train = valid_all[
        train_source
    ]

    gap_train = gap_all[
        train_source
    ]

    y23 = y_all[
        s23
    ]

    valid23 = valid_all[
        s23
    ]

    gap23 = gap_all[
        s23
    ]

    y24 = y_all[
        s24
    ]

    valid24 = valid_all[
        s24
    ]

    gap24 = gap_all[
        s24
    ]

    print(
        "Probe input dim:",
        x_train.shape[
            1
        ],
    )

    model = DirectionProbe(
        x_train.shape[
            1
        ]
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        SEED
    )

    best_auc = -np.inf
    champion_state = None
    champion_epoch = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        order = rng.permutation(
            len(
                x_train
            )
        )

        total_loss = 0.0
        batches = 0

        for start in range(
            0,
            len(order),
            BATCH,
        ):

            idx = order[
                start:
                start + BATCH
            ]

            xb = torch.from_numpy(
                x_train[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            yb = torch.from_numpy(
                y_train[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            vb = torch.from_numpy(
                valid_train[
                    idx
                ]
            ).to(
                device,
                non_blocking=True,
            )

            logits = model(
                xb
            )

            if not vb.any():
                continue

            loss = (
                F.binary_cross_entropy_with_logits(
                    logits[
                        vb
                    ],
                    yb[
                        vb
                    ],
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += float(
                loss.detach()
            )

            batches += 1

        print()
        print(
            "=" * 130
        )

        print(
            f"EPOCH {epoch}/{EPOCHS} "
            f"TRAIN LOSS="
            f"{total_loss/max(batches,1):.6f}"
        )

        result23, _ = evaluate_probe(
            model,
            x23,
            y23,
            valid23,
            gap23,
            device,
            "2023",
        )

        # Selection only from 2023.
        h120 = result23[
            (
                result23[
                    "task"
                ].isin(
                    [
                        6,
                        7,
                        8,
                    ]
                )
            )
            & (
                result23[
                    "gap"
                ]
                == 3.0
            )
        ]

        score = float(
            h120[
                "auc"
            ].mean()
        )

        print(
            "2023 H120 gap>=3 "
            "selection AUC:",
            f"{score:.6f}",
        )

        if score > best_auc:

            best_auc = score
            champion_epoch = epoch

            champion_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model.state_dict().items()
            }

            print(
                "*** NEW PROBE CHAMPION ***"
            )

    if champion_state is None:
        raise RuntimeError(
            "No probe champion."
        )

    model.load_state_dict(
        champion_state
    )

    print()
    print(
        "=" * 130
    )

    print(
        "FROZEN PROBE CHAMPION"
    )

    print(
        "=" * 130
    )

    print(
        "Epoch:",
        champion_epoch,
    )

    print(
        "2023 H120 selection:",
        f"{best_auc:.6f}",
    )

    result23, _ = evaluate_probe(
        model,
        x23,
        y23,
        valid23,
        gap23,
        device,
        "FINAL 2023",
    )

    print()
    print(
        "=" * 130
    )

    print(
        "OPENING 2024 AFTER "
        "PROBE FREEZE"
    )

    print(
        "=" * 130
    )

    result24, _ = evaluate_probe(
        model,
        x24,
        y24,
        valid24,
        gap24,
        device,
        "2024 FROZEN",
    )

    result23.to_csv(
        OUT
        / "probe_2023.csv",
        index=False,
    )

    result24.to_csv(
        OUT
        / "probe_2024.csv",
        index=False,
    )

    torch.save(
        {
            "epoch":
                champion_epoch,

            "selection_auc":
                best_auc,

            "model":
                champion_state,

            "feature_mu":
                feature_mu,

            "feature_sd":
                feature_sd,
        },
        OUT
        / "probe_champion.pt",
    )

    print()
    print(
        "=" * 130
    )

    print(
        "DECISION RULE"
    )

    print(
        "=" * 130
    )

    print(
        "If H120 direction AUC is "
        "materially above 0.5 on both "
        "2023 and 2024, build V6.7.4 "
        "horizon-specific Direction Brain."
    )

    print(
        "If H30/H60/H120 all remain "
        "near 0.5, stop adding heads: "
        "the causal context/representation "
        "is insufficient for direction."
    )

    print(
        "2025: untouched."
    )

    print(
        "2026: untouched."
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
