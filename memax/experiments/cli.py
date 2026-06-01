"""Shared argparse flags for experiment scripts."""

import argparse


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--recurrent-size",
        "--recurrent_size",
        dest="recurrent_size",
        type=int,
        default=None,
        help="Recurrent hidden size",
    )
    parser.add_argument("--num-layers", type=int, default=None, help="Residual layers")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Log to Weights & Biases",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="memax-debug",
        help="W&B project name",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        dest="dataset_name",
        type=str,
        default="sequential_mnist",
        help=("Dataset name (see memax.experiments.datasets.DATASET_NAMES)"),
    )
    parser.add_argument(
        "--loss-function",
        "--loss_function",
        dest="loss_function",
        type=str,
        default=None,
        help="loss_classify_terminal_output or loss_regress_terminal_output",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model names (default: all)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training set size after loading",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Max optimizer steps per epoch",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU-friendly defaults (tiny model, few steps)",
    )
