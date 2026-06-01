"""Hugging Face Hub dataset identifiers used by memax loaders.

Canonical dataset pages:
- https://huggingface.co/datasets/ylecun/mnist
- https://huggingface.co/datasets/bolt-lab/mnist-math-{seq_len}
- https://huggingface.co/datasets/bolt-lab/continuous-localization-{seq_len}
"""

# Raw MNIST images (README: ylecun/mnist). The legacy id "mnist" redirects but logs hub warnings.
MNIST = "ylecun/mnist"

# Published under bolt-lab; 100_000 lives only under smorad.
MNIST_MATH_ORG = "bolt-lab"
MNIST_MATH_ORG_ALT = "smorad"
MNIST_MATH_SEQ_LENS = [100, 1_000, 10_000, 100_000, 1_000_000]

CONTINUOUS_LOCALIZATION_ORG = "bolt-lab"
# Datasets available on the Hub as of upload (longer seq_lens may be added separately).
CONTINUOUS_LOCALIZATION_SEQ_LENS = [20, 100]


def mnist_math_hub_id(seq_len: int) -> str:
    """Return the Hub repo id for a MNIST Math sequence length."""
    if seq_len not in MNIST_MATH_SEQ_LENS:
        raise ValueError(
            f"Invalid seq_len {seq_len}, must be one of {MNIST_MATH_SEQ_LENS}"
        )
    if seq_len == 100_000:
        return f"{MNIST_MATH_ORG_ALT}/mnist-math-{seq_len}"
    return f"{MNIST_MATH_ORG}/mnist-math-{seq_len}"


def mnist_math_hub_url(seq_len: int) -> str:
    return f"https://huggingface.co/datasets/{mnist_math_hub_id(seq_len)}"


def continuous_localization_hub_id(sequence_length: int) -> str:
    if sequence_length not in CONTINUOUS_LOCALIZATION_SEQ_LENS:
        raise ValueError(
            f"Invalid sequence_length {sequence_length}, "
            f"must be one of {CONTINUOUS_LOCALIZATION_SEQ_LENS}"
        )
    return f"{CONTINUOUS_LOCALIZATION_ORG}/continuous-localization-{sequence_length}"


def continuous_localization_hub_url(sequence_length: int) -> str:
    return f"https://huggingface.co/datasets/{continuous_localization_hub_id(sequence_length)}"
