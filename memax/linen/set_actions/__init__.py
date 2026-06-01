"""This module contains set-action-based (classical) recurrent layers.
Each RNN type gets its own file.
+ `memax.linen.set_actions.elman` provides a basic Elman RNN layer.
+ `memax.linen.set_actions.lstm` provides the Long Short-Term Memory layer.
+ `memax.linen.set_actions.gru` provides the Gated Recurrent Unit layer.
+ `memax.linen.set_actions.mgu` provides the Minimal Gated Unit layer.
+ `memax.linen.set_actions.indrnn` provides the Independently Recurrent Neural Network layer.
+ `memax.linen.set_actions.spherical` provides a recurrent formulation of the Rotational RNN.
"""

from memax.linen.set_actions.elman import Elman, ElmanSetAction
from memax.linen.set_actions.gru import GRU, GRUSetAction
from memax.linen.set_actions.indrnn import IndRNN, IndRNNSetAction
from memax.linen.set_actions.lstm import LSTM, LSTMSetAction
from memax.linen.set_actions.mgu import MGU, MGUSetAction
from memax.linen.set_actions.spherical import Spherical, SphericalSetAction

__all__ = [
    "ElmanSetAction",
    "Elman",
    "LSTMSetAction",
    "LSTM",
    "GRUSetAction",
    "GRU",
    "MGUSetAction",
    "MGU",
    "IndRNNSetAction",
    "IndRNN",
    "SphericalSetAction",
    "Spherical",
]
