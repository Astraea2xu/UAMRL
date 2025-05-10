import argparse
from datetime import datetime
import pprint
import argparse


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default="0")
    parser.add_argument('--compound_sequence_dim', type=int, default=65)
    parser.add_argument('--protein_sequence_dim', type=int, default=21)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--diff_weight', type=float, default=1.0)
    parser.add_argument('--sim_weight', type=float, default=1.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)