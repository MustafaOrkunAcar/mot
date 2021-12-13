import argparse


def get_argparser():
    parser = argparse.ArgumentParser(prog='QM9 IPU')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=1e-3,
                        required=False,
                        type=float,
                        help='Learning rate for network parameter updates.')
    parser.add_argument('-n',
                        '--num-ipus',
                        default=1,
                        required=False,
                        type=int,
                        help='Number of IPUs for data parallelism.')
    parser.add_argument('-b',
                        '--batch-size',
                        default=2,
                        required=False,
                        type=int,
                        help='Batch size.')
    parser.add_argument('-e',
                        '--epochs',
                        default=100,
                        required=False,
                        type=int,
                        help='How many epochs to train for.')
    parser.add_argument('-a',
                        '--amount',
                        default=1024,
                        required=False,
                        type=int,
                        help='How many graphs to load.')
    parser.add_argument('-p',
                        '--profile',
                        action='store_true',
                        required=False,
                        help='Whether to profile code. NOTE: number of epochs '
                             'and steps per epoch set to small values if this flag given.')
    parser.add_argument('-d',
                        '--profile-dir',
                        default='./qm9_profile',
                        type=str,
                        required=False,
                        help='Where to store profile data.')
    parser.add_argument('-m',
                        '--model',
                        default='ECC',
                        type=str,
                        required=False,
                        help='Layer')
    return parser
