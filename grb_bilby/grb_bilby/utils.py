import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt

import grb_bilby

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'paper.mplstyle')
plt.style.use(filename)

logger = logging.getLogger('grb_bilby')


def find_path(path):
    if path == 'default':
        return os.path.join(dirname, '../data/GRBData')
    else:
        return path


def setup_logger(outdir='.', label=None, log_level='INFO'):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ==========
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    """

    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    logger = logging.getLogger('grb_bilby')
    logger.propagate = False
    logger.setLevel(level)

    if not any([type(h) == logging.StreamHandler for h in logger.handlers]):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if not any([type(h) == logging.FileHandler for h in logger.handlers]):
        if label is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))

            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info(f'Running grb_bilby version: {grb_bilby.__version__}')


setup_logger(label='test')
