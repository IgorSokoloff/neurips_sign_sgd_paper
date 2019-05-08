import numpy as np
import time
import sys
import os
import argparse

from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file
from numpy.linalg import norm

from logreg_functions import logreg_loss, logreg_sgrad, sample_logreg_sgrad
