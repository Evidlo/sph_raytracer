#!/usr/bin/env python3

import os

def test_examples():
    # ensure that examples run without error

    # turn off plotting
    import matplotlib
    matplotlib.use('Agg')

    os.chdir('examples')

    exec(open('single_vantage.py').read())
    exec(open('static_retrieval.py').read())
    exec(open('dynamic_measurements.py').read())