#!/usr/bin/env python3

def test_examples():
    # ensure that examples run without error

    # turn off plotting
    import matplotlib
    matplotlib.use('Agg')

    exec(open('examples/single_vantage.py').read())
    exec(open('examples/static_retrieval.py').read())