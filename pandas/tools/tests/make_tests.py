"""
Run this file to create timeseries_info_tests.npz needed for
testing the plotting ticks spacing.
"""

import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tsplt
import numpy as np

np.random.seed(12345)
X = np.cumprod(1 + np.random.normal(0, 1, 500)/100)
d1 = ts.Date('A', '1975-01-01')

def get_info(tseries, vminmax=None):
    freq = tseries.freq
    finder = tsplt.get_finder(freq)
    if not vminmax:
        vmin, vmax = tseries.dates[[0,-1]]
    else:
        vmin, vmax = vminmax
    return finder(vmin, vmax, freq)

def make_tseries(*args):
    nobs, freq = args
    return ts.time_series(X[:nobs], start_date=d1.asfreq(freq))

cases = dict(
    # annual
# annual
        ts_a = (25, 'A'),
# quarterly
        # case1 nobs <= 3.5 * 4
        ts_q1 = (10, 'Q'),
        #case2 nobs <= 11 * 4
        ts_q2 = (35, 'Q'),
        #case3 nobs > 11 * 4
        ts_q3 = (50, 'Q'),
# monthly
        # case 1 nobs <= 1.15 * 12
        ts_m1 = (10, 'M'),
        # case 2 nobs <= 2.5 * 12
        ts_m2 = (20, 'M'),
        # case 3 nobs <= 4 * 12
        ts_m3 = (40, 'M'),
        # case 4 <= 11 * 12
        ts_m4 = (100, 'M'),
        # case5 > 11 * 12
        ts_m5 = (145, 'M'),
# daily
        # nobs <= 28 days
        ts_d1 = (20, 'D'),
        # nobs <= 91
        ts_d2 = (75, 'D'),
        # nobs <= 1.15 * 364 (less than 14 months)
        ts_d3 = (350, 'D'),
        #nobs >= 1.15 * 364
        ts_d4 = (None, 'D'),

# business days
        # nobs <= 28 days
        ts_b1 = (20, 'B'),
        # nobs <= 91
        ts_b2 = (91, 'B'),
        # nobs <= 1.15 * 364 (less than 14 months)
        ts_b3 = (350, 'B'),
        #nobs >= 1.15 * 364
        ts_b4 = (None, 'B'),
# hourly
        # random test cases for spacing
        ts_h1 = (20, 'H'),
        ts_h2 = (200, 'H'),
        ts_h3 = (None, 'H'),
# minutes
        # random test cases for spacing
        ts_min1 = (20, 'Min'),
        ts_min2 = (200, 'Min'),
        ts_min3= (None, 'Min'),
# seconds
        # random test cases for spacing
        ts_s1 = (20, 'S'),
        ts_s2 = (200, 'S'),
        ts_s3 = (None, 'S'),
    )

arrz = dict()
for kw, val in cases.iteritems():
    arrz.update({kw : get_info(make_tseries(*val))})

np.savez('timeseries_info_tests.npz', **arrz)
arr = np.load('timeseries_info_tests.npz')
#TODO: irregular
