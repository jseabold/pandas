import datetime
import os

import numpy as np
import numpy.testing as npt

import pandas.util.testing as tm
from pandas.tools.tsplotting import time_series, _infer_freq, get_finder

#TODO: check the finders
#check period_break

####--- some setup

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

def get_info(tseries, vminvmax=None):
    freq = _infer_freq(tseries.index)
    finder = get_finder(freq)
    if not vminvmax:
        vmin, vmax = tseries.index[[0,-1]]
    return finder(vmin, vmax, freq, len(tseries), tseries.index)

def get_ts_info(ts_info, key):
    """
    Do this to drop the scikits.timeseries ordinal value.

    TODO: should we convert this to a datetime or Timestamp and test?
    """
    ts_info = ts_info['ts_a']
    return ts_info[list(ts_info.dtype.names[1:])]

class Data(object):
    @classmethod
    def setupClass(cls):
        np.random.seed(12345)
        X = np.cumprod(1 + np.random.normal(0, 1, 500)/100)
        d1 = datetime.datetime(1975, 1, 1)
        cls.data = X
        cls.start = d1
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        cls.ts_results = np.load(os.path.join(cur_dir,
                                  'timeseries_info_tests.npz'))

    def get_data(self, nobs, freq):
        return time_series(self.data[:nobs], start=self.start, freq=freq)

class TestAnnual(Data):

    def test_annual_plot(self):
        X = self.get_data(*cases['ts_a'])
        panda_info = get_info(X)
        # drop i8 ordinal representation
        panda_info = panda_info[list(panda_info.dtype.names[1:])]
        ts_info = get_ts_info(self.ts_results, 'ts_a')
        npt.assert_equal(panda_info, ts_info)

class TestQuarterlyPlot(Data):
    def test_case1(self):
        # case1 nobs <= 3.5 * 4
        X = self.get_data(*cases['ts_q1'])
        panda_info = get_info(X)
        # drop i8 ordinal representation
        panda_info = panda_info[list(panda_info.dtype.names[1:])]
        ts_info = get_ts_info(self.ts_results, 'ts_q1')
        npt.assert_equal(panda_info, ts_info)

    def test_case2(self):
        #case2 nobs <= 11 * 4
        X = self.get_data(*cases['ts_q2'])
        panda_info = get_info(X)
        # drop i8 ordinal representation
        panda_info = panda_info[list(panda_info.dtype.names[1:])]
        ts_info = get_ts_info(self.ts_results, 'ts_q2')


    def test_case3(self):
        #case3 nobs > 11 * 4
        X = self.get_data(*cases['ts_q3'])
        panda_info = get_info(X)
        # drop i8 ordinal representation
        panda_info = panda_info[list(panda_info.dtype.names[1:])]
        ts_info = get_ts_info(self.ts_results, 'ts_q3')

class TestMonthlyPlot(Data):
    def test_case1(self):
        # case 1 nobs <= 1.15 * 12
        pass

    def test_case2(self):
        # case 2 nobs <= 2.5 * 12
        pass

    def test_case3(self):
        # case 3 nobs <= 4 * 12
        pass

    def test_case4(self):
        # case 4 <= 11 * 12
        pass

    def test_case5(self):
        # case5 > 11 * 12
        pass

class TestDailyPlot(Data):
    def test_case1(self):
        # nobs <= 28 days
        pass

    def test_case2(self):
        # nobs <= 91
        pass

    def test_case3(self):
        # nobs <= 1.15 * 364 (less than 14 months)
        pass

    def test_case4(self):
        #nobs >= 1.15 * 364
        pass

class TestBusinessDays(Data):
    pass

class TestMinutes(Data):
    pass

class TestSeconds(Data):
    pass

class TestIrregular(Data):
    pass
