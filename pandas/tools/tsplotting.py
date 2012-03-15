"""
Code originally written for scikits.timeseries written by Pierre FG
Gerard-Marchant & Matt Knox. Moved to pandas by Skipper Seabold.
"""
import datetime

import numpy as np

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt

from matplotlib.ticker import Formatter, ScalarFormatter, FuncFormatter, \
                              Locator, FixedLocator, MultipleLocator

from pandas import TimeSeries, datetools
from pandas.core.index import DatetimeIndex
import pandas #TODO: clean up imports

#NOTE: I put HACK everywhere that I have to do something in a suboptimal way

# Generic documentation ........................................................

_doc_parameters = dict(
figsize="""figsize : {None, tuple}
        Size of the figure, as a tuple (width, height) in inches.
        If None, defaults to rc figure.figsize.""",
dpi="""dpi : {None, int}, optional
        Resolution in dots per inches.
        If None, defaults to rc figure.dpi.""",
facecolor="""facecolor : {None, string}, optional
        Background color.
        If None, defaults to rc figure.facecolor.""",
edgecolor="""edgecolor : {None, string}, optional
        Border color.
        If None, defaults to rc figure.edgecolor.""",
linewidth="""linewidth : {float, None}
        Width of the patch edge line.""",
frameon="""frameon : {True, False}
        Whether to draw the frame around the figure.""",
subplotpars="""subplotpars : {None, var}
        A :class:`SubplotParams` instance, defaults to rc""",
mandatoryplotargs="""args : var
        Mandatory arguments for the creation of the subplot.
        These arguments should be given as ``nb_of_rows``, ``nb_of_columns``,
        ``plot_number``, or as a single 3-digit number if the 3 previous numbers
        are all lower than 10.""",
                       )

kw_parameters = """
    num : {None, int}, optional
        Number of the figure.
        If None, a new figure is created and ``num`` is incremented.
    %(figsize)s
    %(dpi)s
    %(facecolor)s
    %(edgecolor)s
    %(frameon)s
    %(subplotpars)s
    FigureClass : FigureClass
        Class of the figure to create. Default is TSFigure.
"""
#####----- Helper Funtions

def _check_is_timeseries(series):
    if not isinstance(series, TimeSeries):
        raise ValueError("series is not a TimeSeries. Got type: %s"
                         % type(series))

#NOTE: this was taken from statsmodels, it's a placeholder for now
def _add_datetimes(dates):
    return reduce(lambda x, y: y+x, dates)

def _infer_freq(dates):
    timedelta = datetime.timedelta
    nobs = min(len(dates), 6)
    if nobs == 1:
        raise ValueError("Cannot infer frequency from one date")
    diff = [dates[i+1] - dates[i] for i in range(nobs-1)]
    delta = _add_datetimes(diff)
    nobs -= 1 # after diff
    if delta == timedelta(nobs): #greedily assume 'D'
        return 'D'
    elif delta == timedelta(nobs + 2):
        return 'B'
    elif delta == timedelta(7*nobs):
        return 'W'
    elif delta >= timedelta(28*nobs) and delta <= timedelta(31*nobs):
        return 'M'
    elif delta >= timedelta(90*nobs) and delta <= timedelta(92*nobs):
        return 'Q'
    elif delta >= timedelta(365 * nobs) and delta <= timedelta(366 * nobs):
        return 'A'
    elif delta == timedelta(0, 3600 * nobs):
        return 'H'
    elif delta == timedelta(0, 60 * nobs):
        return 'Min'
    elif delta == timedelta(0, nobs):
        return 'S'
    else:
        return

_freq_to_offset = {
        'B' : datetools.BDay(1),
        'D' : datetools.day,
        'W' : datetools.Week(weekday=6),
        'M' : datetools.monthEnd,
        'A' : datetools.yearEnd,
        'Q' : datetools.quarterEnd,
        'H' : datetools.Hour(),
        'Min' : datetools.Minute(),
        'S' : datetools.Second()
        }

_freq_to_time_rule = {
    'A' : 'A@DEC',
    #    'Q' : 'Q@MAR',
    'M' : 'EOM',
    'W' : 'W@SUN',
    'B' : 'WEEKDAY'
}

def pandas_strftime(x, fmt):
    fmt_ = fmt.lower()
    if ('%q' in fmt_) or ('%f' in fmt_) or ('%F' in fmt):
        #NOTE: this is only going to return "standard" quarters for now
        fmt = fmt.replace('%q', '%1d' % (x.month // 4 + 1))
        fmt = fmt.replace('%Q', '%1d' % (x.month // 4 + 1))
        fmt = fmt.replace('%f', '%02d' % (x.year % 100))
        fmt = fmt.replace('%F', '%d' % x.year)

    return x.strftime(fmt)

def _less_than_hr(freq):
    return freq in ['H', 'Min', 'S']

#####---------------------------------------------------------------------------
#---- --- Matplotlib extensions ---
#####---------------------------------------------------------------------------

def add_generic_subplot(figure_instance, *args, **kwargs):
    """
    Generalizes the :meth:`matplotlib.Figure.add_subplot` method
    of :class:`~matplotlib.figure.Figure` to generic subplots.
    The specific Subplot object class to add is given through the keywords
    ``SubplotClass`` or ``class``.

    Parameters
    ----------
    figure_instance : Figure object
        Figure to which the generic subplot should be attached.
    args : {var}
        Miscellaneous arguments to the subplot.
    kwargs : {Dictionary}
        Optional keywords.
        The same keywords as ``Subplot`` are recognized, with the addition of:

        + *SubplotClass* : {string}
          Type of subplot.
        + *subclass* : {string}
          Shortcut to SubplotClass.
        + any keyword required by the ``SubplotClass`` subclass.

    """
    key = figure_instance._make_key(*args, ** kwargs)
    #!!!: Find why, sometimes, key is not hashable (even if tuple)
    # else, there's a fix below
    try:
        key.__hash__()
    except TypeError:
        key = str(key)

    if figure_instance._axstack.get(key):
        figure_instance.sca(ax)
        return ax

    SubplotClass = kwargs.pop("SubplotClass", Subplot)
    SubplotClass = kwargs.pop("subclass", SubplotClass)
    if isinstance(args[0], Subplot):
        a = args[0]
        assert(a.get_figure() is figure_instance)
    else:
        a = SubplotClass(figure_instance, *args, **kwargs)

    figure_instance.axes.append(a)
    figure_instance._axstack.add(key, a)
    figure_instance.sca(a)
    return a

##### -------------------------------------------------------------------------
#---- --- Freq -> Index Locators ---
##### -------------------------------------------------------------------------

def _get_default_annual_spacing(nyears):
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        (min_spacing, maj_spacing) = (1, 1)
    elif nyears < 20:
        (min_spacing, maj_spacing) = (1, 2)
    elif nyears < 50:
        (min_spacing, maj_spacing) = (1, 5)
    elif nyears < 100:
        (min_spacing, maj_spacing) = (5, 10)
    elif nyears < 200:
        (min_spacing, maj_spacing) = (5, 25)
    elif nyears < 600:
        (min_spacing, maj_spacing) = (10, 50)
    else:
        factor = nyears // 1000 + 1
        (min_spacing, maj_spacing) = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)


def period_break(dates, period, freq):
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : DateArray
        Array of dates to monitor.
    period : string
        Name of the period to monitor.
    """
    if period == 'week': # HACK workaround until DatetimeIndex has week
        current = np.array([i.isocalendar()[-1] for i in dates])
        previous = np.array([i.isocalendar()[-1] for i in dates.shift(-1,
                                    _freq_to_offset[freq])])
    else:
        current = getattr(dates, period)
        previous = getattr(dates.shift(-1, _freq_to_offset[freq]), period)
    return (current - previous).nonzero()[0]


def has_level_label(label_flags, vmin):
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    if the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
    if label_flags.size == 0 or (label_flags.size == 1 and
                                 label_flags[0] == 0 and
                                 (vmin % 1) > 0.0):
        return False
    else:
        return True

#NOTE: to have interactive tick locators span has to be inferred
#dynamically from changing vmin and vmax, it can't be given by len(dates)
#so... how to do this with the current i8 dates? I dunno. -ss

def _daily_finder(vmin, vmax, freq, span, dates):

    periodsperday = -1
    if freq not in ['B','D','W','H','Min','S', None]:
        raise ValueError("Unexpected frequency")

    if _less_than_hr(freq):
        if freq == 'S':
            periodsperday = 24 * 60 * 60
        elif freq == 'Min':
            periodsperday = 24 * 60
        elif freq == 'H':
            periodsperday = 24
        else:
            raise ValueError("unexpected frequency: %s" % check_freq_str(freq))
        periodsperyear = 365 * periodsperday
        periodspermonth = 28 * periodsperday

    elif freq == 'B':
        periodsperyear = 261
        periodspermonth = 19
    elif freq == 'D':
        periodsperyear = 365
        periodspermonth = 28
    elif freq == 'W':
        periodsperyear = 52
        periodspermonth = 3
    elif freq == None:
        periodsperyear = 100
        periodspermonth = 10
    else:
        raise ValueError("unexpected frequency")

    # save this for later usage
    vmin_orig = vmin

    #(vmin, vmax) = (int(vmin), int(vmax))
    #span = vmax - vmin + 1
    #dates_ = date_array(start_date=Date(freq, vmin),
    #                    end_date=Date(freq, vmax))
    # Initialize the output
    info = np.zeros(span,
                    dtype=[('val', int), ('maj', bool), ('min', bool),
                           ('fmt', '|S20')])
    #info['val'][:] = np.arange(vmin, vmax + 1)
    info['val'] = dates
    dates_ = info['val']
    info['fmt'] = ''
    info['maj'][[0, -1]] = True
    # .. and set some shortcuts
    info_maj = info['maj']
    info_min = info['min']
    info_fmt = info['fmt']

    def first_label(label_flags):
        if (label_flags[0] == 0) and (label_flags.size > 1) and \
            ((vmin_orig % 1) > 0.0):
                return label_flags[1]
        else:
            return label_flags[0]

    # Case 1. Less than a month
    if span <= periodspermonth:

        day_start = period_break(dates, 'day', freq)
        month_start = period_break(dates, 'month', freq)

        def _hour_finder(label_interval, force_year_start):
            _hour = dates_.hour
            _prev_hour = (dates-1).hour
            hour_start = (_hour - _prev_hour) != 0
            info_maj[day_start] = True
            info_min[hour_start & (_hour % label_interval == 0)] = True
            year_start = period_break(dates, 'year', freq)
            info_fmt[hour_start & (_hour % label_interval == 0)] = '%H:%M'
            info_fmt[day_start] = '%H:%M\n%d-%b'
            info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'
            if force_year_start and not has_level_label(year_start, vmin_orig):
                info_fmt[first_label(day_start)] = '%H:%M\n%d-%b\n%Y'

        def _minute_finder(label_interval):
            hour_start = period_break(dates, 'hour', freq)
            _minute = dates.minute
            _prev_minute = (dates-1).minute
            minute_start = (_minute - _prev_minute) != 0
            info_maj[hour_start] = True
            info_min[minute_start & (_minute % label_interval == 0)] = True
            year_start = period_break(dates, 'year', freq)
            info_fmt = info['fmt']
            info_fmt[minute_start & (_minute % label_interval == 0)] = '%H:%M'
            info_fmt[day_start] = '%H:%M\n%d-%b'
            info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'

        def _second_finder(label_interval):
            minute_start = period_break(dates, 'minute', freq)
            _second = dates.second
            _prev_second = (dates-1).second
            second_start = (_second - _prev_second) != 0
            info['maj'][minute_start] = True
            info['min'][second_start & (_second % label_interval == 0)] = True
            year_start = period_break(dates, 'year', freq)
            info_fmt = info['fmt']
            info_fmt[second_start & (_second % label_interval == 0)] = '%H:%M:%S'
            info_fmt[day_start] = '%H:%M:%S\n%d-%b'
            info_fmt[year_start] = '%H:%M:%S\n%d-%b\n%Y'

        if span < periodsperday / 12000.0: _second_finder(1)
        elif span < periodsperday / 6000.0: _second_finder(2)
        elif span < periodsperday / 2400.0: _second_finder(5)
        elif span < periodsperday / 1200.0: _second_finder(10)
        elif span < periodsperday / 800.0: _second_finder(15)
        elif span < periodsperday / 400.0: _second_finder(30)
        elif span < periodsperday / 150.0: _minute_finder(1)
        elif span < periodsperday / 70.0: _minute_finder(2)
        elif span < periodsperday / 24.0: _minute_finder(5)
        elif span < periodsperday / 12.0: _minute_finder(15)
        elif span < periodsperday / 6.0:  _minute_finder(30)
        elif span < periodsperday / 2.5: _hour_finder(1, False)
        elif span < periodsperday / 1.5: _hour_finder(2, False)
        elif span < periodsperday * 1.25: _hour_finder(3, False)
        elif span < periodsperday * 2.5: _hour_finder(6, True)
        elif span < periodsperday * 4: _hour_finder(12, True)
        else:
            info_maj[month_start] = True
            info_min[day_start] = True
            year_start = period_break(dates, 'year', freq)
            info_fmt = info['fmt']
            info_fmt[day_start] = '%d'
            info_fmt[month_start] = '%d\n%b'
            info_fmt[year_start] = '%d\n%b\n%Y'
            if not has_level_label(year_start, vmin_orig):
                if not has_level_label(month_start, vmin_orig):
                    info_fmt[first_label(day_start)] = '%d\n%b\n%Y'
                else:
                    info_fmt[first_label(month_start)] = '%d\n%b\n%Y'

    # Case 2. Less than three months
    elif span <= periodsperyear // 4:
        month_start = period_break(dates, 'month', freq)
        info_maj[month_start] = True
        if _less_than_hr(freq):
            info['min'] = True
        else:
            day_start = period_break(dates, 'day', freq)
            info['min'][day_start] = True
        week_start = period_break(dates, 'week', freq)
        year_start = period_break(dates, 'year', freq)
        info_fmt[week_start] = '%d'
        info_fmt[month_start] = '\n\n%b'
        info_fmt[year_start] = '\n\n%b\n%Y'
        if not has_level_label(year_start, vmin_orig):
            if not has_level_label(month_start, vmin_orig):
                info_fmt[first_label(week_start)] = '\n\n%b\n%Y'
            else:
                info_fmt[first_label(month_start)] = '\n\n%b\n%Y'
    # Case 3. Less than 14 months ...............
    elif span <= 1.15 * periodsperyear:
        year_start = period_break(dates, 'year', freq)
        month_start = period_break(dates, 'month', freq)
        week_start = period_break(dates, 'week', freq)
        info_maj[month_start] = True
        info_min[week_start] = True
        info_min[year_start] = False
        info_min[month_start] = False
        info_fmt[month_start] = '%b'
        info_fmt[year_start] = '%b\n%Y'
        if not has_level_label(year_start, vmin_orig):
            info_fmt[first_label(month_start)] = '%b\n%Y'
    # Case 4. Less than 2.5 years ...............
    elif span <= 2.5 * periodsperyear:
        year_start = period_break(dates, 'year', freq)
        quarter_start = period_break(dates, 'quarter', freq)
        month_start = period_break(dates, 'month', freq)
        info_maj[quarter_start] = True
        info_min[month_start] = True
        info_fmt[quarter_start] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    # Case 4. Less than 4 years .................
    elif span <= 4 * periodsperyear:
        year_start = period_break(dates, 'year', freq)
        month_start = period_break(dates, 'month', freq)
        info_maj[year_start] = True
        info_min[month_start] = True
        info_min[year_start] = False

        month_break = dates[month_start].month
        jan_or_jul = month_start[(month_break == 1) | (month_break == 7)]
        info_fmt[jan_or_jul] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    # Case 5. Less than 11 years ................
    elif span <= 11 * periodsperyear:
        year_start = period_break(dates, 'year', freq)
        quarter_start = period_break(dates, 'quarter', freq)
        info_maj[year_start] = True
        info_min[quarter_start] = True
        info_min[year_start] = False
        info_fmt[year_start] = '%Y'
    # Case 6. More than 12 years ................
    else:
        year_start = period_break(dates, 'year', freq)
        year_break = dates[year_start].year
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[(year_break % maj_anndef == 0)]
        info_maj[major_idx] = True
        minor_idx = year_start[(year_break % min_anndef == 0)]
        info_min[minor_idx] = True
        info_fmt[major_idx] = '%Y'
    #............................................
    return info


def _monthly_finder(vmin, vmax, freq, span, dates):
    if freq != 'M':
        raise ValueError("Unexpected frequency")
    periodsperyear = 12

    #vmin_orig = vmin
    #(vmin, vmax) = (int(vmin), int(vmax))
    #span = vmax - vmin + 1
    #..............
    # Initialize the output
    info = np.zeros(span,
                    dtype=[('val', int), ('maj', bool), ('min', bool),
                           ('fmt', '|S8')])
    #info['val'] = np.arange(vmin, vmax + 1)
    info['val'] = dates
    dates_ = info['val']
    info['fmt'] = ''
    year_start = (dates.month == 1).nonzero()[0]
    info_maj = info['maj']
    info_fmt = info['fmt']
    #..............
    if span <= 1.15 * periodsperyear:
        info_maj[year_start] = True
        info['min'] = True

        info_fmt[:] = '%b'
        info_fmt[year_start] = '%b\n%Y'

        if not has_level_label(year_start, vmin):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = '%b\n%Y'
    #..............
    elif span <= 2.5 * periodsperyear:
        quarter_start = (dates.month % 3 == 1).nonzero()
        info_maj[year_start] = True
        # TODO: this was not correct, check -ss
        info_maj[quarter_start] = True
        info['min'] = True

        info_fmt[quarter_start] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    #..............
    elif span <= 4 * periodsperyear:
        info_maj[year_start] = True
        info['min'] = True

        jan_or_jul = (dates.month % 12 == 1) | (dates.month % 12 == 7)
        info_fmt[jan_or_jul] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    #..............
    elif span <= 11 * periodsperyear:
        quarter_start = (dates.month % 3 == 1).nonzero()
        info_maj[year_start] = True
        info['min'][quarter_start] = True

        info_fmt[year_start] = '%Y'
    #..................
    else:
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[::maj_anndef]
        info_maj[major_idx] = True
        info['min'][year_start[::min_anndef]] = True

        info_fmt[major_idx] = '%Y'
    #..............
    return info


def _quarterly_finder(vmin, vmax, freq, span, dates):
    if freq != 'Q':
        raise ValueError("Unexpected frequency")
    periodsperyear = 4
    #vmin_orig = vmin
    #(vmin, vmax) = (int(vmin), int(vmax))
    #span = vmax - vmin + 1
    #............................................
    info = np.zeros(span,
                    dtype=[('val', int), ('maj', bool), ('min', bool),
                           ('fmt', '|S8')])
    #info['val'] = np.arange(vmin, vmax + 1)
    info['val'] = dates
    info['fmt'] = ''
    dates_ = info['val']
    info_maj = info['maj']
    info_fmt = info['fmt']
    first_qtr = np.min(dates.month) # dont assume standard quarters
    year_start = (dates.month == first_qtr).nonzero()[0]
    #year_start = (dates_ % 4 == 1).nonzero()[0]
    #..............
    if span <= 3.5 * periodsperyear:
        info_maj[year_start] = True
        info['min'] = True

        info_fmt[:] = 'Q%q'
        info_fmt[year_start] = 'Q%q\n%F'
        if not has_level_label(year_start, vmin):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = 'Q%q\n%F'
    #..............
    elif span <= 11 * periodsperyear:
        info_maj[year_start] = True
        info['min'] = True
        info_fmt[year_start] = '%F'
    #..............
    else:
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        major_idx = year_start[::maj_anndef]
        info_maj[major_idx] = True
        info['min'][year_start[::min_anndef]] = True
        info_fmt[major_idx] = '%F'
    #..............
    return info


def _annual_finder(vmin, vmax, freq, span, dates):
    if freq != 'A':
        raise ValueError("Unexpected frequency")
    #(vmin, vmax) = (int(vmin), int(vmax + 1))
    #span = vmax - vmin + 1
    #..............
    info = np.zeros(span+1,
                    dtype=[('val', int), ('maj', bool), ('min', bool),
                           ('fmt', '|S8')])
    #info['val'] = np.arange(vmin, vmax + 1)
    info['val'][:-1] = dates
    info['val'][-1] = dates.fshift(1)[[-1]].asi8 #HACK to get one forward in time
    info['fmt'] = ''
    dates_ = info['val']
    #..............
    (min_anndef, maj_anndef) = _get_default_annual_spacing(span)
    info['maj'][::maj_anndef] = True
    info['min'][::min_anndef] = True
    info['fmt'][info['maj']] = '%Y'
    #..............
    return info


def get_finder(freq):
    if freq == 'A':
        return _annual_finder
    elif freq == 'Q':
        return _quarterly_finder
    elif freq == 'M':
        return _monthly_finder
    elif freq in ['B','D','W','H','Min','S', None]:
        return _daily_finder
    else:
        errmsg = "Unsupported frequency: %s" % check_freq_str(freq)
        raise NotImplementedError(errmsg)

#####----- Tick Locator Functions

class TimeSeries_DateLocator(Locator):
    """
    Locates the ticks along an axis controlled by a :class:`pandas.TimeSeries`.

    Parameters
    ----------
    freq : {var}
        Valid frequency specifier.
    minor_locator : {False, True}, optional
        Whether the locator is for minor ticks (True) or not.
    dynamic_mode : {True, False}, optional
        Whether the locator should work in dynamic mode.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """

    def __init__(self, freq, dates, minor_locator=False, dynamic_mode=True,
                 base=1, quarter=1, month=1, day=1, plot_obj=None):
        self.freq = freq
        self.dates = dates
        self.span = len(dates)
        self.base = base
        (self.quarter, self.month, self.day) = (quarter, month, day)
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def asminor(self):
        "Returns the locator set to minor mode."
        self.isminor = True
        return self

    def asmajor(self):
        "Returns the locator set to major mode."
        self.isminor = False
        return self

    def _get_default_locs(self, vmin, vmax):
        "Returns the default locations of ticks."

        if self.plot_obj.date_axis_info is None:
            self.plot_obj.date_axis_info = self.finder(vmin, vmax,
                                                       self.freq, self.span,
                                                       self.dates)

        locator = self.plot_obj.date_axis_info

        if self.isminor:
            return np.compress(locator['min'], locator['val'])
        return np.compress(locator['maj'], locator['val'])

    def __call__(self):
        'Return the locations of the ticks.'

        vi = tuple(self.axis.get_view_interval())
        if vi != self.plot_obj.view_interval:
            self.plot_obj.date_axis_info = None
        self.plot_obj.view_interval = vi
        vmin, vmax = vi

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if self.isdynamic:
            locs = self._get_default_locs(vmin, vmax)
        else:
            base = self.base
            (d, m) = divmod(vmin, base)
            vmin = (d + 1) * base
            locs = self.dates # just use the dates for default TODO: fixme
            #locs = range(int(vmin), int(vmax) + 1, base)
        return locs

    def autoscale(self):
        """
    Sets the view limits to the nearest multiples of base that contain the data.
        """
        # requires matplotlib >= 0.98.0
        (vmin, vmax) = self.axis.get_data_interval()

        locs = self._get_default_locs(vmin, vmax)
        (vmin, vmax) = locs[[0, -1]]
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return nonsingular(vmin, vmax)

#####----- Formatter

class TimeSeries_DateFormatter(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`DateArray`.

    Parameters
    ----------
    freq : {int, string}
        Valid frequency specifier.
    minor_locator : {False, True}
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : {True, False}
        Whether the formatter works in dynamic mode or not.
    """

    def __init__(self, freq, dates, minor_locator=False, dynamic_mode=True,
                 plot_obj=None):
        self.format = None
        self.dates = dates
        self.span = len(dates)
        self.freq = freq
        self.locs = []
        self.formatdict = None
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        self.finder = get_finder(freq)

    def asminor(self):
        "Returns the formatter set to minor mode."
        self.isminor = True
        return self

    def asmajor(self):
        "Returns the fromatter set to major mode."
        self.isminor = False
        return self

    def _set_default_format(self, vmin, vmax):
        "Returns the default ticks spacing."

        if self.plot_obj.date_axis_info is None:
            self.plot_obj.date_axis_info = self.finder(vmin, vmax, self.freq, self.span, self.dates)
        info = self.plot_obj.date_axis_info

        if self.isminor:
            format = np.compress(info['min'] & np.logical_not(info['maj']),
                                 info)
        else:
            format = np.compress(info['maj'], info)
        self.formatdict = dict([(x, f) for (x, _, _, f) in format])
        return self.formatdict

    def set_locs(self, locs):
        'Sets the locations of the ticks'
        # don't actually use the locs. This is just needed to work with
        # matplotlib. Force to use vmin, vmax
        self.locs = locs

        (vmin, vmax) = vi = tuple(self.axis.get_view_interval())
        if vi != self.plot_obj.view_interval:
            self.plot_obj.date_axis_info = None
        self.plot_obj.view_interval = vi
        if vmax < vmin:
            (vmin, vmax) = (vmax, vmin)
        self._set_default_format(vmin, vmax)
    #
    def __call__(self, x, pos=0):
        if self.formatdict is None:
            return ''
        else:
            fmt = self.formatdict.pop(x, '')
            return pandas_strftime(datetools.Timestamp(x), fmt)

#####----- TimeSeries Plots

class TimeSeriesPlot(Subplot, object):
    """
    Based on : :class:`~matplotlib.axes.SubPlot`

    Defines a subclass of :class:`matplotlib.axes.Subplot` to plot time series.

    A :class:`~scikits.timeseries.TimeSeries` is associated with the plot.
    This time series is usually specified at the creation of the plot,
    through the optional parameter ``series``.
    If no series is given at the creation, the first time series being plotted
    will be used as associated series.

    The associated series is stored in the :attr:`~TimeSeriesPlot.series`
    attribute.
    It gives its frequency to the plot.
    This frequency can be accessed through the attribute :attr:`freq`.
    All the other series that will be plotted will be first converted to the
    :attr:`freq` frequency, using their
    :meth:`~scikits.timeseries.TimeSeries.asfreq` method.

    The same parameters used for the instanciation of a standard
    :class:`matplotlib.axes.Subplot` are recognized.

    Parameters
    ----------
    series : {None, TimeSeries}, optional
        The time series allocated to the plot.

    Attributes
    ----------
    freq : int
        Frequency of the plot.
    xdata : DateArray
        The array of dates corresponding to the x axis.
    legendsymbols : list
    legendlabels : list
        List of the labels associated with each plot.
        The first label corresponds to the first plot, the second label to the
        second plot, and so forth.

    Warnings
    --------
    * Because the series to plot are first converted to the plot frequency,
      it is recommended when plotting several series to associate the plot with
      the series with the highest frequency, in order to keep a good level
      of detail.

    """
    def __init__(self, fig=None, *args, **kwargs):

        # Retrieve the series ...................
        _series = kwargs.pop('series', getattr(fig, 'series', None))
        Subplot.__init__(self, fig, *args, **kwargs)

        # Process options .......................
        self.set_series(series=_series)

        self._austoscale = False
        # Get the data to plot
        self.legendsymbols = []
        self.legendlabels = []
        # keep track of axis format and tick info
        self.date_axis_info = None
        # used to keep track of current view interval to determine if we need
        # to reset date_axis_info
        self.view_interval = None


    def set_series(self, series=None):
        """
    Sets the time series associated with the plot.
    If ``series`` is a valid :class:`pandas.TimeSeries` object,
    the :attr:`xdata` attribute is updated to the ``index`` of ``series``.
        """
        if series is not None:
            _check_is_timeseries(series)
            self.xdata = self._series.index
            self._series = _series
        else:
            self._series = None
            self.xdata = None

    def get_series(self):
        """
    Returns the data part of the time series associated with the plot,
    as a (subclass of) :class:`pandas.TimeSeries`.
    """
        return self._series
    #
    series = property(fget=get_series, fset=set_series,
                      doc="Underlying time series.")

    def get_freq(self):
        """
    Returns the underlying frequency of the plot
        """
        #TODO: update this
        return _infer_freq(self.xdata)
    #
    freq = property(fget=get_freq, doc="Underlying frequency.")

    def _check_plot_params(self, *args):
        remaining = list(args)
        noinfo_msg = "No date information available."
        if not len(args):
            if self.xdata is None:
                raise ValueError(noinfo_msg)

        output = []
        while len(remaining):
            arg = remaining.pop(0)
            # the argument is a format: use default dates
            if isinstance(arg, str):
                if self.xdata is None:
                    raise ValueError(noinfo_msg)
                else:
                    output.extend([self.xdata, self.series, arg])

            # argument is a time series use its dates and check for fmt
            elif isinstance(arg, TimeSeries):
                (x, y) = arg.index, arg
                if len(remaining) and isinstance(remaining[0], str):
                    arg = remaining.pop(0)
                    output.extend([x, y, arg])
                else: #NOTE: see if it's feasible to just use 0,n
                    #self.dates = x # keep original x around for formatter
                    output.extend([x,y])

            elif len(remaining) and isinstance(remaining[0], str):
                arg2 = remaining.pop(0)
                if self.xdata is None:
                    raise ValueError(noinfo_msg)
                else:
                    output.extend([self.xdata, arg, arg2])

            elif self.xdata is None:
                raise ValueError(noinf_msg)

            else:
                output.extend([self.xdata, arg])

        # reinitialize the plot if needed
        if self.xdata is None:
            self.xdata = output[0]

        # force xdata to current frequency
        #elif output[0].freq != self.freq:
        #    output = list(output)
        #    try:
        #        output[0] = convert_to_float(output[0], self.freq)
        #    except NotImplementedError:
        #        output[0] = output[0].asfreq(self.freq)

        return output

    def tsplot(self, *args,  **kwargs):
        """
    Plots the data parsed in argument to the current axes.
    This command accepts the same optional keywords as :func:`matplotlib.pyplot.plot`.

    The argument ``args`` is a variable length argument, allowing for multiple
    data to be plotted at once. Acceptable combinations are:

    No arguments or a format string:
       The time series associated with the subplot is plotted with the given
       format.
       If no format string is given, the default format is used instead.
       For example, to plot the underlying time series with the default format,
       use:

          >>> tsplot()

       To plot the underlying time series with a red solid line, use the command:

          >>> tsplot('r-')

    a :class:`~scikits.timeseries.TimeSeries` object or one of its subclass
    with or without a format string:
       The given time series is plotted with the given format.
       If no format string is given, the default format is used instead.

    an array or sequence, with or without a format string:
       The data is plotted with the given format
       using the :attr:`~TimeSeriesPlot.xdata` attribute of the plot as abscissae.

    two arrays or sequences, with or without a format string:
       The data are plotted with the given format, using the first array as
       abscissae and the second as ordinates.


    Parameters
    ----------
    args : var
        Sequence of arguments, as described previously.
    kwargs : var
        Optional parameters.
        The same parameters are accepted as for :meth:`matplotlib.axes.Subplot.plot`.

        """
        args = self._check_plot_params(*args)
        self.legendlabels.append(kwargs.get('label', None))
        plotted = Subplot.plot(self, *args,  **kwargs)
        self.format_dateaxis()

        # when adding a right axis (using add_yaxis), for some reason the
        # x axis limits don't get properly set. This gets around the problem
        xlim = self.get_xlim()
        if xlim[0] == 0.0 and xlim[1] == 1.0:
            # if xlim still at default values, autoscale the axis
            self.autoscale_view()
        self.reset_datelimits()
        return plotted


    def format_dateaxis(self):
        """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the current
    underlying series.
    As the dynamic mode is activated by default, changing the limits of the x
    axis will intelligently change the positions of the ticks.
        """
        # Get the locator class .................
        majlocator = TimeSeries_DateLocator(self.freq, self.xdata,
                                            dynamic_mode=True,
                                            minor_locator=False, plot_obj=self)
        minlocator = TimeSeries_DateLocator(self.freq, self.xdata,
                                            dynamic_mode=True,
                                            minor_locator=True, plot_obj=self)
        self.xaxis.set_major_locator(majlocator)
        self.xaxis.set_minor_locator(minlocator)
        # Get the formatter .....................
        majformatter = TimeSeries_DateFormatter(self.freq, self.xdata,
                                                dynamic_mode=True,
                                                minor_locator=False,
                                                plot_obj=self)
        minformatter = TimeSeries_DateFormatter(self.freq, self.xdata,
                                                dynamic_mode=True,
                                                minor_locator=True,
                                                plot_obj=self)
        self.xaxis.set_major_formatter(majformatter)
        self.xaxis.set_minor_formatter(minformatter)
        plt.draw_if_interactive()

    def set_dlim(self, start_date=None, end_date=None):
        """
    Sets the date limits of the plot to ``start_date`` and ``end_date``.
    The dates can be given as :class:`~scikits.timeseries.Date` objects,
    strings or integers.

    Parameters
    ----------
    start_date : {var}
        Starting date of the plot. If None, the current left limit (earliest
        date) is used.
    end_date : {var}
        Ending date of the plot. If None, the current right limit (latest date)
        is used.
        """
        freq = self.freq
        if freq is None:
            raise ValueError("Undefined frequency! Date limits can't be set!")
        # TODO : Shouldn't we make get_datevalue a more generic function ?
        def get_datevalue(date, freq):
            if isinstance(date, Date):
                return date.asfreq(freq).value
            elif isinstance(date, str):
                return Date(freq, string=date).value
            elif isinstance(date, (int, float)) or \
                (isinstance(date, np.ndarray) and (date.size == 1)):
                return date
            elif date is None:
                return None
            raise ValueError("Unrecognizable date '%s'" % date)
        # Fix left limit ..............
        xleft = get_datevalue(start_date, freq)
        # Fix right limit .......
        xright = get_datevalue(end_date, freq)
        self.set_xlim(xleft, xright)
        return (xleft, xright)

    def reset_datelimits(self):
        """
    Reset the date range of the x axis to the date range of the underlying
    time series.
        """
        #return self.set_xlim()
        return self.set_xlim(self.xdata[[0, -1]].asi8)

    def get_dlim(self):
        """
    Returns the limits of the x axis as a :class:`~scikits.timeseries.DateArray`.
        """
        xlims = self.get_xlim()
        return DateArray(xlims, freq=self.freq)


#####----- TimeSeries Figures

class TimeSeriesFigure(Figure):
    """
    Based on :class:`matplotlib.figure.Figure`

    Create a new :class:`~matplotlib.figure.Figure` object.
    All the subplots share the same time series.

    The same parameters used for the creation of a standard
    :class:`~matplotlib.figure.Figure` are accepted.

    Parameters
    ----------
    series : {None, TimeSeries}, optional
        Underlying time series.
        All the subplots of the figure will share the same series.
    figsize : {None, tuple}
        Size of the figure, as a tuple (width, height) in inches.
        If None, defaults to rc figure.figsize.
    dpi : {None, int}, optional
        Resolution in dots per inches.
        If None, defaults to rc figure.dpi
    facecolor : {None, string}, optional
        Background color.
        If None, defaults to rc figure.facecolor.
    edgecolor : {None, string}, optional
        Border color.
        If None, defaults to rc figure.edgecolor.
    linewidth : {float, None}
        Width of the patch edge line.
    frameon : {True, False}
        Whether to draw the frame around the figure.

    """
    def __init__(self, **kwargs):
        self._series = series = kwargs.pop('series', None)
        Figure.__init__(self, **kwargs)
        fspnum = kwargs.pop('fspnum', None)
        if fspnum is not None:
            self.add_tsplot(fspnum, series=series)


    def add_tsplot(self, *args, **kwargs):
        """
    Adds a :class:`TimeSeriesPlot` subplot to the current figure.

    Parameters
    ----------
    args : var
        Mandatory arguments for the creation of the subplot.
        These arguments should be given as ``nb_of_rows``, ``nb_of_columns``,
        ``plot_number``, or as a single 3-digit number if the 3 previous numbers
        are all lower than 10.
    kwargs : var
        Optional arguments, as recognized by `add_subplot`.
        """
        kwargs.update(SubplotClass=TimeSeriesPlot)
        if self._series is not None:
            kwargs.update(series=self._series)
        return add_generic_subplot(self, *args, **kwargs)

    add_subplot = add_tsplot
TSFigure = TimeSeriesFigure

def tsfigure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None,
             frameon=True, subplotpars=None, series=None, FigureClass=TSFigure):
    """
    Creates a new :class:`TimeSeriesFigure` object.

    Parameters
    ----------
    num : {None, int}, optional
        Number of the figure.
        If None, a new figure is created and ``num`` is incremented.
    %(figsize)s
    %(dpi)s
    %(facecolor)s
    %(edgecolor)s
    %(frameon)s
    %(subplotpars)s
    FigureClass : FigureClass
        Class of the figure to create
    """
    figargs = dict(num=num, figsize=figsize, dpi=dpi, facecolor=facecolor,
                   frameon=frameon, FigureClass=FigureClass,
                   subplotpars=subplotpars, series=series)
    fig = plt.figure(**figargs)
    return fig
tsfigure.__doc__ %= _doc_parameters

#####----- Pandas TimeSeries Mix-In?

class Plotting(object):
    def plot_series(self, num=None, figsize=None, dpi=None, facecolor=None,
                          edgecolor=None, frameon=True, subplotpars=None,
                          series=None, FigureClass=TSFigure):
        """

        Parameters
        ----------

        kwargs are used to instantiate the matplotlib.figure

        %(kwargs)
        """
        # deal with kwargs
        fig = plt.figure(num=num, figsize=figsize, dpi=dpi,
                   facecolor=facecolor, edgecolor=edgecolor,
                   frameon=frameon, subplotpars=subplotpars, series=series,
                   FigureClass=FigureClass)

        ax = fig.add_tsplot()



    #kw_parameters %= _doc_parameters
    #plot_series.__doc__ %= dict(kwargs=kw_parameters)

####------ time_series factory function prototype


def time_series(data, dates=None, start=None, freq=None, offset=None):
    """
    Parameters
    ----------
    data : array-like
        Can be array-like or pandas.TimeSeries
    dates : array-like, optional
        Date array
    start : datetime-like, optional
        The starting date if dates aren't given.
    freq : str
        An easy entry point to frequency. Defaults to end of period. If
        you want more control, use offset. Available options are
        'A' - Annual
        'Q' - Quarterly
        'M' - Monthly
        'W' - Weekly
        'B' - Business Day
        'D' - Daily
        'H' - Hourly
        'Min' - Minutes
        'S' - Seconds
    offset : str, optional
        For finer control over frequency. pandas offset.
    """
    #TODO: make it so offset can take the current freq strings, eg A@JAN
    if dates is None:
        _dates = getattr(data, 'index', None)
    elif isinstance(dates, (DatetimeIndex, pandas.DateRange)):
        pass # pass to a factory function?
        _dates = dates
    elif isinstance(dates, (tuple, list, np.ndarray)):
        _dates = DatetimeIndex(dates)

    if _dates is not None:
        pass # check the frequency if we have a freq arg?
    else:
        if start is None:
            raise ValueError("Must specify dates or start")
        if freq is None and offset is None:
            raise ValueError("Must specify either freq or offset with start")
        if freq is not None:
            #_check_freq
            offset = _freq_to_offset.get(freq)
            if offset is None:
                raise ValueError("freq not understood")
        trule = _freq_to_time_rule.get(freq)
        if trule:
            #HACK do this shifting in the datetimeindex instantiation
            offset = _freq_to_offset[freq]
            if not offset.onOffset(start):
                start = offset.rollforward(start)
            _dates = DatetimeIndex(start=start, freq=trule, n=len(data))
        else:
            #NOTE: these indices won't carry freq information yet
            # so we'll have to workaround this
            _dates = DatetimeIndex(pandas.DateRange(start=start, offset=offset,
                                  periods=len(data)))


    return TimeSeries(data, index = _dates)

if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas
    import datetime
    #dtq = sm.tsa.datetools.dates_from_range("1990:Q1", "2000:Q4")
    # so we have _some_ kind of frequency info
    dtq = DatetimeIndex(
            start=datetime.datetime(1990, 1, 31),
            end=datetime.datetime(2001,1,31),
            freq='Q@JAN')
    dtq_a = DatetimeIndex(
            start=datetime.datetime(1990, 1, 31),
            end=datetime.datetime(2001,1,31),
            freq='A@JAN')
    assert _infer_freq(dtq) == 'Q'
    assert _infer_freq(dtq_a) == 'A'
    x = np.random.random(45)

    #info = _quarterly_finder(0,45,'Q')
    # for debugging
    import scikits.timeseries as ts
    tsdta = ts.time_series(x,
            dates=map(datetime.datetime.fromordinal,
                      [i.toordinal() for i in dtq]), freq='Q')

    tseries = TimeSeries(x, index=dtq)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(tseries)
    #plt.show()

    X = np.random.random(55)

    d1 = datetime.datetime(1975, 1, 1)
    # annual
    #ts_a = time_series(X, start=d1, freq='A')
    #vmin, vmax = ts_a.index.asi8[[0,-1]]
    #info_a = _annual_finder(vmin, vmax,'A', len(ts_a), ts_a.index)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_a)
    #ax.set_title('Annual')
    #plt.show()

    # quarterly
    # case1 <= 3.5 * 4
    # case2 <= 11 * 4
    # case3 > 11 * 4
    #ts_q = time_series(X[:10], start=d1, freq='Q')
    #vmin, vmax = ts_q.index.asi8[[0,-1]]
    #info_q = _quarterly_finder(vmin, vmax,'Q', len(ts_q), ts_q.index)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_q)
    #ax.set_title('Quarterly')
    #plt.show()

    # monthly
    # case1 <= 1.15 * 12
    # case2 <= 2.5 * 12
    # case3 <= 4 * 12
    # case4 <= 11 * 12
    # case5 > 11 * 12
    #TODO: as soon as you zoom on the axis and Feb gets a level label
    # it keeps it. I don't know why yet
    #ts_mon = time_series(X[:10], start=d1, freq='M')
    #vmin, vmax = ts_mon.index.asi8[[0,-1]]
    #info_mon = _monthly_finder(vmin, vmax,'M', len(ts_mon), ts_mon.index)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_mon)
    #ax.set_title('Monthly')
    #plt.show()

    ## daily
    # case1 < one month
    #ts_d = time_series(X[:20], start=d1, freq='D')
    #vmin, vmax = ts_d.index.asi8[[0,-1]]
    #info_d = _daily_finder(vmin, vmax,'D', len(ts_d), ts_d.index)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_d)
    #ax.set_title('Daily')
    #plt.show()

    # case 2 < 365 // 4
    #TODO: this needs some rethinking... too crowded
    #ts_d = time_series(X, start=d1, freq='D')
    #vmin, vmax = ts_d.index.asi8[[0,-1]]
    #info_d = _daily_finder(vmin, vmax,'D', len(ts_d), ts_d.index)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_d)
    #ax.set_title('Daily')
    #plt.show()

    # case 3 <= 1.15 * 364 Less than 14 months
    #X = np.random.random(364)
    #ts_d = time_series(X, start=d1, freq='D')
    #vmin, vmax = ts_d.index.asi8[[0,-1]]
    #info_d = _daily_finder(vmin, vmax,'D', len(ts_d), ts_d.index)
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_d)
    #ax.set_title('Daily')
    #plt.show()

    ## business daily
    #ts_b = time_series(X, start=d1, freq='B')
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_b)
    #ax.set_title('Business Days')
    #plt.show()

    ## minutes
    #ts_min = time_series(X, start=d1, freq='Min')
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_min)
    #ax.set_title('Minutes')
    #plt.show()

    ## seconds
    #ts_s = time_series(X, start=d1, freq='S')
    #fig = tsfigure()
    #ax = fig.add_tsplot(111)
    #ax.tsplot(ts_s)
    #ax.set_title('Seconds')
    #plt.show()

    # irregular

    #dhours = pandas.DateRange(start=datetime.datetime(2010,1,15),
    #                      end=datetime.datetime(2010,1,16),
    #                      offset=datetools.Hour())

    #dminutes = pandas.DateRange(start=datetime.datetime(2010,1,15),
    #                      end=datetime.datetime(2010,1,16),
    #                      offset=datetools.Minute())

    #dseconds = pandas.DateRange(start=datetime.datetime(2010,1,15),
    #                      end=datetime.datetime(2010,1,16),
    #                      offset=datetools.Second())

    # try to replicate examples from ts docs

    ## ts/docs/source/plotting/zoom1.py
    ## generate some random data
    #num_points = 12
    #data = np.cumprod(1 + np.random.normal(0, 1, num_points)/100)
    #series = ts.time_series(data, start_date=ts.now('d')-num_points)
    #fig = tsfigure()
    #fsp = fig.add_tsplot(111)
    #fsp.tsplot(series, '-')
    #fsp.set_xlim(int(series.start_date), int(series.end_date))
    #fsp.set_title('%i days' % num_points)

    ## ts/docs/source/plotting/zoom2.py
    ## generate some random data
    #num_points = 55
    #data = np.cumprod(1 + np.random.normal(0, 1, num_points)/100)
    #series = ts.time_series(data, start_date=ts.now('d')-num_points)
    #fig = tsfigure()
    #fsp = fig.add_tsplot(111)
    #fsp.tsplot(series, '-')
    #fsp.set_xlim(int(series.start_date), int(series.end_date))
    #fsp.set_title('%i days' % num_points)

    ## ts/docs/source/plotting/zoom3.py
    ## generate some random data
    #num_points = 250
    #data = np.cumprod(1 + np.random.normal(0, 1, num_points)/100)
    #series = ts.time_series(data, start_date=ts.now('d')-num_points)
    #fig = tsfigure()
    #fsp = fig.add_tsplot(111)
    #fsp.tsplot(series, '-')
    #fsp.set_xlim(int(series.start_date), int(series.end_date))
    #fsp.set_title('%i days' % num_points)

    ## ts/docs/source/plotting/zoom4.py
    ## generate some random data
    #num_points = 5000
    #data = np.cumprod(1 + np.random.normal(0, 1, num_points)/100)
    #series = ts.time_series(data, start_date=ts.now('d')-num_points)
    #fig = tsfigure()
    #fsp = fig.add_tsplot(111)
    #fsp.tsplot(series, '-')
    #fsp.set_xlim(int(series.start_date), int(series.end_date))
    #fsp.set_title('%i days' % num_points)
    #plt.show()
