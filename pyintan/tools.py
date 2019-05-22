from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import numpy as np
import quantities as pq

def read_qstring(f):
    length = np.fromfile(f, dtype='uint32', count=1)[0]
    if length == 0xFFFFFFFF or length == 0:
        return ''
    txt = f.read(length).decode('utf-16')
    return txt


def read_variable_header(f, header):
    info = {}
    for field_name, field_type in header:
        if field_type == 'QString':
            field_value = read_qstring(f)
        else:
            field_value = np.fromfile(f, dtype=field_type, count=1)[0]
        info[field_name] = field_value
    return info


def plural(n):

    # s = plural(n)
    #
    # Utility function to optionally plurailze words based on the value
    # of n.
    if n == 1:
        s = ''
    else:
        s = 's'

    return s


def parse_digital_signal(dig, times):
    '''

    :param dig:
    :param times:
    :return:
    '''
    channels = []
    states = []
    timestamps = []
    unit = times.units
    for i in range(16):
        idx_i = np.where(dig == 2**i)[0]
        if len(idx_i) > 0:
            rising, falling = [], []
            for id in idx_i:
                if dig[id - 1] == 0:
                    rising.append(id)
                if dig[id + 1] == 0:
                    falling.append(id)
            # rising, falling = get_rising_falling_edges(idx_i)
            channels.append(i)
            ts_idx = []
            st = []
            for (r, f) in zip(rising, falling):
                ts_idx.append(int(r))
                st.append(1)
                ts_idx.append(int(f))
                st.append(-1)
            timestamps.append(times[ts_idx])
            states.append(st)

    channels = np.array(channels)
    states = np.array(states)
    timestamps = timestamps

    return channels, states, timestamps


def clip_anas(analog_signals, clipping_times, start_end):
    '''

    Parameters
    ----------
    analog_signals
    times
    clipping_times
    start_end

    '''
    if len(analog_signals.signal) != 0:
        times = analog_signals.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')

        if len(analog_signals.signal.shape) == 2:
            analog_signals.signal = analog_signals.signal[:, idx[0]]
        else:
            analog_signals.signal = analog_signals.signal[idx[0]]
        analog_signals.times = times[idx]


def clip_digital(events, clipping_times, start_end):
    '''

    Parameters
    ----------
    digital_signals
    clipping_times
    start_end

    Returns
    -------

    '''
    if len(events.times) != 0:
        times = events.times.rescale(pq.s)
        if len(clipping_times) == 2:
            idx = np.where((times > clipping_times[0]) & (times < clipping_times[1]))
        elif len(clipping_times) ==  1:
            if start_end == 'start':
                idx = np.where(times > clipping_times[0])
            elif start_end == 'end':
                idx = np.where(times < clipping_times[0])
        else:
            raise AttributeError('clipping_times must be of length 1 or 2')
        events.times = times[idx]
        events.channel_states = events.channel_states[idx]
        events.channels = events.channels[idx]

def clip_times(times, clipping_times, start_end='start'):
    '''

    Parameters
    ----------
    times
    clipping_times
    start_end

    Returns
    -------

    '''
    times.rescale(pq.s)

    if len(clipping_times) == 2:
        idx = np.where((times > clipping_times[0]) & (times <= clipping_times[1]))
    elif len(clipping_times) ==  1:
        if start_end == 'start':
            idx = np.where(times >= clipping_times[0])
        elif start_end == 'end':
            idx = np.where(times <= clipping_times[0])
    else:
        raise AttributeError('clipping_times must be of length 1 or 2')
    times_clip = times[idx]

    return times_clip


def clip_stimulation(stimulation, clipping_times, start_end='start'):
    '''

    Parameters
    ----------
    times
    clipping_times
    start_end

    Returns
    -------

    '''
    times = stimulation.times.rescale(pq.s)
    if len(clipping_times) == 2:
        idx = np.where((times > clipping_times[0]) & (times <= clipping_times[1]))
    elif len(clipping_times) == 1:
        if start_end == 'start':
            idx = np.where(times >= clipping_times[0])
        elif start_end == 'end':
            idx = np.where(times <= clipping_times[0])
    else:
        raise AttributeError('clipping_times must be of length 1 or 2')
    stimulation.times = stimulation.times[idx]
    stimulation.signal = stimulation.signal[idx]
