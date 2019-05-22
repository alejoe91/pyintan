"""
Python library for reading INTAN files.
Depends on: os
            numpy
            quantities
Authors: Alessio Buccino @CINPLA,
         Svenn-Arne Dragly @CINPLA,
         Milad H. Mobarhan @CINPLA,
         Mikkel E. Lepperod @CINPLA
"""

import quantities as pq
import os.path as op
import numpy as np
from datetime import datetime
import locale
import platform

from .tools import parse_digital_signal, clip_anas, clip_digital, clip_times, clip_stimulation
from .intan import read_rhd, read_rhs

_signal_channel_dtype = [
    ('name', 'U64'),
    ('id', 'int64'),
    ('sampling_rate', 'float64'),
    ('dtype', 'U16'),
    ('units', 'U64'),
    ('gain', 'float64'),
    ('offset', 'float64'),
    ('group_id', 'int64'),
]


class AnalogSignal:
    def __init__(self, channel_id, channel_names, signal, times):
        self.signal = signal
        self.channel_names = channel_names,
        self.channel_id = channel_id
        self.times = times

    def __str__(self):
        return "<Intan headstage analog signal:shape: {}>".format(
            self.signal.shape
        )


class AnalogIn:
    def __init__(self, channel_id, channel_names, signal, times):
        self.signal = signal
        self.channel_names = channel_names,
        self.channel_id = channel_id
        self.times = times

    def __str__(self):
        return "<Intan Analog In signals:shape: {}>".format(
            self.signal.shape
        )


class AnalogOut:
    def __init__(self, channel_id, channel_names, signal, times):
        self.signal = signal
        self.channel_names = channel_names,
        self.channel_id = channel_id
        self.times = times

    def __str__(self):
        return "<Intan Analog Out signals:shape: {}>".format(
            self.signal.shape
        )


class DigitalIn:
    def __init__(self, times, channels, channel_states):
        self.times = times
        self.channels = channels
        self.channel_states = channel_states

    def __str__(self):
        return "<Intan Digital In>"


class DigitalOut:
    def __init__(self, times, channels, channel_states):
        self.times = times
        self.channels = channels
        self.channel_states = channel_states

    def __str__(self):
        return "<Intan Digital Out>"


class Stimulation:
    def __init__(self, stim_channels, stim_signal, stim_param, times):
        self.channels = stim_channels
        self.signal = stim_signal
        self.param = stim_param
        self.times = times
        self._compute_extra_params()

    def __str__(self):
        return "<Intan stimulation signals:shape: {}, param: {}>".format(
            self.signal.shape, self.param
        )

    def _compute_extra_params(self):
        #TODO: characterize biphasic, biphasic with delay, and triphasic stim
        idx_non_zero = np.where(self.signal != 0)
        stim_clip = self.signal[idx_non_zero]
        period = np.mean(np.diff(self.times))

        self.current_levels = np.unique(stim_clip)
        self.phase = np.min(np.diff(np.where(np.diff(self.signal) != 0))) * period


class File:
    """
    Class for reading experimental data from an Intan dataset.
    """
    def __init__(self, filename, load_stim=True, verbose=False):
        self.absolute_filename = op.abspath(filename)
        self.fname = op.split(filename)[-1]
        self.absolute_foldername = op.split(filename)[0]
        self.verbose = verbose

        fname = op.split(filename)[-1]
        under_date = fname[:-4].split('_')[-2:]
        # read date in US format
        if platform.system() == 'Windows':
            locale.setlocale(locale.LC_ALL, 'english')
        elif platform.system() == 'Darwin':
            # bad hack...
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
            except Exception:
                pass
        else:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
        self._start_datetime = datetime.strptime(under_date[0] + under_date[1], '%y%m%d%H%M%S')

        header_size = 0
        data_dtype = None
        if self.absolute_filename.endswith('.rhs'):
            self._global_info, self._ordered_channels, data_dtype, \
                header_size, self._block_size = read_rhs(self.absolute_filename)
            self.acquisition_system = 'Intan Recording Stimulation GUI'
            stimulation = True
        elif self.absolute_filename.endswith('.rhd'):
            self._global_info, self._ordered_channels, data_dtype, \
                header_size, self._block_size = read_rhd(self.absolute_filename)
            self.acquisition_system = 'Intan Recording Stimulation GUI'
            stimulation = False
        else:
            raise Exception("Only '.rhd' and '.rhs' files are supported")

        # memmap raw data with the complicated structured dtype
        self._raw_data = np.memmap(self.absolute_filename, dtype=data_dtype, mode='r', offset=header_size)

        self._sample_rate = self._global_info['sampling_rate'] * pq.Hz

        # check timestamp continuity
        timestamp = self._raw_data['timestamp'].flatten()
        self._times = timestamp / self._sample_rate
        assert np.all(np.diff(timestamp) == 1), 'timestamp have gaps'

        # signals
        sig_channels = []
        for c, chan_info in enumerate(self._ordered_channels):
            name = chan_info['native_channel_name']
            chan_id = c  # the chan_id have no meaning in intan
            if chan_info['signal_type'] == 20:
                # exception for temperature
                sig_dtype = 'int16'
            else:
                sig_dtype = 'uint16'
            sig_channels.append((name, chan_id, chan_info['sampling_rate'],
                                sig_dtype, chan_info['units'], chan_info['gain'],
                                chan_info['offset'], chan_info['signal_type']))
        self._sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        self._shape = self._raw_data[self._sig_channels[0]['name']].shape
        self._size = self._raw_data[self._sig_channels[0]['name']].size
        # some channel (temperature) have 1D field so shape 1D
        # because 1 sample per block
        self._anas_chan = [sig for sig in self._sig_channels if 'ANALOG' not in sig['name'] and 'STIM' not in sig['name']]
        anas_len = len(self._anas_chan)
        if self.verbose:
            print('Found ', anas_len, ' recording channels')
        self._analog_in_chan = [sig for sig in self._sig_channels if 'ANALOG-IN' in sig['name']]
        analog_in_len = len(self._analog_in_chan)
        if self.verbose:
            print('Found ', analog_in_len, ' analog in channels')
        self._analog_out_chan = [sig for sig in self._sig_channels if 'ANALOG-OUT' in sig['name']]
        analog_out_len = len(self._analog_out_chan)
        if self.verbose:
            print('Found ', analog_out_len, ' analog out channels')
        self._stim_chan = [sig for sig in self._sig_channels if 'STIM' in sig['name']]
        stim_len = len(self._analog_out_chan)
        if self.verbose:
            print('Found ', stim_len, ' stimulation channels')

        self._anas_dirty = True
        self._anasin_dirty = True
        self._anasout_dirty = True
        self._digin_dirty = True
        self._digout_dirty = True
        self._stim_dirty = True

        # read digital events
        _ = self.digital_in_events
        _ = self.digital_out_events

    @property
    def start_time(self):
        return self._times[0]

    @property
    def datetime(self):
        return self._start_datetime

    @property
    def duration(self):
        return self._times[-1] - self._times[0]

    @property
    def sample_rate(self):
            return self._sample_rate

    @property
    def analog_signals(self):
        if self._anas_dirty:
            anas = self._read_analog(self._anas_chan).T
            self._analog_signals = [AnalogSignal(
                channel_id=range(anas.shape[0]),
                channel_names=[ch['name'] for ch in self._anas_chan],
                signal=anas,
                times=self._times,
            )]
            self._anas_dirty = False
        return self._analog_signals

    @property
    def analog_in_signals(self):
        if self._anasin_dirty:
            anas = self._read_analog(self._analog_in_chan).T
            self._analog_in = [AnalogSignal(
                channel_id=range(anas.shape[0]),
                channel_names=[ch['name'] for ch in self._analog_in_chan],
                signal=anas,
                times=self._times,
            )]
            self._anasin_dirty = False
        return self._analog_in

    @property
    def analog_out_signals(self):
        if self._anasout_dirty:
            anas = self._read_analog(self._analog_out_chan).T
            self._analog_out = [AnalogSignal(
                channel_id=range(anas.shape[0]),
                channel_names=[ch['name'] for ch in self._analog_out_chan],
                signal=anas,
                times=self._times,
            )]
            self._anasout_dirty = False
        return self._analog_out

    @property
    def digital_in_events(self):
        if self._digin_dirty:
            i_start, i_stop, block_size, block_start, block_stop, sl0, sl1 = self._get_block_info()
            try:
                data_chan = self._raw_data['DIGITAL-IN']
                if len(self._shape) == 1:
                    digital_in = data_chan[i_start:i_stop]
                else:
                    digital_in = data_chan[block_start:block_stop].flatten()[sl0:sl1]
                channels_in, states_in, times_in = parse_digital_signal(digital_in, self._times)
                if self.verbose:
                    print('Found ', len(channels_in), ' digital in channels')
            except:
                if self.verbose:
                    print('Found  0  digital in channels')
                self._digital_in = None
                channels_in, states_in, times_in = [], [], []
            self._digital_in = []
            for i, ch in enumerate(channels_in):
                self._digital_in.append(DigitalIn(channels=ch * np.ones(len(times_in[i])),
                                                  channel_states=np.array(states_in[i]),
                                                  times=times_in[i]))
            self._digin_dirty = False
        return self._digital_in

    @property
    def digital_out_events(self):
        if self._digout_dirty:
            i_start, i_stop, block_size, block_start, block_stop, sl0, sl1 = self._get_block_info()
            try:
                data_chan = self._raw_data['DIGITAL-OUT']
                if len(self._shape) == 1:
                    digital_out = data_chan[i_start:i_stop]
                else:
                    digital_out = data_chan[block_start:block_stop].flatten()[sl0:sl1]
                channels_out, states_out, times_out = parse_digital_signal(digital_out, self._times)
                if self.verbose:
                    print('Found ', len(channels_out), ' digital in channels')
            except:
                if self.verbose:
                    print('Found  0  digital in channels')
                self._digital_out = None
                channels_out, states_out, times_out = [], [], []
            self._digital_out = []
            for i, ch in enumerate(channels_out):
                self._digital_out.append(DigitalOut(channels=ch * np.ones(len(times_out[i])),
                                                    channel_states=np.array(states_out[i]),
                                                    times=times_out[i]))
            self._digout_dirty = False
        return self._digital_out

    @property
    def stimulation(self):
        if self._stim_dirty:
            stim_data = self._read_analog(self._stim_chan).T
            stim_parameters = {'stim_step_size': self._global_info['stim_step_size'],
                               'charge_recovery_current_limit': self._global_info['recovery_current_limit'],
                               'charge_recovery_target_voltage': self._global_info['recovery_target_voltage'],
                               'amp_settle_mode': self._global_info['amp_settle_mode'],
                               'charge_recovery_mode': self._global_info['charge_recovery_mode']
                               }
            stim_channels = []
            stim_signals = []
            if np.count_nonzero(stim_data) != 0:
                stim_polarity = np.zeros(stim_data.shape)
                compliance_limit_data_idx = np.where(stim_data >= 2 ** 15)
                stim_data[compliance_limit_data_idx] -= 2 ** 15
                charge_recovery_data_idx = np.where(stim_data >= 2 ** 14)
                stim_data[charge_recovery_data_idx] -= 2 ** 14
                amp_settle_data_idx = np.where(stim_data >= 2 ** 13)
                stim_data[amp_settle_data_idx] -= 2 ** 13

                stim_polarity_idx = np.where(stim_data >= 2 ** 8)
                stim_polarity[stim_polarity_idx] = 1
                stim_data[stim_polarity_idx] -= 2 ** 8
                stim_polarity = 1 - 2 * stim_polarity  # convert(0 = pos, 1 = neg) to + / -1
                stim_data *= stim_polarity
                stim_data = stim_parameters['stim_step_size'] * stim_data / float(1e-6)  # units = microamps

                for ch, stim in enumerate(stim_data):
                    if np.count_nonzero(stim) != 0:
                        stim_channels.append(ch)
                        stim_signals.append(stim)
                # Clear variables
                del stim_polarity, stim_data
            else:
                stim_channels = []
                stim_signals = []
                stim_parameters = {}

            self._stimulation = []
            if len(stim_channels) > 0:
                for i, st in enumerate(stim_channels):
                    self._stimulation.append(Stimulation(
                        stim_channels=st,
                        stim_signal=stim_signals[i],
                        stim_param=stim_parameters,
                        times=self._times
                    ))
            self._stim_dirty = False
        return self._stimulation

    @property
    def times(self):
        return self._times

    def _get_block_info(self):
        if len(self._shape) == 2:
            # this is the general case with 2D
            block_size = self._shape[1]
            block_start = 0
            block_stop = self._size // block_size + 1
            i_start = 0
            i_stop = self._size

            sl0 = i_start % block_size
            sl1 = sl0 + (i_stop - i_start)
        else:
            i_start = 0
            i_stop = self._size
            block_size = None
            block_start = None
            block_stop = None
            sl0 = None
            sl1 = None

        return i_start, i_stop, block_size, block_start, block_stop, sl0, sl1

    def _read_analog(self, chan):
        i_start, i_stop, block_size, block_start, block_stop, sl0, sl1 = self._get_block_info()
        anas = np.zeros((i_stop - i_start, len(chan)), dtype='float32')

        for i, ch in enumerate(chan):
            data_chan = self._raw_data[ch['name']]
            if len(self._shape) == 1:
                anas[:, i] = data_chan[i_start:i_stop] * ch['gain'] + ch['offset']
            else:
                anas[:, i] = data_chan[block_start:block_stop].flatten()[sl0:sl1] * ch['gain'] + ch['offset']

        return anas

    def clip_recording(self, clipping_times, start_end='start'):

        if clipping_times is not None:
            if clipping_times is not list:
                if type(clipping_times[0]) is not pq.quantity.Quantity:
                    print('clipping_times is not a quantity: seconds is used')
                    clipping_times = clipping_times * pq.s

            clipping_times = [t.rescale(pq.s) for t in clipping_times]

            for anas in self.analog_signals:
                clip_anas(anas, clipping_times, start_end)
            for anas in self.analog_in_signals:
                clip_anas(anas, clipping_times, start_end)
            for anas in self.analog_out_signals:
                clip_anas(anas, clipping_times, start_end)
            for ev in self.digital_in_events:
                clip_digital(ev, clipping_times, start_end)
            for ev in self.digital_out_events:
                clip_digital(ev, clipping_times, start_end)
            for stim in self.stimulation:
                clip_stimulation(stim, clipping_times, start_end)

            self._times = clip_times(self._times, clipping_times, start_end)
            self._duration = self._times[-1] - self._times[0]
        else:
            print('Empty clipping times list.')
