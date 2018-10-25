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
import os
import os.path as op
import numpy as np
from datetime import datetime
import time
import locale
import platform

from .tools import fread_QString, plural, read_python, get_rising_falling_edges


class Channel:
    def __init__(self, index, name, gain, channel_id):
        self.index = index
        self.id = channel_id
        self.name = name
        self.gain = gain


class ChannelGroup:
    def __init__(self, channel_group_id, filename, channels, attrs):
        self.attrs = attrs
        self.filename = filename
        self.channel_group_id = channel_group_id
        self.channels = channels

    def __str__(self):
        return "<OpenEphys channel_group {}: channel_count: {}>".format(
            self.channel_group_id, len(self.channels)
        )


class AnalogSignal:
    def __init__(self, channel_id, signal, sample_rate):
        self.signal = signal
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<Intan analog signals:shape: {}, sample_rate: {}>".format(
            self.signal.shape, self.sample_rate
        )

class ADCSignal:
    def __init__(self, channel_id, signal, sample_rate):
        self.signal = signal
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<Intan ADC signals:shape: {}, sample_rate: {}>".format(
            self.signal.shape, self.sample_rate
        )

class DACSignal:
    def __init__(self, channel_id, signal, sample_rate):
        self.signal = signal
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<Intan DAC signals:shape: {}, sample_rate: {}>".format(
            self.signal.shape, self.sample_rate
        )

class DigitalSignal:
    def __init__(self, times, channel_id, sample_rate):
        self.times = times
        self.channel_id = channel_id
        self.sample_rate = sample_rate

    def __str__(self):
        return "<Intan digital signal: nchannels: {}>".format(
            self.channel_id
        )

class Stimulation:
    def __init__(self, stim_channels, stim_signal, amp_settle, charge_recovery, compliance_limit, stim_param):
        self.stim_channels = stim_channels
        self.stim_signal = stim_signal
        self.amp_settle = amp_settle
        self.charge_recovery = charge_recovery
        self.compliance_limit = compliance_limit
        self.stim_param = stim_param

    def __str__(self):
        return "<Intan stimulation signals:shape: {}, param: {}>".format(
            self.stim_param.shape, self.stim_param
        )

class File:
    """
    Class for reading experimental data from an OpenEphys dataset.
    """
    def __init__(self, filename, probefile=None, save_binary=True, no_load=False):
        self._absolute_filename = op.abspath(filename)
        self._fname = op.split(filename)[-1]
        self._absolute_foldername = op.split(filename)[0]
        self._channel_info = dict()
        self._channel_groups_dirty = True

        fname_root = op.split(filename)[-1][:-4]
        filenames = [f for f in os.listdir(self._absolute_foldername)]
        if any(fname_root + '.dat' in f for f in filenames) and not no_load:
            load_binary = True
        else:
            load_binary = False

        data = self.load(filename, load_binary)

        # extract date and time, automatically appended at the end of filename
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
        self._start_datetime = datetime.strptime(under_date[0]+under_date[1], '%y%m%d%H%M%S')
        self._sample_rate = data['frequency_parameters']['amplifier_sample_rate']

        # apply probe channel mapping
        # TODO multiple ports -> append channel index
        recorded_channels = []
        newchan = 0
        for port in np.arange(len(data['amplifier_channels'])):
            # for ch in data['amplifier_channels'][port]['chip_channel']:
            recorded_channels.append(newchan)
            newchan += 1

        print('Recorded channels: ', recorded_channels)
        self._channel_info['channels'] = recorded_channels
        if probefile is not None:
            self._probefile_ch_mapping = read_python(probefile)['channel_groups']
            for group_idx, group in self._probefile_ch_mapping.items():
                group['gain'] = []
                # prb file channels are sequential, 'channels' are not as they depend on FPGA channel selection -> Collapse them into array
                for chan, intan_chan in zip(group['channels'],
                                         group['intan_channels']):
                    if intan_chan not in recorded_channels:
                        raise ValueError('Channel "' + str(intan_chan) +
                                         '" in channel group "' +
                                         str(group_idx) + '" in probefile' +
                                         probefile +
                                         ' is not marked as recorded ' +
                                         'in intan file' +
                                         self._fname)
                    if recorded_channels.index(intan_chan) != chan:
                        print(intan_chan)
                        print(recorded_channels.index(intan_chan))
                        print(chan)
                        raise ValueError('Channel mapping does not match ' +
                                         'sequence of recorded channels')
                    group['gain'].append(
                        self._channel_info['gain'][str(intan_chan)]
                    )
            self._keep_channels = [chan for group in
                                   self._probefile_ch_mapping.values()
                                   for chan in group['channels']]
            print('Number of selected channels: ', len(self._keep_channels))
        else:
            self._keep_channels = None  # HACK
            # TODO sequential channel mapping
            print('sequential channel mapping')

        if load_binary and not no_load:
            # load amp.dat
            numchan = int(len(recorded_channels))
            fdat = fname_root + '.dat'
            with open(op.join(self._absolute_foldername, fdat), "rb") as fh:
                nsamples = os.fstat(fh.fileno()).st_size // (numchan * 4)
                print('Estimated samples: ', int(nsamples), ' Numchan: ', numchan)
                amp_data = np.memmap(fh, np.dtype('f4'), mode='r',
                                    shape=(numchan, nsamples))
            # generate objects
            self._analog_signals = [AnalogSignal(
                signal=amp_data,
                channel_id=np.array(recorded_channels),
                sample_rate=self.sample_rate
            )]
            # load binary_data.npz
            fbin = 'binary_data.npz'
            npzfile = np.load(op.join(self._absolute_foldername, fbin))
            self._digital_in_signals = npzfile['dig_in']
            self._digital_out_signals =  npzfile['dig_out']
            self._adc_signals =  npzfile['adc']
            self._dac_signals =  npzfile['dac']
            self._stimulation =  npzfile['stim']
            self._times = npzfile['t']*pq.s

        else:
            self._digital_in_signals = [DigitalSignal(
                times=data['board_dig_in_data'],
                channel_id=np.array([data['board_dig_in_channels'][ch]['chip_channel']
                                     for ch in np.arange(len(data['board_dig_in_channels']))]),
                sample_rate=self.sample_rate
            )]

            self._digital_out_signals = [DigitalSignal(
                times=data['board_dig_out_data'],
                channel_id=np.array([data['board_dig_out_channels'][ch]['chip_channel']
                                     for ch in np.arange(len(data['board_dig_out_channels']))]),
                sample_rate=self.sample_rate
            )]

            self._adc_signals = [ADCSignal(
                signal=data['board_adc_data'],
                channel_id=np.array([data['board_adc_channels'][ch]['chip_channel']
                                     for ch in np.arange(len(data['board_adc_channels']))]),
                sample_rate=self.sample_rate
            )]

            self._dac_signals = [DACSignal(
                signal=data['board_dac_data'],
                channel_id=np.array([data['board_dac_channels'][ch]['chip_channel']
                                     for ch in np.arange(len(data['board_dac_channels']))]),
                sample_rate=self.sample_rate
            )]

            self._stimulation = [Stimulation(
                stim_channels=data['stim_channels'],
                stim_signal=data['stim_signal'],
                amp_settle=data['amp_settle_data'],
                charge_recovery=data['charge_recovery_data'],
                compliance_limit=data['compliance_limit_data'],
                stim_param=data['stim_parameters']
            )]

            self._times = data['t']

            # save binary and npy for future loadings
            if save_binary and not load_binary:
                # save amp.dat
                fdat = op.join(self._absolute_foldername, fname_root + '.dat')
                print('Saving ', fdat)
                with open(fdat, 'wb') as f:
                    np.array(data['amplifier_data'], dtype='float32').tofile(f)

                # save binary_data.npz
                fbin = op.join(self._absolute_foldername, 'binary_data')
                print('Saving ', fbin)
                np.savez(fbin, dig_in=self._digital_in_signals, dig_out=self._digital_out_signals,
                         adc=self._adc_signals, dac=self._dac_signals, stim=self._stimulation, t=self._times)

            # memmap AnalogSignals to .dat to reduce memory usage
            numchan = int(len(recorded_channels))
            fdat = fname_root + '.dat'
            with open(op.join(self._absolute_foldername, fdat), "rb") as fh:
                nsamples = os.fstat(fh.fileno()).st_size // (numchan * 4)
                print('Estimated samples: ', int(nsamples), ' Numchan: ', numchan)
                amp_data = np.memmap(fh, np.dtype('f4'), mode='r',
                                     shape=(numchan, nsamples))
            # generate objects
            self._analog_signals = [AnalogSignal(
                signal=amp_data,
                channel_id=np.array(recorded_channels),
                sample_rate=self.sample_rate
            )]


        self._duration = self._times[-1] - self._times[0]

        # Clear data
        del data


    @property
    def session(self):
        return os.path.split(self._absolute_filename)[-1][:-4]

    @property
    def datetime(self):
        return self._start_datetime

    @property
    def duration(self):
        return self._duration

    @property
    def sample_rate(self):
            return self._sample_rate

    def channel_group(self, channel_id):
        if self._channel_groups_dirty:
            self._read_channel_groups()

        return self._channel_id_to_channel_group[channel_id]

    @property
    def channel_groups(self):
        if self._channel_groups_dirty:
            self._read_channel_groups()

        return self._channel_groups

    @property
    def analog_signals(self):
        return self._analog_signals

    @property
    def adc_signals(self):
        return self._adc_signals

    @property
    def dac_signals(self):
        return self._dac_signals

    @property
    def digital_in_signals(self):
        return self._digital_in_signals

    @property
    def digital_out_signals(self):
        return self._digital_out_signals

    @property
    def stimulation(self):
        return self._stimulation

    @property
    def times(self):
        return self._times


    def _read_channel_groups(self):
        self._channel_id_to_channel_group = {}
        self._channel_group_id_to_channel_group = {}
        self._channel_count = 0
        self._channel_groups = []
        for channel_group_id, channel_info in self._probefile_ch_mapping.items():
            num_chans = len(channel_info['channels'])
            self._channel_count += num_chans
            channels = []
            for idx, chan in enumerate(channel_info['channels']):
                channel = Channel(
                    index=idx,
                    channel_id=chan,
                    name="channel_{}_channel_group_{}".format(chan,
                                                              channel_group_id),
                    gain=channel_info['gain'][idx]
                )
                channels.append(channel)

            channel_group = ChannelGroup(
                channel_group_id=channel_group_id,
                filename=None,#TODO,
                channels=channels,
                attrs=None #TODO
            )
            ana = self.analog_signals[0]
            analog_signals = []
            for channel in channels:
                analog_signals.append(AnalogSignal(signal=ana.signal[channel.id],
                                                   channel_id=channel.id,
                                                   sample_rate=ana.sample_rate))

            channel_group.analog_signals = analog_signals

            self._channel_groups.append(channel_group)
            self._channel_group_id_to_channel_group[channel_group_id] = channel_group

            for chan in channel_info['channels']:
                self._channel_id_to_channel_group[chan] = channel_group

        # TODO channel mapping to file
        self._channel_ids = np.arange(self._channel_count)
        self._channel_groups_dirty = False


    # def clip_recording(self, clipping_times, start_end='start'):
    #
    #     if clipping_times is not None:
    #         if clipping_times is not list:
    #             if type(clipping_times[0]) is not pq.quantity.Quantity:
    #                 raise AttributeError('clipping_times must be a quantity list of length 1 or 2')
    #
    #         clipping_times = [t.rescale(pq.s) for t in clipping_times]
    #
    #         for anas in self.analog_signals:
    #             anas.signal = clip_anas(anas, self.times, clipping_times, start_end)
    #         for anas in self.adc_signals:
    #             anas.signal = clip_anas(anas, self.times, clipping_times, start_end)
    #         for anas in self.dac_signals:
    #             anas.signal = clip_anas(anas, self.times, clipping_times, start_end)
    #         for digs in self.digital_in_signals:
    #             digs.times = clip_digs(digs, clipping_times, start_end)
    #         for digs in self.digital_out_signals:
    #             digs.times = clip_digs(digs, clipping_times, start_end)
    #         for stim in self.stimulation:
    #             stim.stim_signal = clip_stimulation(stim, self.times, clipping_times, start_end)
    #
    #         self._times = clip_times(self._times, clipping_times, start_end)
    #         self._duration = self._times[-1] - self._times[0]
    #     else:
    #         print('Empty clipping times list.')


    def load(self, filepath, load_binary):
        # redirects to code for individual file types
        if 'rhs' in filepath:
            data = self.loadRHS(filepath, load_binary)
        elif 'rhd' in filepath:
            print('to be implemented soon')
            data = []
        else:
            raise Exception("Not a recognized file type. Please input a .continuous, .spikes, or .events file")

        return data


    def loadRHS(self, filepath, load_binary):
        t1 = time.time()
        data = dict()
        print('Loading intan data')

        f = open(filepath, 'rb')
        filesize = os.fstat(f.fileno()).st_size - f.tell()

        # Check 'magic number' at beginning of file to make sure this is an Intan
        # Technologies RHS2000 data file.
        magic_number = np.fromfile(f, np.dtype('u4'), 1)
        if magic_number != int('d69127ac', 16):
            raise IOError('Unrecognized file type.')

        # Read version number.
        data_file_main_version_number = np.fromfile(f, 'i2', 1)[0]
        data_file_secondary_version_number = np.fromfile(f, 'i2', 1)[0]

        print('Reading Intan Technologies RHS2000 Data File, Version ', data_file_main_version_number, \
            data_file_secondary_version_number)

        num_samples_per_data_block = 128

        # Read information of sampling rate and amplifier frequency settings.
        sample_rate = np.fromfile(f, 'f4', 1)[0]
        dsp_enabled = np.fromfile(f, 'i2', 1)[0]
        actual_dsp_cutoff_frequency = np.fromfile(f, 'f4', 1)[0]
        actual_lower_bandwidth = np.fromfile(f, 'f4', 1)[0]
        actual_lower_settle_bandwidth = np.fromfile(f, 'f4', 1)[0]
        actual_upper_bandwidth = np.fromfile(f, 'f4', 1)[0]

        desired_dsp_cutoff_frequency = np.fromfile(f, 'f4', 1)[0]
        desired_lower_bandwidth = np.fromfile(f, 'f4', 1)[0]
        desired_lower_settle_bandwidth = np.fromfile(f, 'f4', 1)[0]
        desired_upper_bandwidth = np.fromfile(f, 'f4', 1)[0]

        # This tells us if a software 50/60 Hz notch filter was enabled during the data acquistion
        notch_filter_mode = np.fromfile(f, 'i2', 1)[0]
        notch_filter_frequency = 0
        if notch_filter_mode == 1:
            notch_filter_frequency = 50
        elif notch_filter_mode == 2:
            notch_filter_frequency = 60

        desired_impedance_test_frequency = np.fromfile(f, 'f4', 1)[0]
        actual_impedance_test_frequency = np.fromfile(f, 'f4', 1)[0]

        amp_settle_mode = np.fromfile(f, 'i2', 1)[0]
        charge_recovery_mode = np.fromfile(f, 'i2', 1)[0]

        stim_step_size = np.fromfile(f, 'f4', 1)[0]
        charge_recovery_current_limit = np.fromfile(f, 'f4', 1)[0]
        charge_recovery_target_voltage = np.fromfile(f, 'f4', 1)[0]

        # Place notes in data structure
        notes = {'note1': fread_QString(f),
                 'note2': fread_QString(f),
                 'note3': fread_QString(f)}

        # See if dc amplifier was saved
        dc_amp_data_saved = np.fromfile(f, 'i2', 1)[0]

        # Load eval board mode
        eval_board_mode = np.fromfile(f, 'i2', 1)[0]

        reference_channel = fread_QString(f)

        # Place frequency-related information in data structure.
        frequency_parameters = {
        'amplifier_sample_rate': sample_rate * pq.Hz,
        'board_adc_sample_rate': sample_rate * pq.Hz,
        'board_dig_in_sample_rate': sample_rate * pq.Hz,
        'desired_dsp_cutoff_frequency': desired_dsp_cutoff_frequency,
        'actual_dsp_cutoff_frequency': actual_dsp_cutoff_frequency,
        'dsp_enabled': dsp_enabled,
        'desired_lower_bandwidth': desired_lower_bandwidth,
        'desired_lower_settle_bandwidth': desired_lower_settle_bandwidth,
        'actual_lower_bandwidth': actual_lower_bandwidth,
        'actual_lower_settle_bandwidth': actual_lower_settle_bandwidth,
        'desired_upper_bandwidth': desired_upper_bandwidth,
        'actual_upper_bandwidth': actual_upper_bandwidth,
        'notch_filter_frequency': notch_filter_frequency,
        'desired_impedance_test_frequency': desired_impedance_test_frequency,
        'actual_impedance_test_frequency': actual_impedance_test_frequency}

        stim_parameters = {
        'stim_step_size': stim_step_size,
        'charge_recovery_current_limit': charge_recovery_current_limit,
        'charge_recovery_target_voltage': charge_recovery_target_voltage,
        'amp_settle_mode': amp_settle_mode,
        'charge_recovery_mode': charge_recovery_mode}

        # Define data structure for spike trigger settings.
        spike_trigger_struct = {
        'voltage_trigger_mode': {},
        'voltage_threshold': {},
        'digital_trigger_channel': {},
        'digital_edge_polarity': {} }


        spike_triggers = []

        # Define data structure for data channels.
        channel_struct = {
        'native_channel_name': {},
        'custom_channel_name': {},
        'native_order': {},
        'custom_order': {},
        'board_stream': {},
        'chip_channel': {},
        'port_name': {},
        'port_prefix': {},
        'port_number': {},
        'electrode_impedance_magnitude': {},
        'electrode_impedance_phase': {} }

        # Create structure arrays for each type of data channel.
        amplifier_channels = []
        board_adc_channels = []
        board_dac_channels = []
        board_dig_in_channels = []
        board_dig_out_channels = []

        amplifier_index = 0
        board_adc_index = 0
        board_dac_index = 0
        board_dig_in_index = 0
        board_dig_out_index = 0

        # Read signal summary from data file header.

        number_of_signal_groups = np.fromfile(f, 'i2', 1)[0]
        print('Signal groups: ', number_of_signal_groups)

        for signal_group in range(number_of_signal_groups):
            signal_group_name = fread_QString(f)
            signal_group_prefix = fread_QString(f)
            signal_group_enabled = np.fromfile(f, 'i2', 1)[0]
            signal_group_num_channels = np.fromfile(f, 'i2', 1)[0]
            signal_group_num_amp_channels = np.fromfile(f, 'i2', 1)[0]

            if signal_group_num_channels > 0 and signal_group_enabled > 0:
                new_channel = {}
                new_trigger_channel = {}

                new_channel['port_name'] = signal_group_name
                new_channel['port_prefix'] = signal_group_prefix
                new_channel['port_number'] = signal_group
                for signal_channel in range(signal_group_num_channels):
                    new_channel['native_channel_name'] = fread_QString(f)
                    new_channel['custom_channel_name'] = fread_QString(f)
                    new_channel['native_order'] = np.fromfile(f, 'i2', 1)[0]
                    new_channel['custom_order'] = np.fromfile(f, 'i2', 1)[0]
                    signal_type = np.fromfile(f, 'i2', 1)[0]
                    channel_enabled = np.fromfile(f, 'i2', 1)[0]
                    new_channel['chip_channel'] = np.fromfile(f, 'i2', 1)[0]
                    skip = np.fromfile(f, 'i2', 1)[0] # ignore command_stream
                    new_channel['board_stream'] = np.fromfile(f, 'i2', 1)[0]
                    new_trigger_channel['voltage_trigger_mode'] = np.fromfile(f, 'i2', 1)[0]
                    new_trigger_channel['voltage_threshold'] = np.fromfile(f, 'i2', 1)[0]
                    new_trigger_channel['digital_trigger_channel'] = np.fromfile(f, 'i2', 1)[0]
                    new_trigger_channel['digital_edge_polarity'] = np.fromfile(f, 'i2', 1)[0]
                    new_channel['electrode_impedance_magnitude'] = np.fromfile(f, 'f4', 1)[0]
                    new_channel['electrode_impedance_phase'] = np.fromfile(f, 'f4', 1)[0]

                    if channel_enabled:
                        if signal_type == 0:
                            ch = new_channel.copy()
                            amplifier_channels.append(ch)
                            spike_triggers.append(new_trigger_channel)
                            amplifier_index = amplifier_index + 1
                        elif signal_type == 1:
                            # aux inputs not used in RHS2000 system
                            pass
                        elif signal_type == 2:
                            # supply voltage not used in RHS2000 system
                            pass
                        elif signal_type == 3:
                            ch = new_channel.copy()
                            board_adc_channels.append(ch)
                            board_adc_index = board_adc_index + 1
                        elif signal_type == 4:
                            ch = new_channel.copy()
                            board_dac_channels.append(ch)
                            board_dac_index = board_dac_index + 1
                        elif signal_type == 5:
                            ch = new_channel.copy()
                            board_dig_in_channels.append(ch)
                            board_dig_in_index = board_dig_in_index + 1
                        elif signal_type == 6:
                            ch = new_channel.copy()
                            board_dig_out_channels.append(ch)
                            board_dig_out_index = board_dig_out_index + 1
                        else:
                            raise Exception('Unknown channel type')


        # Summarize contents of data file.
        num_amplifier_channels = amplifier_index
        num_board_adc_channels = board_adc_index
        num_board_dac_channels = board_dac_index
        num_board_dig_in_channels = board_dig_in_index
        num_board_dig_out_channels = board_dig_out_index

        print('Found ', num_amplifier_channels, ' amplifier channel' , plural(num_amplifier_channels))
        if dc_amp_data_saved != 0:
            print('Found ', num_amplifier_channels, 'DC amplifier channel' , plural(num_amplifier_channels))
        print('Found ', num_board_adc_channels, ' board ADC channel' , plural(num_board_adc_channels))
        print('Found ', num_board_dac_channels, ' board DAC channel' , plural(num_board_adc_channels))
        print('Found ', num_board_dig_in_channels, ' board digital input channel' , plural(num_board_dig_in_channels))
        print('Found ', num_board_dig_out_channels, ' board digital output channel' , plural(num_board_dig_out_channels))

        # Determine how many samples the data file contains.

        # Each data block contains num_samplesper_data_block amplifier samples
        bytes_per_block = num_samples_per_data_block * 4  # timestamp data
        if dc_amp_data_saved != 0:
            bytes_per_block += num_samples_per_data_block * (2 + 2 + 2) * num_amplifier_channels
        else:
            bytes_per_block += num_samples_per_data_block * (2 + 2) * num_amplifier_channels
        # Board analog inputs are sampled at same rate as amplifiers
        bytes_per_block += num_samples_per_data_block * 2 * num_board_adc_channels
        # Board analog outputs are sampled at same rate as amplifiers
        bytes_per_block += num_samples_per_data_block * 2 * num_board_dac_channels
        # Board digital inputs are sampled at same rate as amplifiers
        if num_board_dig_in_channels > 0:
            bytes_per_block += num_samples_per_data_block * 2
        # Board digital outputs are sampled at same rate as amplifiers
        if num_board_dig_out_channels > 0:
            bytes_per_block += num_samples_per_data_block * 2

        # How many data blocks remain in this file?
        data_present = 0
        bytes_remaining = filesize - f.tell()
        if bytes_remaining > 0:
            data_present = 1


        num_data_blocks = int(bytes_remaining / bytes_per_block)

        num_amplifier_samples = num_samples_per_data_block * num_data_blocks
        num_board_adc_samples = num_samples_per_data_block * num_data_blocks
        num_board_dac_samples = num_samples_per_data_block * num_data_blocks
        num_board_dig_in_samples = num_samples_per_data_block * num_data_blocks
        num_board_dig_out_samples = num_samples_per_data_block * num_data_blocks

        record_time = num_amplifier_samples / sample_rate

        if data_present:
            print('File contains ', record_time, ' seconds of data.  '
                                                 'Amplifiers were sampled at ', sample_rate / 1000 , ' kS/s.')
        else:
            print('Header file contains no data.  Amplifiers were sampled at ', sample_rate / 1000 ,  'kS/s.')

        if data_present:

            anas_gain = 0.195
            anas_offset = 32768
            dc_gain = -0.01923
            dc_offset = 512

            self._channel_info['gain'] = {}
            for ch in np.arange(num_amplifier_channels):
                self._channel_info['gain'][str(ch)] = anas_gain

            if not load_binary:
                # Pre-allocate memory for data.
                print('Allocating memory for data')

                t = np.zeros(num_amplifier_samples)

                amplifier_data = np.zeros((num_amplifier_channels, num_amplifier_samples))
                if dc_amp_data_saved != 0:
                    dc_amplifier_data = np.zeros((num_amplifier_channels, num_amplifier_samples))

                stim_data = np.zeros((num_amplifier_channels, num_amplifier_samples))
                board_adc_data = np.zeros((num_board_adc_channels, num_board_adc_samples))
                board_dac_data = np.zeros((num_board_dac_channels, num_board_dac_samples))
                board_dig_in_raw = np.zeros(num_board_dig_in_samples)
                board_dig_out_raw = np.zeros(num_board_dig_out_samples)

                # Read sampled data from file.
                print('Reading data from file')

                amplifier_index = 0
                board_adc_index = 0
                board_dac_index = 0
                board_dig_in_index = 0
                board_dig_out_index = 0

                print_increment = 10
                percent_done = print_increment

                print('num_data_blocks: ', num_data_blocks)

                for i in range(num_data_blocks):
                    t[amplifier_index:(amplifier_index + num_samples_per_data_block)] = \
                        np.fromfile(f, 'i4', num_samples_per_data_block)
                    if num_amplifier_channels > 0:
                        amplifier_data[:, amplifier_index:(amplifier_index + num_samples_per_data_block)] = \
                            np.reshape(np.fromfile(f, 'u2', num_samples_per_data_block*num_amplifier_channels),
                                        (num_amplifier_channels, num_samples_per_data_block))
                        if dc_amp_data_saved != 0:
                            dc_amplifier_data[:, amplifier_index:(amplifier_index + num_samples_per_data_block)] = \
                                np.reshape(np.fromfile(f, 'u2', num_samples_per_data_block * num_amplifier_channels),
                                           (num_amplifier_channels, num_samples_per_data_block))
                        stim_data[:, amplifier_index:(amplifier_index + num_samples_per_data_block)] = \
                            np.reshape(np.fromfile(f, 'u2', num_samples_per_data_block * num_amplifier_channels),
                                       (num_amplifier_channels, num_samples_per_data_block))

                    if num_board_adc_channels > 0:
                        board_adc_data[:, board_adc_index:(board_adc_index + num_samples_per_data_block)] = \
                            np.reshape(np.fromfile(f, 'u2', num_samples_per_data_block*num_board_adc_channels),
                                        (num_board_adc_channels, num_samples_per_data_block))
                    if num_board_dac_channels > 0:
                        board_dac_data[:, board_dac_index:(board_dac_index + num_samples_per_data_block)] = \
                            np.reshape(np.fromfile(f, 'u2', num_samples_per_data_block*num_board_dac_channels),
                                        (num_board_dac_channels, num_samples_per_data_block))
                    if num_board_dig_in_channels > 0:
                        board_dig_in_raw[board_dig_in_index:(board_dig_in_index + num_samples_per_data_block)] = \
                        np.fromfile(f, 'u2', num_samples_per_data_block)
                    if num_board_dig_out_channels > 0:
                        board_dig_out_raw[board_dig_out_index:(board_dig_out_index + num_samples_per_data_block)] = \
                        np.fromfile(f, 'u2', num_samples_per_data_block)

                    amplifier_index += num_samples_per_data_block
                    board_adc_index += num_samples_per_data_block
                    board_dac_index += num_samples_per_data_block
                    board_dig_in_index += num_samples_per_data_block
                    board_dig_out_index += num_samples_per_data_block

                    fraction_done = 100 * float((i+1) / float(num_data_blocks))
                    if fraction_done >= percent_done:
                        print(percent_done, '% done')
                        percent_done += print_increment

                # Make sure we have read exactly the right amount of data.
                bytes_remaining = filesize - f.tell()
                if bytes_remaining != 0:
                    # raise Error('Error: End of file not reached.')
                    pass

                # Close data file.
                f.close()

                t2 = time.time()
                print('Loading done. time: ', t2 - t1)

                if data_present:

                    print('Parsing data')

                    # Check for gaps in timestamps.
                    num_gaps = len(np.where(np.diff(t) != 1)[0])
                    if num_gaps == 0:
                        print('No missing timestamps in data.')
                    else:
                        print('Warning: ', num_gaps, ' gaps in timestamp data found.  Time scale will not be uniform!')
                    # Scale time steps (units = seconds).
                    t = t / frequency_parameters['amplifier_sample_rate']

                    # # Extract digital input channels times separate variables.
                    if np.count_nonzero(board_dig_in_raw) != 0:
                        board_dig_in_data = []
                        for i in range(num_board_dig_in_channels):
                            # find idx of high level
                            mask = 2**board_dig_in_channels[i]['native_order']*np.ones(len(board_dig_in_raw))
                            idx_high = np.where(np.bitwise_and(board_dig_in_raw.astype(dtype='int'),
                                                               mask.astype(dtype='int')) > 0)
                            rising, falling = get_rising_falling_edges(idx_high)
                            board_dig_in_data.append(t[rising])
                        board_dig_in_data = np.array(board_dig_in_data)
                    else:
                        print('No digital input data')
                        board_dig_in_data = np.array([])

                    if np.count_nonzero(board_dig_out_raw) != 0:
                        board_dig_out_data = []
                        for i in range(num_board_dig_out_channels):
                            # find idx of high level
                            mask = 2 ** board_dig_out_channels[i]['native_order'] * np.ones(len(board_dig_out_raw))
                            print(mask.shape, len(board_dig_out_data))
                            idx_high = np.where(np.bitwise_and(board_dig_out_raw.astype(dtype='int'),
                                                               mask.astype(dtype='int')) > 0)
                            rising, falling = get_rising_falling_edges(idx_high)
                            board_dig_out_data.append(t[rising])
                        board_dig_out_data = np.array(board_dig_out_data)
                    else:
                        print('No digital output data')
                        board_dig_out_data = np.array([])

                    # Clear variables
                    del board_dig_out_raw
                    del board_dig_in_raw

                    #TODO optimize memory-wise: e.g. only save time and chan of compliance, ampsett, charge recovery as list

                    # Scale voltage levels appropriately.
                    amplifier_data -= anas_offset  # units = microvolts
                    amplifier_data *= anas_gain  # units = microvolts
                    if dc_amp_data_saved != 0:
                        dc_amplifier_data -= dc_offset  # units = volts
                        dc_amplifier_data *= dc_gain  # units = volts


                    if np.count_nonzero(stim_data) != 0:
                        # TODO only save stim channel and respective signals waveform
                        stim_polarity = np.zeros((num_amplifier_channels, num_amplifier_samples))

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

                        stim_channels = []
                        stim_signal = []

                        for ch, stim in enumerate(stim_data):
                            if np.count_nonzero(stim) != 0:
                                stim_channels.append(ch)
                                stim_signal.append(stim)
                        stim_channels = np.array(stim_channels)
                        stim_signal = np.array(stim_signal)

                        # Clear variables
                        del stim_polarity, stim_data

                        amp_settle_data = []
                        charge_recovery_data = []
                        compliance_limit_data = []

                        for chan in np.arange(num_amplifier_channels):
                            if len(np.where(amp_settle_data_idx[0] == chan)[0]) != 0:
                                amp_settle_data.append(t[amp_settle_data_idx[1][np.where(amp_settle_data_idx[0] == chan)[0]]])
                            else:
                                amp_settle_data.append([])
                            if len(np.where(charge_recovery_data_idx[0] == chan)[0]) != 0:
                                charge_recovery_data.append(
                                    t[charge_recovery_data_idx[1][np.where(charge_recovery_data_idx[0] == chan)[0]]])
                            else:
                                charge_recovery_data.append([])
                            if len(np.where(compliance_limit_data_idx[0] == chan)[0]) != 0:
                                compliance_limit_data.append(
                                    t[compliance_limit_data_idx[1][np.where(compliance_limit_data_idx[0] == chan)[0]]])
                            else:
                                compliance_limit_data.append([])

                        amp_settle_data = np.array(amp_settle_data)
                        charge_recovery_data = np.array(charge_recovery_data)
                        compliance_limit_data = np.array(compliance_limit_data)
                    else:
                        print('No stimulation data')
                        stim_channels = np.array([])
                        stim_signal = np.array([])
                        amp_settle_data = np.array([])
                        charge_recovery_data = np.array([])
                        compliance_limit_data = np.array([])

                    if np.count_nonzero(board_adc_data) != 0:
                        board_adc_data -= 32768  # units = volts
                        board_adc_data *= 312.5e-6  # units = volts
                    else:
                        del board_adc_data
                        print('No ADC data')
                        board_adc_data = np.array([])

                    if np.count_nonzero(board_dac_data) != 0:
                        board_dac_data -= 32768  # units = volts
                        board_dac_data *= 312.5e-6  # units = volts
                    else:
                        del board_dac_data
                        print('No DAC data')
                        board_dac_data = np.array([])

                    t3 = time.time()
                    print('Parsing done. time: ', t3 - t2)

                # Create data dictionary
                print('Creating data structure...')
                data['notes'] = notes
                data['frequency_parameters'] = frequency_parameters
                data['stim_parameters'] =  stim_parameters
                if data_file_main_version_number > 1:
                    data['reference_channel'] = reference_channel

                if num_amplifier_channels > 0:
                    data['amplifier_channels'] = amplifier_channels
                    if data_present:
                        data['amplifier_data'] = amplifier_data
                        if dc_amp_data_saved != 0:
                            data['dc_amplifier_data'] = dc_amplifier_data

                        data['stim_channels'] = stim_channels
                        data['stim_signal'] = stim_signal
                        data['amp_settle_data'] = amp_settle_data
                        data['charge_recovery_data'] = charge_recovery_data
                        data['compliance_limit_data'] = compliance_limit_data
                        data['t'] = t

                    data['spike_triggers'] = spike_triggers

                if num_board_adc_channels > 0:
                    data['board_adc_channels'] = board_adc_channels
                    if data_present:
                        data['board_adc_data'] = board_adc_data
                else:
                    data['board_adc_data'] = np.array([])
                    data['board_adc_channels'] = np.array([])

                if num_board_dac_channels > 0:
                    data['board_dac_channels'] = board_dac_channels
                    if data_present:
                        data['board_dac_data'] = board_dac_data
                else:
                    data['board_dac_data'] = np.array([])
                    data['board_dac_channels'] = np.array([])

                if num_board_dig_in_channels > 0:
                    data['board_dig_in_channels'] = board_dig_in_channels
                    if data_present:
                        data['board_dig_in_data'] = board_dig_in_data
                else:
                    data['board_dig_in_data'] = np.array([])
                    data['board_dig_in_channels'] = np.array([])


                if num_board_dig_out_channels > 0:
                    data['board_dig_out_channels'] = board_dig_out_channels
                    if data_present:
                        data['board_dig_out_data'] = board_dig_out_data
                else:
                    data['board_dig_out_data'] = np.array([])
                    data['board_dig_out_channels'] = np.array([])


                if data_present:
                    print('Extracted data are now available in the python workspace.')
                else:
                    print('Extracted waveform information is now available in the python workspace.')
            else:
                # Create data dictionary
                print('Creating data structure...')
                data['notes'] = notes
                data['frequency_parameters'] = frequency_parameters
                data['stim_parameters'] = stim_parameters
                if data_file_main_version_number > 1:
                    data['reference_channel'] = reference_channel
                if num_amplifier_channels > 0:
                    data['amplifier_channels'] = amplifier_channels
                    data['spike_triggers'] = spike_triggers

                if num_board_adc_channels > 0:
                    data['board_adc_channels'] = board_adc_channels
                else:
                    data['board_adc_channels'] = np.array([])

                if num_board_dac_channels > 0:
                    data['board_dac_channels'] = board_dac_channels
                else:
                    data['board_dac_channels'] = np.array([])

                if num_board_dig_in_channels > 0:
                    data['board_dig_in_channels'] = board_dig_in_channels
                else:
                    data['board_dig_in_channels'] = np.array([])

                if num_board_dig_out_channels > 0:
                    data['board_dig_out_channels'] = board_dig_out_channels
                else:
                    data['board_dig_out_channels'] = np.array([])

                if data_present:
                    print('Extracted data are now available in the python workspace.')
                else:
                    print('Extracted waveform information is now available in the python workspace.')

        return data
