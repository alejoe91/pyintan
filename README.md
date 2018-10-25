# pyintan
Python reader for INTAN rhs files

# Installation

Open a terminal and:

```
git clone https://github.com/alejoe91/pyintan/
cd pyintan
python setup.py develop
```

# Basic usage

In python (or ipython):

```
import pyintan

file = pyintan.File('path-to-rhs')

# all fields are lists of objcts

analog_signals = file.analog_signals[0]
adc_signals = file.adc_signals[0]
dac_signals = file.dac_signals[0]
# actual signals are: analog_signals.signal

digital_in_signals = file.digital_in_signals[0]
digital_out_signals = file.digital_out_signals[0]
# event times are: digital_in_signals.times

stimulation = file.stimulation[0]
# contains stimulation information
```
