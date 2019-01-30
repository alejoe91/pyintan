from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import numpy as np
import os.path as op
from six import exec_

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

def read_python(path):
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata

def get_rising_falling_edges(idx_high):
    '''
    :param idx_high: indeces where dig signal is '1'
    :return: rising and falling indices lists
    '''
    rising = []
    falling = []

    idx_high = idx_high[0]

    if len(idx_high) != 0:
        for i, idx in enumerate(idx_high[:-1]):
            if i==0:
                # first idx is rising
                rising.append(idx)
            else:
                if idx_high[i+1] != idx + 1:
                    falling.append(idx)
                if idx - 1 != idx_high[i-1]:
                    rising.append(idx)

    return rising, falling