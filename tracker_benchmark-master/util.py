import os
import numpy as np

DIFF_NAMES = ('Human4-2', 'Jogging-1', 'Jogging-2',
                  'Skating2-1', 'Skating2-2')

def _convert_name(oldname):
    newname = None
    if oldname == 'Human4-2':
        newname = 'Human4'
    elif oldname == 'Jogging-1':
        newname = 'Jogging.1'
    elif oldname == 'Jogging-2':
        newname = 'Jogging.2'
    elif oldname == 'Skating2-1':
        newname = 'Skating2.1'
    elif oldname == 'Skating2-2':
        newname = 'Skating2.2'

    return newname


def compare_name(resnames, seqnames):
    flag = True
    assert len(resnames) == len(seqnames),\
        'The length between resnames and seqnames is not same!'

    for seqName in seqnames:
        if seqName in DIFF_NAMES:
            seqName = _convert_name(seqName)

        if seqName not in resnames:
            print seqName + ' is not in resnames'
            flag = False
            break
    return flag

def calcu_speed(timetxtdir, seq_name):
    speed = 0.0

    if seq_name in DIFF_NAMES:
        seq_name = _convert_name(seq_name)

    time_file = os.path.join(
        timetxtdir, 'times/%s_time.txt' % seq_name)
    if os.path.isfile(time_file):
        times = np.loadtxt(time_file)
        times = times[times > 0]
        if len(times) > 0:
            speed = np.mean(1. / times)
    return speed

def load_seq_result(boxtxtdir, seq_name):
    if seq_name in DIFF_NAMES:
        seq_name = _convert_name(seq_name)

    record_file = os.path.join(
        boxtxtdir, '%s.txt' % seq_name)
    boxes = np.loadtxt(record_file, delimiter=',')
    return boxes
