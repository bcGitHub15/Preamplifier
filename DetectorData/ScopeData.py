#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:54:10 2026

Routines to work with detector traces from the SigLint scope.

@author: bcollett
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

# A DataFilter provides a test method that
# is passed a trace and returns True or False
# depending on whether the trace meets a condition.
# The actual conditions are implemented by sub-classes.
# The base class is pretty abstract
class DataFilter:
    def __init__(self):
        self.type = 'None'

    def test(self, trace):
        if not isinstance(trace, np.ndarray):
            raise ValueError('Filter must be passed a trace.')
        return self._test(trace)
    
    # Internal method is overridden by sub-class.
    def _test(self, trace):
        return False

# Min filter is initialized with a minimum value and tests
# whether any point in the trace falls below
class MinFilter(DataFilter):
    def __init__(self, min):
        self.type = 'Minimum'
        self.min = float(min)
    
    def _test(self, trace):
        return np.min(trace) <= self.min

class DetData:
    '''
    def __init__(self, filename):
        self.data = np.load(filename)
        self.times = np.linspace(0, 1.0e-5, num=2000, endpoint = False)
        self.ntrace = self.data.shape[0]
    '''
    def __init__(self, array):
        # Check that array makes sense.
        if not isinstance(array, np.ndarray):
            raise ValueError('DetData expects a numpy array')
        a_shape = array.shape
        if len(a_shape) != 2:
            raise ValueError(f'DetData expects a 2-D array, found shape {a_shape}')
        n_val = a_shape[1]
        if n_val != 2000:
            raise ValueError(f'DetData expects 2nd dim to be 2000, found {n_val}')
        # Build ourselves
        self.data = array
        self.times = np.linspace(0, 1.0e-5, num=2000, endpoint = False)
        self.ntrace = a_shape[0]

    def plot_one_on(self, ax, trace):
        ax.plot(self.times, self.data[trace,:])
    
    def plot_one(self, trace):
        fig = plt.figure()
        ax = fig.add_subplot()
        self.plot_on(ax, trace)

    def plot_many_on(self, ax, first=0, last=None):
        if last is None:
            last = self.ntrace
        for trace in range(first, last):
            ax.plot(self.times, self.data[trace,:])
    
    def plot_many(self, first=0, last=None):
        fig = plt.figure()
        ax = fig.add_subplot()
        self.plot_many_on(ax, first, last)

    # compute size of noise from first 4 us of one trace
    def find_noise(self, num):
        return np.max(np.abs(self.data[num, :800]))
    
    # statistics on noise from first 4 us of all traces
    def find_avg_noise(self):
        sum = 0
        sumsq = 0
        for i in range(self.ntrace):
            n = self.find_noise(i)
            sum += n
            sumsq += n * n
        avg = sum / self.ntrace
        avsq = sumsq / self.ntrace
        std = np.sqrt(avsq - avg * avg)
        return (avg, std)

    # Return subset of traces that pass a given filter
    def filter(self, theFilter):
        if not isinstance(theFilter, DataFilter):
            raise ValueError(f'Filter expected a data filter but found {type(theFilter)}.')
        count = 0
        for num in range(self.ntrace):
            trace = self.data[num, :]
            if theFilter.test(trace):
                count += 1
        print(f'Found {count} traces matching filter')
        filtered = np.zeros((count, 2000))
        count = 0
        for num in range(self.ntrace):
            trace = self.data[num, :]
            if theFilter.test(trace):
                filtered[count, :] = trace
                count += 1
        return DetData(filtered)

    # This is a factory to create a new DetData by loading the data
    # from a file. It is a class function so it has no self.
    def load(filename, verbose = True):
        array = np.load(filename)
        if verbose:
            print(f'Loaded array of {array.shape[0]} traces.')
        return DetData(array)
#
#   Main program compresses a set of .csv files into a single .npy
#
if __name__ == '__main__':
    path_str = '/Users/bcollett/Research/BL3/Deposits and Sources/DetectorData'
    pathlist = Path(path_str).glob('*.csv')
    nfile = 0
    for path in pathlist:
        nfile += 1
    print(f'Directory holds {nfile} csv files.')
    new_arr = np.zeros((nfile, 2000))
    anum = 0
    pathlist = Path(path_str).glob('*.csv')
    for path in pathlist:
        # because path is object not string
        path_str = str(path)   
        print(path_str)
        ar = np.loadtxt(path_str,skiprows=1,delimiter=',')
        print(ar.shape)
        voltages = ar[:, 1]
        times = ar[:, 0]
        if np.abs(times[-1] - times[0]) < 1e-7:
            times = np.linspace(0, 1.0e-5, num=20000, endpoint = False)
            print(times)
        dv = scipy.signal.decimate(voltages, 10)
        dt = scipy.signal.decimate(times, 10)
        new_arr[anum, :] = dv
        anum += 1
        plt.figure()
        plt.plot(times, voltages, dt, dv, '.')
        np.save("DataTest",new_arr)
        