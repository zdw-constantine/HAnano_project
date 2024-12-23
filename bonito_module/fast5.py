import sys
from glob import glob
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing import Pool
from datetime import timedelta, timezone

import numpy as np
import reader
from tqdm import tqdm
from dateutil import parser
from ont_fast5_api.fast5_interface import get_fast5_file

from scipy.signal import find_peaks

def find_noisiest_section(signal, samples=100, threshold=6.0):
    """Find the noisiest section of a signal.
    
    Args:
        signal (np.array): raw nanopore signal
        samples (int): defaults to 100
        threshold (float): defaults to 6.0
        
    Returns:
        np.array : with a section (or all) the input signal that has the noisiest section
    """
    
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0; noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        tonorm = signal[info['left_bases'][widest]: info['right_bases'][widest]]
    else:
        tonorm = signal
        
    return tonorm

def med_mad(signal, factor=1.4826):
    """
    Calculate signal median and median absolute deviation
    
    Args:
        signal (np.array): array of data to calculate med and mad
        factor (float): factor to scale the mad
        
    Returns:
        float, float : med and mad values
    """
    med = np.median(signal)
    mad = np.median(np.absolute(signal - med)) * factor
    return med, mad

def trim(signal, window_size=40, threshold_factor=2.4, min_elements=3):
        """

        from: https://github.com/nanoporetech/bonito/blob/master/bonito/fast5.py
        """

        min_trim = 10
        signal = signal[min_trim:]

        med, mad = med_mad(signal[-(window_size*100):])

        threshold = med + mad * threshold_factor
        num_windows = len(signal) // window_size

        seen_peak = False

        for pos in range(num_windows):
            start = pos * window_size
            end = start + window_size
            window = signal[start:end]
            if len(window[window > threshold]) > min_elements or seen_peak:
                seen_peak = True
                if window[-1] > threshold:
                    continue
                return min(end + min_trim, len(signal)), len(signal)

        return min_trim, len(signal)

'''def trim(signal, window_size=40, threshold=2.4, min_trim=10, min_elements=3, max_samples=8000, max_trim=0.3):

    seen_peak = False
    num_windows = min(max_samples, len(signal)) // window_size

    for pos in range(num_windows):
        start = pos * window_size + min_trim
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            if end >= min(max_samples, len(signal)) or end / len(signal) > max_trim:
                return min_trim
            return end

    return min_trim
'''

class Read(reader.Read):

    def __init__(self, read, filename, meta=False, do_trim=True, scaling_strategy=None, norm_params=None):

        self.meta = meta
        self.read_id = read.read_id
        self.filename = filename.name
        self.run_id = read.get_run_id()
        if type(self.run_id) in (bytes, np.bytes_):
            self.run_id = self.run_id.decode('ascii')

        tracking_id = read.handle[read.global_key + 'tracking_id'].attrs

        try:
            self.sample_id = tracking_id['sample_id']
        except KeyError:
            self.sample_id = 'unset'
        if type(self.sample_id) in (bytes, np.bytes_):
            self.sample_id = self.sample_id.decode()

        #self.exp_start_time = tracking_id['exp_start_time']
        self.exp_start_time = tracking_id['exp_start_time']
        
        
        self.exp_start_time = "2024"
        
        if type(self.exp_start_time) in (bytes, np.bytes_):
            self.exp_start_time = self.exp_start_time.decode('ascii')
            pass
        self.exp_start_time = self.exp_start_time.replace('Z', '')

        self.flow_cell_id = tracking_id['flow_cell_id']
        if type(self.flow_cell_id) in (bytes, np.bytes_):
            self.flow_cell_id = self.flow_cell_id.decode('ascii')

        self.device_id = tracking_id['device_id']
        if type(self.device_id) in (bytes, np.bytes_):
            self.device_id = self.device_id.decode('ascii')

        if self.meta:
            return

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        self.offset = int(channel_info['offset'])
        self.sample_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']

        self.mux = read_attrs['start_mux']
        self.read_number = read_attrs['read_number']
        self.channel = channel_info['channel_number']
        if type(self.channel) in (bytes, np.bytes_):
            self.channel = self.channel.decode()

        self.start = read_attrs['start_time'] / self.sample_rate
        self.duration = read_attrs['duration'] / self.sample_rate

        exp_start_dt = parser.parse(self.exp_start_time)
        start_time = exp_start_dt + timedelta(seconds=self.start)
        self.start_time = start_time.astimezone(timezone.utc).isoformat(timespec="milliseconds")

        #start_time = exp_start_dt 
        #self.start_time = start_time

        raw = read.handle[read.raw_dataset_name][:]
        #print(raw)
        # print(self.offset)
        # print(channel_info['range'])
        # print(channel_info['digitisation'])
        self.scaled = np.array(self.scaling * (raw + self.offset), dtype=np.float64)

        #print(self.scaled)
        
        self.num_samples = len(self.scaled)

        med_mad_signal = find_noisiest_section(self.scaled,samples=100,threshold=6.0)
        #print(med_mad_signal)

        self.shift, self.scale = reader.normalisation(med_mad_signal, scaling_strategy, norm_params)

        #print(self.shift)
        #print(self.scale)
        #self.trimmed_samples = reader.trim(self.scaled, threshold=self.scale * 2.4 + self.shift) if do_trim else 0
        #self.template_start = self.start + (self.trimmed_samples / self.sample_rate)
        #self.template_duration = self.duration - (self.trimmed_samples / self.sample_rate)

        self.signal = (self.scaled.astype(np.float32) - self.shift) / self.scale
        #print(self.signal)

        trimed, _ = trim(self.signal[:8000])
        self.signal = self.signal[trimed:]

        #print("norm_signal")
        #print(self.signal)
        #print(len(self.signal))

def get_meta_data(filename, read_ids=None, skip=False):
    """
    Get the meta data from the fast5 file for a given `filename`.
    """
    meta_reads = []
    with get_fast5_file(filename, 'r') as f5_fh:
        try:
            all_read_ids = f5_fh.get_read_ids()
        except RuntimeError as e:
            sys.stderr.write(f"> warning: f{filename} - {e}\n")
            return meta_reads
        for read_id in all_read_ids:
            if read_ids is None or (read_id in read_ids) ^ skip:
                meta_reads.append(
                    Read(f5_fh.get_read(read_id), filename, meta=True)
                )
        return meta_reads


def get_read_groups(directory, model, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None):
    """
    Get all the read meta data for a given `directory`.
    """
    groups = set()
    num_reads = 0
    pattern = "**/*.fast5" if recursive else "*.fast5"
    fast5s = [Path(x) for x in glob(directory + "/" + pattern, recursive=True)]
    get_filtered_meta_data = partial(get_meta_data, read_ids=read_ids, skip=skip)

    with Pool(n_proc) as pool:
        for reads in tqdm(
                pool.imap(get_filtered_meta_data, fast5s), total=len(fast5s), leave=False,
                desc="> preprocessing reads", unit=" fast5s", ascii=True, ncols=100
        ):
            groups.update({read.readgroup(model) for read in reads})
            num_reads += len(reads)
        return groups, num_reads


def get_read_ids(filename, read_ids=None, skip=False):
    """
    Get all the read_ids from the file `filename`.
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        try:
            ids = [(filename, rid) for rid in f5_fh.get_read_ids()]
        except RuntimeError as e:
            sys.stderr.write(f"> warning: f{filename} - {e}\n")
            return []
        if read_ids is None:
            return ids
        return [rid for rid in ids if (rid[1] in read_ids) ^ skip]


def get_raw_data_for_read(info, do_trim=True, scaling_strategy=None, norm_params=None):
    """
    Get the raw signal from the fast5 file for a given filename, read_id pair
    """
    filename, read_id = info
    with get_fast5_file(filename, 'r') as f5_fh:
        return Read(f5_fh.get_read(read_id), filename, do_trim=do_trim, scaling_strategy=scaling_strategy, norm_params=norm_params)


def get_raw_data(filename, read_ids=None, skip=False):
    """
    Get the raw signal and read id from the fast5 files
    """
    with get_fast5_file(filename, 'r') as f5_fh:
        for read_id in f5_fh.get_read_ids():
            if read_ids is None or (read_id in read_ids) ^ skip:
                yield Read(f5_fh.get_read(read_id), filename)


def get_reads(directory, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None, do_trim=True, scaling_strategy=None, norm_params=None):
    """
    Get all reads in a given `directory`.
    """
    #print("get_reads")
    pattern = "**/*.fast5" if recursive else "*.fast5"
    get_filtered_reads = partial(get_read_ids, read_ids=read_ids, skip=skip)
    get_raw_data = partial(get_raw_data_for_read, do_trim=do_trim, scaling_strategy=scaling_strategy, norm_params=norm_params)
    reads = (Path(x) for x in glob(directory + "/" + pattern, recursive=True))
    with Pool(n_proc) as pool:
        for job in chain(pool.imap(get_filtered_reads, reads)):
            for read in pool.imap(get_raw_data, job):
                yield read
                if cancel is not None and cancel.is_set():
                    return