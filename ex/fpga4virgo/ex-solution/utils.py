# ============== IMPORTS ==============
from datetime import datetime
import gc
from gwpy.timeseries import TimeSeries
import numpy as np
import os
from pathlib import Path
import pprint
from pynq import Overlay, allocate
import time


# ============== MATH UTILITIES ==============
# Check if number is power of 2
def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


# ============== DATA LOADING UTILITIES ==============
# Load data
def load_data(input_dir, nfiles=100, target_shape=64, normalize=True):

    if not is_power_of_2(target_shape):
        raise ValueError("The target shape must be a power of 2")

    # This transforms the input data in the desired target shape
    if target_shape > 256:
        raise ValueError("The target shape is too large! Please choose a smaller value!")
    if target_shape == 256:
        NSPLITS=512
    elif target_shape == 128:
        NSPLITS=1024
    elif target_shape == 64:
        NSPLITS=2048
    elif target_shape == 32:
        NSPLITS=4096
    elif target_shape == 16:
        NSPLITS=8192
    else:
        raise ValueError("The target shape is too small! Please choose a larger value!")
    
    # Read input files
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".hdf5")]
    files_to_load = all_files[:nfiles]

    # list of loaded inputs
    time_series_list = []

    # Loop on input files and read data
    counter=0
    for fname in files_to_load:
        counter=counter+1
        path = os.path.join(input_dir, fname)
        ts = TimeSeries.read(path)
        array = np.array(ts)
        splitted = np.split(array, NSPLITS)
        time_series_list.extend(splitted)
    chunks = np.stack(time_series_list)
    chunks = chunks[..., np.newaxis]

    if normalize:
        # Normalize the inputs
        X_min = chunks.min(axis=1, keepdims=True)
        X_max = chunks.max(axis=1, keepdims=True)
        chunks = 2 * (chunks - X_min) / (X_max - X_min) - 1

    return chunks


# ============== HLS4ML UTILITIES ==============
# Pretty-print of hls4ml configuration file
def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

# Parse the synthesis report
def get_report(indir):
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    data_ = {}
    if report_csynth.is_file():
        print('Found valid synth in {} -> Fetching numbers:'.format(indir))

        # Get the resources from the logic synthesis report
        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            # Latency data
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_us']   = float(lat_line.split('|')[3].split()[0])
            data_['latency_ii']   = int(lat_line.split('|')[6])
            # Resources data
            res_tot   = "  " + lines[np.argwhere(np.array(['Utilization Estimates' in line for line in lines])).flatten()[0] + 14]
            res_avail = "  " + lines[np.argwhere(np.array(['Utilization Estimates' in line for line in lines])).flatten()[0] + 20]
            res_perc  = "  " + lines[np.argwhere(np.array(['Utilization Estimates' in line for line in lines])).flatten()[0] + 22]

            resources_line = '''
  +---------------------+---------+------+---------+---------+-----+
  |         Name        | BRAM_18K|  DSP |    FF   |   LUT   | URAM|
  +---------------------+---------+------+---------+---------+-----+
'''
            endline = "  +---------------------+---------+------+---------+---------+-----+"
            data_['resources'] = resources_line + res_tot + res_avail + res_perc + endline

    return data_

# Print the synthesis report
def print_report(indir):
    rep = get_report(indir)
    output=" Latency (clks): {} \n Latency (us)  : {} \n Latency II    : {} \n Resources     : {}".format(
        rep['latency_clks'], rep['latency_us'], rep['latency_ii'], rep['resources']
    )
    print(output)


# ============== NEURAL NETWORK OVERLAY ==============
class NeuralNetworkOverlay(Overlay):
    def __init__(self, xclbin_name, dtbo=None, download=True, ignore_version=False, device=None):
        super().__init__(xclbin_name, dtbo=dtbo, download=download, ignore_version=ignore_version, device=device)
        self.input_buffer = None
        self.output_buffer = None
        self._current_input_shape = None
        self._current_output_shape = None
        self._current_dtype = None

    def allocate_mem(self, X_shape, y_shape, dtype=np.float32, trg_in=None, trg_out=None):
        """Buffer allocation in the accelerator's memory.
        
        Reuses buffers if shapes and dtype match. Only reallocates if needed.

        Args:
            X_shape (tuple): Input buffer shape.
            y_shape (tuple): Output buffer shape.
            dtype (dtype, optional): The data type of the elements. Defaults to np.float32.
            trg_in (optional): Input buffer target memory. Defaults to None.
            trg_out (optional): Output buffer target memory. Defaults to None.
        """
        # Check if we can reuse existing buffers
        if (self.input_buffer is not None and 
            self._current_input_shape == X_shape and 
            self._current_output_shape == y_shape and 
            self._current_dtype == dtype):
            # Buffers already allocated with correct shapes, reuse them
            return
        
        # Need to reallocate - free old buffers first
        self._free_buffers()
        
        # Allocate new buffers
        self.input_buffer = allocate(shape=X_shape, dtype=dtype, target=trg_in)
        self.output_buffer = allocate(shape=y_shape, dtype=dtype, target=trg_out)
        
        # Store current shapes for future comparison
        self._current_input_shape = X_shape
        self._current_output_shape = y_shape
        self._current_dtype = dtype

    def _free_buffers(self):
        """Helper method to properly free buffers."""
        if self.input_buffer is not None:
            try:
                self.input_buffer.flush()
                del self.input_buffer
            except Exception:
                pass
            self.input_buffer = None
            
        if self.output_buffer is not None:
            try:
                self.output_buffer.flush()
                del self.output_buffer
            except Exception:
                pass
            self.output_buffer = None
        
        self._current_input_shape = None
        self._current_output_shape = None
        self._current_dtype = None

    def predict(self, X, y_shape, dtype=np.float32, debug=False, profile=False, encode=None, decode=None):
        """Obtain the predictions of the NN implemented in the FPGA.

        Args:
            X (ndarray): The input tensor.
            y_shape (tuple): The shape of the output tensor.
            dtype (dtype, optional): The data type of the elements. Defaults to np.float32.
            debug (bool, optional): Print debug information. Defaults to False.
            profile (bool, optional): Print performance metrics. Defaults to False.
            encode (Callable, optional): Function to transform the input tensor. Defaults to None.
            decode (Callable, optional): Function to transform the output tensor. Defaults to None.

        Returns:
            np.ndarray or tuple: Output predictions, and optionally (latency, throughput) if profile=True
        """
        # Allocate buffers if needed (will reuse if shapes match)
        self.allocate_mem(X_shape=X.shape, y_shape=y_shape, dtype=dtype)
        
        if profile:
            timea = datetime.now()
            
        if encode is not None:
            X = encode(X)
            
        in_size = np.prod(X.shape)
        out_size = np.prod(y_shape)
        
        # Copy data to device
        self.input_buffer[:] = X
        self.input_buffer.sync_to_device()
        
        if debug:
            print("Send OK")
            
        # Execute kernel
        self.krnl_rtl_1.call(self.input_buffer, self.output_buffer, in_size, out_size)
        
        if debug:
            print("Kernel call OK")
            
        # Get results from device
        self.output_buffer.sync_from_device()
        
        if debug:
            print("Receive OK")
            
        # Copy result
        result = self.output_buffer.copy()
        
        if decode is not None:
            result = decode(result)
        
        # DON'T free buffers here - keep them for reuse!
        
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, len(X))
            return result, dts, rate
            
        return result

    def free_overlay(self):
        """Free the overlay and all associated buffers."""
        # Free buffers first
        self._free_buffers()
        
        # Free the overlay
        try:
            self.free()
        except Exception as e:
            print(f"Warning during overlay free: {e}")
        
        # Force garbage collection and wait for XRT
        gc.collect()
        time.sleep(0.1)

    def _print_dt(self, timea, timeb, N):
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = N / dts
        print(f"Classified {N} samples in {dts} seconds ({rate} inferences / s)")
        print(f"Or {1 / rate * 1e6} us / inferences")
        return dts, rate

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self._free_buffers()
        except Exception:
            pass
