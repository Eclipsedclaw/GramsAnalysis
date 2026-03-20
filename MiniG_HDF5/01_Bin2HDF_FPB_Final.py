import struct
import numpy as np
import h5py
import os
import re
import argparse
import time
from multiprocessing import Pool, cpu_count
import detector_config as dc  

def extract_metadata(filename):
    """
    Parses filename for: Acq ID, HV Status, Run Config, Digitizer, Timestamp.
    """
    base = os.path.basename(filename)
    base_lower = base.lower()
    
    # Initialize dict for metadata
    meta = {}
    
    # --- Identify acquisition no. ---
    acq_match = re.search(r'(?i)acq(\d+)', base)
    if acq_match:
        meta['acq_tag'] = f"acq{acq_match.group(1)}"
    else:
        meta['acq_tag'] = f"acq_unknown_{base}"

    # --- Identify acquisition/run configuration ---
    if 'light-pedestal' in base_lower:
        meta['run_config'] = 'light_only'
        meta['hv_tag'] = 'PedestalDataConverted'
        meta['field_on'] = False
        
    elif 'light-sig' in base_lower:
        meta['run_config'] = 'light_only'
        meta['hv_tag'] = 'ScintDataConverted'  
        meta['field_on'] = False

    elif 'combo-pedestal' in base_lower:
        meta['run_config'] = 'combo'
        meta['hv_tag'] = 'PedestalDataConverted'
        meta['field_on'] = False
        
    elif 'combo' in base_lower: 
        meta['run_config'] = 'combo'
        hv_match = re.search(r'TPCHV(\d+)', base)
        if hv_match:
            meta['hv_tag'] = f"TPCHV{hv_match.group(1)}"
            meta['field_on'] = True
        else:
            meta['hv_tag'] = 'PedestalDataConverted' 
            meta['field_on'] = False

    else:
        print(f"[WARN] Unknown Run Config for {base}. Defaulting to 'light_only'.")
        meta['run_config'] = 'light_only'
        meta['hv_tag'] = 'UnknownConfig'
        meta['field_on'] = False

    # --- Identify digitizer ID ---
    dig_match = re.search(r'(dig\d+)', base)
    if dig_match:
        meta['dig_id'] = dig_match.group(1)
    elif 'dig2' in base: 
        meta['dig_id'] = 'dig2'
    else:
        meta['dig_id'] = 'unknown_dig'

    # --- Identify WV2 timestamp ---
    ts_match = re.search(r'(\d{14})', base)
    if ts_match:
        meta['timestamp'] = ts_match.group(1)
    else:
        meta['timestamp'] = "00000000000000"

    return meta

# --- WORKER FUNCTION ---
def convert_single_file(task):
    input_path, output_root, limit = task
    filename = os.path.basename(input_path)
    
    meta = extract_metadata(filename)
    
    # Organize: Output / Category / Acq_ID / File.h5
    save_dir = os.path.join(output_root, meta['hv_tag'], meta['acq_tag'])
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass 
            
    # Naming: [timestamp]_[config]_[hv]_[acq].h5
    output_name = f"{meta['timestamp']}_{meta['run_config']}_{meta['hv_tag']}_{meta['acq_tag']}.h5"
    save_path = os.path.join(save_dir, output_name)

    print(f"[Start] {filename} -> .../{meta['acq_tag']}/{output_name}")

    try:
        with h5py.File(save_path, 'w') as hf:
            dset_waveforms = None
            dset_ids = None
            dset_timestamps = None
            
            for k, v in meta.items():
                hf.attrs[k] = v
            
            events_processed = 0
            
            with open(input_path, 'rb') as f:
                while True:
                    if limit > 0 and events_processed >= limit:
                        break

                    header_bytes = f.read(28)
                    if not header_bytes or len(header_bytes) < 28:
                        break 
                        
                    ev_num, timestamp, n_samples, resolution, n_channels = struct.unpack("<IQIQi", header_bytes)
                    
                    if dset_waveforms is None:
                        chunk_shape = (1, n_channels, n_samples)
                        dset_waveforms = hf.create_dataset("waveforms_mV", 
                                                           shape=(0, n_channels, n_samples), 
                                                           maxshape=(None, n_channels, n_samples),
                                                           dtype='f4', 
                                                           chunks=chunk_shape)
                        dset_ids = hf.create_dataset("event_ids", shape=(0,), maxshape=(None,), dtype='i4')
                        dset_timestamps = hf.create_dataset("timestamps", shape=(0,), maxshape=(None,), dtype='u8')
                        
                        hf.attrs['resolution_ns'] = resolution
                        
                        # --- DYNAMIC DETECTOR MAP CALL ---
                        # Use detector_config to identify channel types based on run_config
                        det_types = [dc.get_channel_type(ch, meta['run_config']) for ch in range(n_channels)]
                        hf.attrs['detector_config'] = np.array([s.encode('utf-8') for s in det_types])

                    event_waveforms = np.zeros((n_channels, n_samples), dtype=np.float32)
                    
                    for i in range(n_channels):
                        f.read(2) 
                        bytes_to_read = n_samples * 4
                        wave_bytes = f.read(bytes_to_read)
                        
                        if len(wave_bytes) < bytes_to_read:
                            print(f"[Error] {filename}: Unexpected EOF")
                            return 

                        event_waveforms[i, :] = np.frombuffer(wave_bytes, dtype=np.float32)

                    new_size = dset_ids.shape[0] + 1
                    dset_waveforms.resize((new_size, n_channels, n_samples))
                    dset_ids.resize((new_size,))
                    dset_timestamps.resize((new_size,))
                    
                    dset_waveforms[new_size-1] = event_waveforms
                    dset_ids[new_size-1] = ev_num
                    dset_timestamps[new_size-1] = timestamp
                    
                    events_processed += 1
            
            print(f"[Done]  {output_name}: {events_processed} events.")
            
    except Exception as e:
        print(f"[FAIL]  {filename}: {e}")
        return

# --- MAIN ---
def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Input file or directory")
    parser.add_argument('--output', '-o', default='./HDF5_Output', help="Destination Directory")
    parser.add_argument('--workers', '-w', type=int, default=cpu_count(), help="Cores")
    parser.add_argument('--limit', '-l', type=int, default=0, help="Max events")
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    tasks = []
    if os.path.isfile(args.input):
         if args.input.endswith(('.bin', '.dat')):
            tasks.append((args.input, args.output, args.limit))
    elif os.path.isdir(args.input):
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.endswith(('.bin', '.dat')):
                    tasks.append((os.path.join(root, file), args.output, args.limit))
    
    print(f"Files Found: {len(tasks)}")
    print(f"Workers:     {args.workers}")
    
    if len(tasks) > 0:
        with Pool(args.workers) as p:
            p.map(convert_single_file, tasks)
    else:
        print("No files found.")

    print(f"Complete. Time: {time.time() - script_start_time:.2f} s")

if __name__ == "__main__":
    main()