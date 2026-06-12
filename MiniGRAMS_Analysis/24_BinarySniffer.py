import os
import struct
import glob
import sys

def sniff_caen_channels(input_dir):
    """Sniffs the first binary file to find the true CAEN hardware channel IDs."""
    files = sorted(glob.glob(os.path.join(input_dir, "**/*.bin"), recursive=True) + 
                   glob.glob(os.path.join(input_dir, "**/*.dat"), recursive=True))
    
    if not files:
        print("No binary files found.")
        return
        
    first_file = files[0]
    print(f"Sniffing Binary Headers in: {os.path.basename(first_file)}")
    
    with open(first_file, 'rb') as f:
        header_bytes = f.read(28)
        if not header_bytes or len(header_bytes) < 28:
            print("File too small or corrupted.")
            return
            
        ev_num, timestamp, n_samples, resolution, n_channels = struct.unpack("<IQIQi", header_bytes)
        
        caen_channels = []
        for i in range(n_channels):
            # THIS IS THE MAGIC 2-BYTE HEADER!
            ch_header = f.read(2)
            caen_id = struct.unpack("<H", ch_header)[0]
            caen_channels.append(caen_id)
            
            # Skip the waveform payload to get to the next channel header
            f.read(n_samples * 4) 
            
        print("-" * 50)
        print(f"DAQ recorded {n_channels} channels.")
        print(f"TRUE CAEN HARDWARE IDs FOUND: {caen_channels}")
        print("-" * 50)
        
        if caen_channels != [30, 31, 32, 33]:
            print(f"WARNING: Your colleague did NOT record channels 30-33!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sniff_channels.py /path/to/binary/files")
    else:
        sniff_caen_channels(sys.argv[1])