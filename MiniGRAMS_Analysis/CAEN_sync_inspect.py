import struct
import os
import argparse
import glob

def diagnose_caen_binary(filepath, events_to_read=3):
    """
    Tactical script to peek into a CAEN wavedump2 binary file,
    decode the headers, and verify payload byte alignment.
    """
    filename = os.path.basename(filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"--- Opening: {filename} ({file_size_mb:.2f} MB) ---")
    print(f"{'='*60}")
    
    # Header format from documentation: 
    # Event number (4, uint), Timestamp (8, uint), Samples (4, uint), 
    # Sampling Period (8, uint), Channels (4, int)
    header_format = "<IQIQi"
    header_size = struct.calcsize(header_format) # Should be 28 bytes
    
    try:
        with open(filepath, 'rb') as f:
            for ev in range(events_to_read):
                print(f"\n[Event {ev + 1} / {events_to_read}]")
                header_bytes = f.read(header_size)
                
                if not header_bytes or len(header_bytes) < header_size:
                    print("--> EOF or incomplete header reached.")
                    break
                    
                ev_num, timestamp, n_samples, sampling_period, n_channels = struct.unpack(header_format, header_bytes)
                
                print(f"  Event Number    : {ev_num}")
                print(f"  Timestamp       : {timestamp}")
                print(f"  Samples/Ch      : {n_samples}")
                print(f"  Sampling Period : {sampling_period} ns")
                print(f"  Channels        : {n_channels}")
                
                # Check for the 2-byte channel header previously observed in our old pipeline
                print("  -- Inspecting first channel payload --")
                ch_header_test = f.read(2)
                if len(ch_header_test) == 2:
                    print(f"  First 2 bytes (Hex): {ch_header_test.hex()}")
                
                # Rewind the 2 bytes to safely skip the whole payload
                f.seek(-2, os.SEEK_CUR) 
                
                # Calculate the payload size to skip to the next event
                # Assuming 2-byte channel header + (4 bytes per sample for float mV)
                bytes_per_channel = 2 + (n_samples * 4) 
                total_payload_skip = n_channels * bytes_per_channel
                
                print(f"  Attempting to skip {total_payload_skip} bytes to reach next event...")
                f.seek(total_payload_skip, os.SEEK_CUR)
                
    except Exception as e:
        print(f"[FAIL] Error reading file {filename}: {e}")

def sweep_directory(target_dir, events=3):
    """Scans a directory for binary files and runs the diagnostic on each."""
    if not os.path.isdir(target_dir):
        print(f"[ERROR] Target is not a valid directory: {target_dir}")
        return

    # Find all .bin and .dat files in the directory
    search_pattern_bin = os.path.join(target_dir, '*.bin')
    search_pattern_dat = os.path.join(target_dir, '*.dat')
    
    target_files = glob.glob(search_pattern_bin) + glob.glob(search_pattern_dat)
    
    if not target_files:
        print(f"[WARN] No .bin or .dat files found in {target_dir}")
        return
        
    print(f"[SITREP] Found {len(target_files)} binary files in {target_dir}. Commencing sweep...")
    
    for filepath in sorted(target_files):
        diagnose_caen_binary(filepath, events)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Board CAEN Binary Diagnostic Recon")
    parser.add_argument('target_dir', help="Directory containing the .bin or .dat files from a single acquisition")
    parser.add_argument('--events', type=int, default=3, help="Number of events to read per file")
    args = parser.parse_args()
    
    sweep_directory(args.target_dir, args.events)