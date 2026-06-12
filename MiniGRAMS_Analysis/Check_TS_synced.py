import struct
import argparse

def read_first_n_timestamps(filepath, num_events=20):
    timestamps = []
    header_format = "<IQIQi"
    header_size = struct.calcsize(header_format)
    
    try:
        with open(filepath, 'rb') as f:
            for _ in range(num_events):
                header_bytes = f.read(header_size)
                if not header_bytes or len(header_bytes) < header_size: break
                
                ev_num, ts, n_samples, samp_period, n_channels = struct.unpack(header_format, header_bytes)
                timestamps.append(ts)
                
                payload_size = (n_channels * 2) + (n_channels * n_samples * 4)
                f.seek(payload_size, 1) 
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return timestamps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file72', help="Path to first .bin file for Board 72")
    parser.add_argument('file75', help="Path to first .bin file for Board 75")
    parser.add_argument('file85', help="Path to first .bin file for Board 85")
    args = parser.parse_args()
    
    print(f"--- DAQ TIMESTAMP OFFSET RECON ---")
    ts_72 = read_first_n_timestamps(args.file72)
    ts_75 = read_first_n_timestamps(args.file75)
    ts_85 = read_first_n_timestamps(args.file85)
    
    print(f"\n{'Evt':<5} | {'Board 72 (Master)':<20} | {'Board 75 (Slave 1)':<20} | {'Board 85 (Slave 2)':<20}")
    print("-" * 75)
    
    for i in range(max(len(ts_72), len(ts_75), len(ts_85))):
        t72 = ts_72[i] if i < len(ts_72) else "-"
        t75 = ts_75[i] if i < len(ts_75) else "-"
        t85 = ts_85[i] if i < len(ts_85) else "-"
        print(f"{i:<5} | {str(t72):<20} | {str(t75):<20} | {str(t85):<20}")

if __name__ == "__main__":
    main()