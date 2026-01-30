## How to plot pedestal data
### For file per board
1. First, make sure that your .bin file has been converted into a .root TTree. This can be done using scripts in the file_convert folder.
2. Run fileperboard_root_pedestal.py
3. This script will generate a histogram of the pedestal noise in a directory of your choosing.

### For file per channel
TO-DO

## How to plot combo data/create graphs of events
### For file per board
1. First, make sure that your .bin file has been converted into a .root TTree. This can be done using scripts in the file_convert folder.
2. You can choose either to plot the raw waveforms or the filtered waveforms. In either case, you must make sure to update the channel mapping in the scripts accordingly.
3. For the raw waveforms, run fileperboard_root_plot_raw.py
4. For the filtered waveforms, run fileperboard_root_plot_filtered.py

### For file per channel
TO-DO
