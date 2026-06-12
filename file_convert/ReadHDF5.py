import h5py

filename = '/NAS/pGRAMS_backup_chamber/Run1/pGRAMS_backup_GAr_combo_vis100TRG_acq15_HV500V_20260218/pGRAMS_backup_GAr_combo_vis100TRG_acq15_HV500V_20260218.h5'
with h5py.File(filename, 'r') as f:
    # List all top-level objects (datasets and groups) in the file
    print("Keys:", list(f.keys()))

    # Access a specific dataset by its name
    dataset_name = 'my_dataset' 
    # Dataset data type: [('event_num', '<u4'), ('timestamp', '<u8'), ('num_of_samples', '<u4'), ('resolution', '<u8'), ('num_of_channels', '<i4'), ('active_channels', '<i2', (33,)), ('waveform_data', '<f4', (33, 37500))]
    if dataset_name in f:
        data = f[dataset_name] 
        print("\nDataset shape:", data.shape)
        print("Dataset data type:", data.dtype)

        # Read the entire dataset into a NumPy array
        # Use [()] to load the whole dataset into memory
        data_array = data[()] 
        # print("\n Array for the first event:", data_array[:1])
        print("\n First event array:", data_array['waveform_data'][0][1])
    else:
        print(f"\nDataset '{dataset_name}' not found.")