import os
import re

def getAllFiles(Binary_directory):
    items = []
    for root, dirs, files in os.walk(Binary_directory):
        for file in files:
            if file.endswith(".bin"):
            	if os.path.getsize(os.path.join(root, file)) > 0:
                    items.append(os.path.join(root, file))
    return sorted(items)

# Binary_directory = '/Users/nabinpoudyal/Data/MultiChannelDirectory/' # ONLY CHANGE THE FILE DIRECTORY. 
Binary_directory = '/NAS/LAr_TPC_runs/Run31/LArCombo5cmDrift_Run31_UPS_Pedestal6_4ch_12032024/' # 5 is 12
output_file = 'ListOfBinaryFilesToConvert.txt'

fileNameAndPath = getAllFiles(Binary_directory)

print(fileNameAndPath[0])
# /Users/nabinpoudyal/Data/MultiChannelDirectory/CH0/wave_dig2-192.168.0.254_CH0_20231113122550-02.bin

filename_ = fileNameAndPath[0].split("/")[-1].replace(".bin", ".root")
filename = re.sub(r'_CH\d+_','_', filename_)

print(filename)


with open(output_file, 'w') as f:
    f.write(filename +'\n')
    for path in fileNameAndPath:
        f.write(path + '\n')

print(f"All files have been written to '{output_file}'.")

