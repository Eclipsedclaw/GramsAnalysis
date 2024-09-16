import os
import re

def getAllFiles(Binary_directory):
    items = []
    for root, dirs, files in os.walk(Binary_directory):
        for file in files:
            if file.endswith(".bin"):
                items.append(os.path.join(root, file))
    return sorted(items)

# Binary_directory = '/Users/nabinpoudyal/Data/MultiChannelDirectory/' # ONLY CHANGE THE FILE DIRECTORY. 

Binary_directory = '/HDD/jon/CSP_test/LArComboFullDrift_Run27_EmmaTile_FilteredLAr8_90Deg_CSP_fluctuation_32chans_090132024/'

output_file = 'ListOfBinaryFilesToConvert.txt'

fileNameAndPath = getAllFiles(Binary_directory)

print(fileNameAndPath[0])

filename = fileNameAndPath[0].split("/")[-1].replace(".bin", ".root")
# filename = re.sub(r'_CH\d+_','_', filename_)

print(filename)


with open(output_file, 'w') as f:
    f.write(filename +'\n')
    for path in fileNameAndPath:
        f.write(path + '\n')

print(f"All files have been written to '{output_file}'.")

