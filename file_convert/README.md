#### How to process collected pedestal data
1. The pedestal data is initially stored in binary format by the CAEN. For our purposes, we will need to convert this into a ROOT file. main.C is the script that will do this for us.
2. You will have to provide the location of the saved binary file. After running, it will create a .root file in the same location as the initial binary file.
3. Now, you have to run the python script in the other directory. It is called micro_GRAMS_pedestal.py. You will have to provide the location of the root file as well as the number of channels that data has been collected for. This is usually indicated in the file name.
4. After running, the script will save an image of a graph that displays all the pedestal values. This file will be in the same location as the initial file by default
