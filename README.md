# active used script
## If you are using data taking from Wavedump2 file per board, please follow the steps as follow
It is recommended to run these scripts using CAEN account so the environment is set correctely. 

To convert Wavedump2 file per board binary file to root, for multiple binary files run
```bash
python3 file_convert/robin_multibin_tree_write.py 
```
for single binary file run
```bash
python3 file_convert/robin_tree_write.py
```
This will pop up the input section for path of the home directory (Directory that prior to all the binary files) show as below. It will also ask you for output path and output name, defualt will be the same input directory. Please input the file name that you want

<img width="1097" height="461" alt="Screenshot 2025-12-12 at 9 16 28 AM" src="https://github.com/user-attachments/assets/4592a3d8-384c-4722-92c5-e714403fa764" />

After conversion finished, you can use root file for further analysis. To make plot (currently plot mapping fixed) you can run

```bash
python3 other/plot_raw_per_board.py
```
input the root file that you just generated. You can select the output path, defualt will be the same input directory.

<img width="1976" height="245" alt="Screenshot 2025-12-12 at 9 23 14 AM" src="https://github.com/user-attachments/assets/e663bd2f-14dc-469b-9285-3845ca69444a" />

## If you are using data taking from Wavedump2 file per channel, please follow the steps as follow
Convert bin to root file script in [file_convert](./file_convert). First run root script in terminal
```bash
root -l file_convert/WD2_fileperchannel_bin_2_root.C
```
This will pop up the input section for path of the file per channel home directory (Directory that prior to all the channel folders) show as below

<img width="802" height="97" alt="Screenshot 2025-12-11 at 11 30 05 AM" src="https://github.com/user-attachments/assets/579b68c2-64b3-4b75-b495-9d02c244133c" />

Once the processing finished, the root file should be saved also under the same input directory. Then you could perform further analysis and plot using this generated root file.

All plot and analysis scripts for R&D in lab under folder [other](./other). 
To make raw waveform plot, run the script shows below
```bash
python other/fileperchannel_root_plot_raw.py
```
Input the root file that was generated above. Then enter directly for default output folder that is under the same directory. If you prefer other options, you could also input output directory by yourself

<img width="1765" height="203" alt="Screenshot 2025-12-11 at 11 33 52 AM" src="https://github.com/user-attachments/assets/80dd682e-accb-49cd-9b87-e1685590457e" />

After selecting the ooutput, script will read the root file structure and number of channels information etc. At this point, you will be asked to input the channel mapping question for plot. The defualt mapping is here [label](https://docs.google.com/spreadsheets/d/1PFdLic8A5gqCuOfUtG62JrcyVAmm5fGXz86ElL5RMYo/edit?gid=0#gid=0), you could also make your own google sheet mapping to input here, just make sure your google sheet is public accessible.

<img width="1719" height="882" alt="Screenshot 2025-12-11 at 11 35 40 AM" src="https://github.com/user-attachments/assets/8d2b75b5-ed01-4b5d-87d7-678051513478" />

For google sheet, you only need to input column B(label) and C(flag), B(label) will determine the plot legend for the channel, while C(label) will determine the channel active or not / channel property.

<img width="540" height="729" alt="Screenshot 2025-12-11 at 11 42 15 AM" src="https://github.com/user-attachments/assets/b1a81574-0a48-4bdf-9371-888b6e6195fa" />

As for making filtered waveform plot, run the script shows below
```bash
python other/fileperchannel_root_plot_filtered.py
```
The rest of the steps stay the same as fileperchannel_root_plot_raw.py.


# GramsAnalysis
analysis pacakge for GRAMS experiment data

So far it is not uploaded yet. To use it, clone this repo and copy ~/GRAMSlib to your script directory.
