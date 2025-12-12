
// The constructor in this class creates a root file containing trees/histograms. The public function fills the trees/histograms for each binary file.

#include <iostream>
#include <readline/readline.h>
#include <readline/history.h>
#include <filesystem>
#include <vector>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;

// Function to generate suggestions for tab completion
char* directoryCompletion(const char* text, int state) {
    static std::vector<std::string> matches;
    static size_t index = 0;

    if (state == 0) { // Initialize new completion cycle
        matches.clear();
        index = 0;
        std::string prefix(text);

        // Check for directory completion
        fs::path basePath = prefix.empty() ? fs::current_path() : fs::path(prefix).parent_path();
        std::string searchPrefix = fs::path(prefix).filename().string();

        try {
            for (const auto& entry : fs::directory_iterator(basePath)) {
                std::string entryPath = entry.path().string();
                std::string filename = entry.path().filename().string();

                if (filename.find(searchPrefix) == 0) { // Matches prefix
                    if (entry.is_directory()) {
                        matches.push_back(entryPath + "/"); // Add trailing slash for directories
                    } else {
                        matches.push_back(entryPath);
                    }
                }
            }
            std::sort(matches.begin(), matches.end()); // Optional: sort suggestions
        } catch (const fs::filesystem_error&) {
            // Handle invalid paths gracefully
        }
    }

    if (index < matches.size()) {
        return strdup(matches[index++].c_str());
    }

    return nullptr; // No more matches
}

// Configure readline with tab completion
void configureReadline() {
    rl_attempted_completion_function = [](const char* text, int start, int end) -> char** {
        rl_completion_suppress_append = 1; // Prevent appending space after directory completion
        return rl_completion_matches(text, directoryCompletion);
    };
}

int numberOfEvents;

class RootFileUpdater {
private:
    TFile* outfile;
    TTree* tree;
    TH1F* histogram;

    unsigned int channel=9999, event_id=9999, numberOfSamples=9999;
    unsigned long long int timestamp=9999, resolution=9999;
    static const Int_t MaxSampleNumber = 125000;
    float waveform_samples[MaxSampleNumber]={-9999}; // waveform // hard-coded with max of 125000 dynamic array . 
    bool closeOutputFile = false;

public:
    // Constructor accepts binary directory and root file name as parameters
    RootFileUpdater(const std::string& binaryDirectory, const std::string& rootFileName) {
    fs::path outputDir(binaryDirectory); // Use the user-provided binary directory
    fs::path inputPath(rootFileName);    // Use the input root filename to get the stem
    std::string outputFileName = (outputDir / inputPath.stem()).string() + ".root";  // Construct output filename in the user directory

    // Print the output file location
    std::cout << "Output file will be saved to: " << outputFileName << std::endl;
    
    // Open the output ROOT file
    outfile = new TFile(outputFileName.c_str(), "RECREATE");

    // Initialize the tree and histogram
    tree = dynamic_cast<TTree*>(outfile->Get("tree"));
    if (!tree) {
        tree = new TTree("tree", "Tree");
    }

    histogram = dynamic_cast<TH1F*>(outfile->Get("histogram"));
    if (!histogram) {
        histogram = new TH1F("histogram", "Waveform max value", 200, 0, 200);
    }

    // Define the branches of the tree
    tree->Branch("event_id", &event_id, "event_id/i");
    tree->Branch("timestamp", &timestamp, "timestamp/g");
    tree->Branch("channel", &channel, "channel/i");
    tree->Branch("resolution", &resolution, "resolution/g");
    tree->Branch("numberOfSamples", &numberOfSamples, "numberOfSamples/i");
    tree->Branch("waveform_samples", waveform_samples, "waveform_samples[numberOfSamples]/F");
}


    void FillDataFromRawFile(const string rawBinaryFilename, unsigned int channelNumber, bool closeFile=false) {

        closeOutputFile = closeFile;

        std::ifstream *raw_input = new ifstream(rawBinaryFilename, std::ios::binary);

        //int numberOfEvents = 0;

        bool endOfTheFile = false; // Flag to control the outer while loop

        while (!endOfTheFile) {
            raw_input->read(reinterpret_cast<char*>(&event_id),4); // event number
            raw_input->read(reinterpret_cast<char*>(&timestamp),8); // time stamp
            channel = channelNumber;
            raw_input->read(reinterpret_cast<char*>(&numberOfSamples),4); // number of samples
            raw_input->read(reinterpret_cast<char*>(&resolution),8); // resolution

            for (int i = 0; i < numberOfSamples; ++i) {
                raw_input->read(reinterpret_cast<char*>(&waveform_samples[i]),4); // correct one.  read the 4 bytes of waveform sample
                if (raw_input->eof()) {
                //cout << "EOF reached !!" << endl; 
                endOfTheFile = true; break;}  

                // float temp_sample=0;
                // raw_input->read(reinterpret_cast<char*>(&temp_sample), 4); // Read the 4 bytes of waveform sample
                // cout << " temp "<< temp_sample << endl;
                // waveform_samples->emplace_back(temp_sample); 
            }
            if (endOfTheFile==true) break;

            float * max_ptr = std::max_element(std::begin(waveform_samples), std::end(waveform_samples));
            // auto max_ptr = std::max_element(waveform_samples->begin(), waveform_samples->end());

            float max_value = *max_ptr;
			/*
            if (numberOfEvents%1==0) {
                cout << "Summary:_________________________________________________________________________" << endl; 
                cout << "Number of events processed--:" << numberOfEvents << endl;
                cout << "Event ID:-------------------:" << event_id << endl;
                cout << "Time stamp (# of samples)---:" << timestamp << endl;
                cout << "Channel number--------------:" << channel << endl;
                cout << "Number of Samples:----------:" << numberOfSamples << endl;
                cout << "Resolution (ns):------------:" << resolution << endl;
                cout << "waveform max value (mV)-----:" << max_value << endl;
            }
            */
            tree->Fill();
            histogram->Fill(max_value);
            numberOfEvents++;
        }


        //cout << "Total number of events processed: " << numberOfEvents << endl;  
        tree->Write("", TObject::kOverwrite);
        histogram->Write("", TObject::kOverwrite);
        
        raw_input->close();

        if (closeOutputFile ==  true){
            outfile->Close();
        }


    }
};

// Function to generate a list of .bin files
std::vector<std::string> getBinaryFiles(const std::string& directory) {
    std::vector<std::string> fileList;

    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            if (fs::file_size(entry) > 0) {
                fileList.push_back(entry.path().string());
            }
        }
    }
    std::sort(fileList.begin(), fileList.end()); // Sort the files
    return fileList;
}

// Function to derive the root file name from the first binary file
std::string getRootFileName(const std::vector<std::string>& fileList) {
    if (fileList.empty()) {
        throw std::runtime_error("No files to process.");
    }

    std::string rootFileName = fs::path(fileList[0]).stem().string(); // Remove directory and extension
    std::regex channelPattern("_CH\\d+_");
    rootFileName = std::regex_replace(rootFileName, channelPattern, "_");
    rootFileName += ".root";
    return rootFileName;
}

int WD2_fileperchannel_bin_2_root() {
    auto start = std::chrono::steady_clock::now();

    // Configure readline for tab completion
    configureReadline();

    // Prompt user for input with tab completion enabled
    char* input = readline("Enter the path to the binary directory: ");
    if (input == nullptr || std::string(input).empty()) {
        std::cerr << "No input provided. Exiting." << std::endl;
        return 1;
    }

    std::string binaryDirectory(input);
    free(input);

    // Check if the directory exists
    if (!fs::exists(binaryDirectory) || !fs::is_directory(binaryDirectory)) {
        std::cerr << "Error: Directory does not exist or is not a valid directory." << std::endl;
        return 1;
    }

    std::cout << "Directory entered: " << binaryDirectory << std::endl;

    // Generate the list of binary files
    auto fileList = getBinaryFiles(binaryDirectory);

    if (fileList.empty()) {
        std::cerr << "No valid .bin files found in the directory: " << binaryDirectory << std::endl;
        return 1;
    }

    // Get the derived root file name
    std::string rootFileName;
    try {
        rootFileName = getRootFileName(fileList);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "Root file name: " << rootFileName << std::endl;

    // Check if the root file already exists and remove it
    if (std::filesystem::exists(rootFileName)) {
        std::cout << "File exists. Deleting " << rootFileName << "..." << std::endl;
        std::filesystem::remove(rootFileName);
    }

    // Initialize the RootFileUpdater
    RootFileUpdater createRootFile(binaryDirectory, rootFileName);

    // Process binary files in a loop
    bool closeFile = false;
    for (size_t i = 0; i < fileList.size(); ++i) {
        const std::string& binaryFile = fileList[i];

        unsigned int channel_number = 9999;
        std::regex pattern("_CH(\\d+)_");
        std::smatch match;

        if (std::regex_search(binaryFile, match, pattern)) {
            channel_number = std::stoi(match[1].str());
        }

        std::cout << "Processing file: " << binaryFile << " (Channel: " << channel_number << ")" << std::endl;

        // Close the root file after processing the last binary file
        closeFile = (i == fileList.size() - 1);

        // Fill data from the binary file into the root file
        createRootFile.FillDataFromRawFile(binaryFile, channel_number, closeFile);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to process all the events: " << duration.count() / 60000.0 << " Minutes" << std::endl;

    return 0;
}



/** 

How to use ROOT CINT to read a root file, show/scan files/events, draw the waveform. 

root -l *.root // get the tree with its name. Here tree has name tree. 
tree->Show(0) // shows the first event. channel, event id, timestamp, resolution, waveform, etc. 
tree->Scan("channel:event_id") // Scan() show everything but Scan("channel:event_id") shows the only two of them. 
tree->Draw("waveform_samples:Iteration$","channel==0 && event_id==0") // draw the waveform for channel 0, and event# 0. 

**/

