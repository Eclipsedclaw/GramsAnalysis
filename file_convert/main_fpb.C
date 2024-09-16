

#include <TFile.h>
#include <TTree.h>
#include <map>
#include <string>

#include <vector>    
#include <TROOT.h>   
#include <TString.h>   
#include <TTree.h>   
#include <TFile.h>   
#include <TH1F.h>    
#include <iostream>  
#include <fstream>  
#include <algorithm> 
#include <ctime>     
#include <iomanip>   
#include <cmath>     

#include <filesystem>

#pragma link C++ class std::map<int, std::vector<float>>;


int numberOfEvents;

class RootFileUpdater {
private:
    TFile* outfile;
    TTree* tree;
    TH1F* histogram;

    unsigned int totalNumberOfChannels=9999, event_id=9999, numberOfSamples=9999;
    unsigned long long int timestamp=9999, resolution=9999;

    std::map<int, std::vector<float>> waveform_samples;

    bool closeOutputFile = false;

public:
    RootFileUpdater(const char* filename) {
        // Open ROOT file in update mode
        outfile = new TFile(filename, "UPDATE");

        tree = dynamic_cast<TTree*>(outfile->Get("tree"));
        if (!tree) {
            tree = new TTree("tree", "Tree");
        }

        histogram = dynamic_cast<TH1F*>(outfile->Get("histogram"));
        if (!histogram) {
            histogram = new TH1F("histogram", "Waveform max value", 200, 0, 200);
        }

        tree->Branch("event_id",              &event_id,              "event_id/i"); // event id
        tree->Branch("timestamp",             &timestamp,             "timestamp/g"); // trigger time stamp 
        tree->Branch("resolution",            &resolution,            "resolution/g"); // resolution
        tree->Branch("numberOfSamples",       &numberOfSamples,       "numberOfSamples/i"); // number of samples
        tree->Branch("totalNumberOfChannels", &totalNumberOfChannels, "totalNumberOfChannels/i"); // event id
        tree->Branch("waveform_samples", &waveform_samples); // wavefrom samples vector 

    }

    void FillDataFromRawFile(const string rawBinaryFilename, int channelMapping[], bool closeFile=false) {

        closeOutputFile = closeFile;

        std::ifstream *raw_input = new ifstream(rawBinaryFilename, std::ios::binary);
            
        if (!raw_input) {
            cout << "ERROR!. Input file is corrupted !!!" << endl;
            exit(0);
        }

        bool endOfTheFile = false; // Flag to control the outer while loop
                
        float each_sample = 0; 
        
        while (!endOfTheFile) {

            raw_input->read(reinterpret_cast<char*>(&event_id),4); // event number
            raw_input->read(reinterpret_cast<char*>(&timestamp),8); // time stamp
            raw_input->read(reinterpret_cast<char*>(&numberOfSamples),4); // number of samples
            raw_input->read(reinterpret_cast<char*>(&resolution),8); // resolution
            raw_input->read(reinterpret_cast<char*>(&totalNumberOfChannels),4); // total number of channels
            
            // if (totalNumberOfChannels != channelMapping.size()){
            //     cout << "ERROR!. Mismatch between metadata channel numbers and input channel numbers!!!" << end;
            //     cout << "Exiting Program!!!" << endl;
            //     exit(0);
            // }
            
            // cout << channelMapping.size() << endl;
            // for (int nCh = 0; nCh < totalNumberOfChannels; ++nCh) {
            //     cout << channelMapping[nCh] << endl;
            // }

            for (int nCh = 0; nCh < totalNumberOfChannels; ++nCh) {
                std::vector<float> waveform; 
                for (int i = 0; i < numberOfSamples; ++i) {
                    raw_input->read(reinterpret_cast<char*>(&each_sample),4); // correct one.  read the 4 bytes of waveform sample
                    // float temp_sample=0;
                    // raw_input->read(reinterpret_cast<char*>(&temp_sample), 4); // Read the 4 bytes of waveform sample
                    // if (i%100==0) {  cout << i <<" temp "<< each_sample << endl;}
                    // waveform_samples->emplace_back(temp_sample); 
                    waveform.push_back(each_sample);
                    if (raw_input->eof()) {cout << "EOF reached !!" << endl; endOfTheFile = true; break;}  

                }
                if (endOfTheFile==true) break;

                waveform_samples[nCh]= waveform;
            
                // float * max_ptr = std::max_element(std::begin(waveform_samples), std::end(waveform_samples));
                // // auto max_ptr = std::max_element(waveform_samples->begin(), waveform_samples->end());
                // float max_value = *max_ptr;
            }

            if (endOfTheFile==true) break;

            // if (numberOfEvents%100==0) {
                cout << "Summary:_________________________________________________________________________" << endl; 
                cout << "This is event number:------:" << numberOfEvents << endl;
                cout << "Event ID:------------------:" << event_id << endl;
                cout << "Time stamp (# of samples)--:" << timestamp << endl;
                cout << "Number of Samples:---------:" << numberOfSamples << endl;
                cout << "Resolution (ns):-----------:" << resolution << endl;
                cout << "Number of Channels:--------:" << totalNumberOfChannels << endl;
            // }

            tree->Fill();
            // histogram->Fill(max_value);
            numberOfEvents++;
        }

        cout << "Total number of events processed: " << numberOfEvents << endl;  
        tree->Write("", TObject::kOverwrite);
        histogram->Write("", TObject::kOverwrite);
        
        raw_input->close();

        if (closeOutputFile ==  true){
            outfile->Close();
        }
    }
    ClassDef(RootFileUpdater, 1);  // ROOT macro for class definition
};

ClassImp(RootFileUpdater);  // ROOT macro for class implementation

int main_fpb(int startCh, int endCh) {

    numberOfEvents = 0;

    int totalNumberOfchannel = endCh - startCh +1; 
    if (totalNumberOfchannel <= 1 ){
        cout << "ERROR!. Enter Input Channel Numbers in Ascending Order!!!" << endl;
        cout << "Exiting the program!!!" << endl;
        exit(0);
    }
    auto start = std::chrono::steady_clock::now();

    // int result = std::system("python fileListGenerator.py");

    // if (result == 0) {
    //     std::cout << "fileListGenerator.py executed successfully." << std::endl;
    // } else {
    //     std::cerr << "Error executing Python script." << std::endl;
    // }


    std::ifstream inputFile("ListOfBinaryFilesToConvert.txt"); 

    if (!inputFile) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    std::string outfileName_;
    std::getline(inputFile, outfileName_); // get first line as outfilename
    const char* outfileName = outfileName_.c_str(); 
    // cout << outfileName << endl;

    std::string line;
    std::vector<std::string> fileList;

    // Read lines from the file and push them into the vector
    while (std::getline(inputFile, line)) {
        fileList.push_back(line);
    }

    // If the expected roofile name already existed in the directory then remove it first. 
    if (std::filesystem::exists(outfileName)) {
        std::cout << "File exists. Deleting " << outfileName << "..." << std::endl;
        std::filesystem::remove(outfileName);
    }

    bool stopLoop = false;
    bool closeFile = false;

    RootFileUpdater createRootFile(outfileName);

    int channelMapping[totalNumberOfchannel];
    for (int nCh = 0; nCh < totalNumberOfchannel; ++nCh) {
        channelMapping[nCh] = startCh+nCh;
        // cout << channelMapping[nCh] << endl;
    }

    // for ( auto& iFile : fileList) {

    for (auto iFile = fileList.begin(); iFile != fileList.end(); ++iFile) {

        // cout << *iFile << endl;
        // unsigned int channel_number = 9999;
        // std::regex pattern("_CH(\\d+)_");
        // std::smatch match;

        // if (std::regex_search(*iFile, match, pattern))
            // channel_number = std::stoi(match[1].str());

        // std::cout << "channel_number number: " << channel_number << std::endl;

        // start fill the raw data
        if (std::next(iFile) != fileList.end()) {
            cout << "Keep root file open:" << *iFile << endl;
            createRootFile.FillDataFromRawFile(*iFile, channelMapping, closeFile = false);

        } else if (std::next(iFile) == fileList.end()){
            cout << "Close the root file:" << *iFile << endl;
            createRootFile.FillDataFromRawFile(*iFile, channelMapping, closeFile = true);

        }
        // end fill raw data
        // std::cout << ": Data added to ROOT file from: " << fullFileName << std::endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to process all the events: " << duration.count()/1000 << " seconds" << std::endl;
    return 0;
}


/** 

root -l main_fpb.C+'(0,63)'

How to use ROOT CINT to read a root file, show/scan files/events, draw the waveform. 

root -l *.root // get the tree with its name. Here tree has name tree. 
tree->Show(0) // shows the first event. channel, event id, timestamp, resolution, waveform, etc. 
tree->Scan("channel:event_id") // Scan() show everything but Scan("channel:event_id") shows the only two of them. 
tree->Draw("waveform_samples:Iteration$","channel==0 && event_id==0") // draw the waveform for channel 0, and event# 0. 

**/

