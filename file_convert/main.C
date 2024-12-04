
// The constructor in this class creates a root file containing trees/histograms. The public function fills the trees/histograms for each binary file.

// #include <vector>

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
    RootFileUpdater(const char* filename) {
        outfile = new TFile(filename, "UPDATE");

        tree = dynamic_cast<TTree*>(outfile->Get("tree"));
        if (!tree) {
            tree = new TTree("tree", "Tree");
        }

        histogram = dynamic_cast<TH1F*>(outfile->Get("histogram"));
        if (!histogram) {
            histogram = new TH1F("histogram", "Waveform max value", 200, 0, 200);
        }

        tree->Branch("event_id",         &event_id,        "event_id/i"); // event id
        tree->Branch("timestamp",        &timestamp,       "timestamp/g"); // trigger time stamp 
        tree->Branch("channel",          &channel,         "channel/i"); // event id
        tree->Branch("resolution",       &resolution,      "resolution/g"); // resolution
        tree->Branch("numberOfSamples",  &numberOfSamples, "numberOfSamples/i"); // number of samples
        tree->Branch("waveform_samples", waveform_samples, "waveform_samples[numberOfSamples]/F"); // wavefrom samples. This is an array. 
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


int main() {

    
    
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

    // for ( auto& iFile : fileList) {

    for (auto iFile = fileList.begin(); iFile != fileList.end(); ++iFile) {
		
		numberOfEvents = 0;
        unsigned int channel_number = 9999;
        std::regex pattern("_CH(\\d+)_");
        std::smatch match;

        if (std::regex_search(*iFile, match, pattern))
            channel_number = std::stoi(match[1].str());

        //std::cout << "channel_number number: " << channel_number << std::endl;

        if (stopLoop == true){
            createRootFile.FillDataFromRawFile(*iFile, channel_number, closeFile = true);
        } else if (std::next(iFile) != fileList.end()) {
            createRootFile.FillDataFromRawFile(*iFile, channel_number, closeFile = false);

        } else {
            createRootFile.FillDataFromRawFile(*iFile, channel_number, closeFile = true);

        }

        
        if(stopLoop == true) {break;}
        // std::cout << ": Data added to ROOT file from: " << fullFileName << std::endl;

    }


 
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time to process all the events: " << duration.count()/60000 << " Minutes" << std::endl;
    return 0;
}


/** 

How to use ROOT CINT to read a root file, show/scan files/events, draw the waveform. 

root -l *.root // get the tree with its name. Here tree has name tree. 
tree->Show(0) // shows the first event. channel, event id, timestamp, resolution, waveform, etc. 
tree->Scan("channel:event_id") // Scan() show everything but Scan("channel:event_id") shows the only two of them. 
tree->Draw("waveform_samples:Iteration$","channel==0 && event_id==0") // draw the waveform for channel 0, and event# 0. 

**/

