#include "AnalogDiscoveryIO.hh"
#include <chrono>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <poll.h>

using namespace gramsballoon;


bool check_ManualTrigger() {
  struct pollfd cli_input = {};
  cli_input.fd = 0; // 0 means keyboard
  cli_input.events = POLLIN;
  return poll(&cli_input, 1, 0) == 1;
}


int main(int argc, char const *argv[]) {
  double voltage = 0.0;

  if (argc == 1) {
    voltage = 0.0;
  }
  else if (argc == 2) {
    try {
      voltage = std::stod(argv[1]);
    }
    catch (const std::invalid_argument &e) {
      std::cerr << "Invalid argument: " << argv[1] << std::endl;
      return 1;
    }
  }
  else {
    std::cerr << "Usage: " << argv[0] << " voltage" << std::endl;
    return 1;
  }
  AnalogDiscoveryIO io;
  const int ret = io.initialize(); // Detect how many AD2 are connected, and start Commnication with them.
  if (ret != 0) {
    return ret;
  }
  const int connect = io.connect(0); // Connect to the first AD2
  //const int connect = io.connect(); // Connect to all AD2
  if (connect != 0) {
    return connect;
  }



  std::cout << "Waiting for the trigger signal" << std::endl;
  std::string ManualTrigger, PPSextTrigger, PAT = "PAT", PPS = "PPS";

  while (true){

    while(true){
      io.setupAnalogOut(0, 0, voltage, PPS); 
      std::this_thread::sleep_for(std::chrono::seconds(1)); //T2 use trigger input for PPS generation
        
      if (check_ManualTrigger()) {
        std::getline(std::cin, ManualTrigger);  
        std::cout << "Received trigger: " << ManualTrigger << std::endl;

        if (ManualTrigger == "1") {
          io.setupAnalogOut(0, 1, voltage, PAT); //T1

        }
      }
     
      // Fed to DIO pin and apply OR logic
      //io.implementORLogic(); 
      


    }

  }



  return 0;
}

