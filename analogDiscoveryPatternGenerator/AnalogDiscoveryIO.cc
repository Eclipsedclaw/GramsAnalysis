#include "AnalogDiscoveryIO.hh"
#include <chrono>
#include <thread>

namespace gramsballoon {

std::string convert_vector_string(const std::vector<char> &vec) {
  std::string str;
  for (const auto &c: vec) {
    if (c == '\0') {
      break;
    }
    str.push_back(c);
  }
  return str;
}

AnalogDiscoveryIO::AnalogDiscoveryIO() {
}

int AnalogDiscoveryIO::initialize() {
  if (!FDwfEnum(enumfilterAll, &numDevices_)) {
    FDwfGetLastErrorMsg(szError_);
    std::cerr << "FDwfEnum failed: " << szError_ << std::endl;
    return -1;
  }

  std::cout << "number of Analog Discovery: " << numDevices_ << std::endl;

  handlerList_.resize(numDevices_);
  deviceName_.resize(numDevices_, std::vector<char>(32));
  deviceSerialName_.resize(numDevices_, std::vector<char>(32));

  for (int i = 0; i < numDevices_; i++) {
    FDwfEnumDeviceName(i, &deviceName_[i][0]);
    FDwfEnumSN(i, &deviceSerialName_[i][0]);
  }

  return 0;
}

int AnalogDiscoveryIO::connect(int device_id) {
  if (device_id < 0 || device_id >= numDevices_) {
    std::cerr << "Device ID " << device_id << " not connected" << std::endl;
    return -1;
  }
  if (!FDwfDeviceOpen(device_id, &handlerList_[device_id])) {
    FDwfGetLastErrorMsg(szError_);
    std::cerr << "Device open failed: device id = " << device_id << convert_vector_string(deviceName_[device_id]) << ",\n"
              << szError_ << std::endl;
    return -1;
  }
  return 0;
}

int AnalogDiscoveryIO::connect() {
  for (int i = 0; i < numDevices_; i++) {
    connect(i);
  }
  return 0;
}

void AnalogDiscoveryIO::setupAnalogOut(int device_id, int channel, double init_value, std::string signalType) {
  if (device_id < 0 || device_id >= NumDevices()) {
    std::cerr << "Device ID " << device_id << " not connected" << std::endl;
    return;
  }
  if (channel < 0 || channel >= 2) {
    std::cerr << "Channel " << channel << " is inappropriate." << std::endl;
    return;
  }

  // Parameters to setup for 
  const int sampleCount = 1e4;
  const double sampleRate = 1e7; // 10 MHz ==> 100 ns per sample, 
  std::vector<double> samples(sampleCount, 0.0);

  // Set the 5001st sample to 5V
  samples[sampleCount/2] = 5.0;


  FDwfAnalogOutReset(handlerList_[device_id], 0); // Reset the W1 before generating output
  FDwfAnalogOutNodeEnableSet(handlerList_[device_id], channel, AnalogOutNodeCarrier, true);

  FDwfAnalogOutNodeFunctionSet(handlerList_[device_id], channel, AnalogOutNodeCarrier, funcCustom);
  
  FDwfAnalogOutNodeDataSet(handlerList_[device_id], channel, AnalogOutNodeCarrier, samples.data(), sampleCount); // set the custom pulse data
  FDwfAnalogOutFrequencySet(handlerList_[device_id], channel, sampleRate/sampleCount); // set 1000Hz means 1 ms spacing here. 
  FDwfAnalogOutNodeAmplitudeSet(handlerList_[device_id], channel, AnalogOutNodeCarrier, 5);
  FDwfAnalogOutNodeOffsetSet(handlerList_[device_id], channel, AnalogOutNodeCarrier, init_value);

  FDwfAnalogOutTriggerSlopeSet(handlerList_[device_id], channel, DwfTriggerSlopeRise); 
  // bool TriggerCondition = FDwfAnalogOutTriggerSourceSet(handlerList_[device_id], channel, triggerInput); 

  // Configure trigger T1 for W1 to generate PAT 
  if (signalType=="PAT") {
      FDwfAnalogOutRunSet(handlerList_[device_id], channel,0.01 ); // 10ms
      FDwfAnalogOutConfigure(handlerList_[device_id], channel, ANALOG_OUT_START); // In the end to start the run. 
      // Run the pulse generator for 8 ms
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      // Stop the waveform generation 

  }
  if (signalType=="PPS") {
    FDwfAnalogOutRunSet(handlerList_[device_id], channel,0.001 ); // 1ms
    FDwfAnalogOutConfigure(handlerList_[device_id], channel, ANALOG_OUT_START); // In the end to start the run. 
    // Run the pulse generator for 8 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Stop the waveform generation 

  }

  // FDwfAnalogOutRunSet(handlerList_[device_id], channel, 5/1000); // Switch with sleep, somehow did not work

  // FDwfAnalogOutRepeatSet(device_data.handle, channel, ctypes.c_int(repeat))

  // std::cout << "Waiting for trigger T1 to start pulse generation..." << std::endl;
  // // Start waveform generation upon trigger
  // FDwfAnalogOutConfigure(handlerList_[device_id], channel, ANALOG_OUT_START); // In the end to start the run. 

  // // Run the pulse generator for 8 ms
  // std::this_thread::sleep_for(std::chrono::seconds(5));
  // // Stop the waveform generation 
  FDwfAnalogOutReset(handlerList_[device_id], channel);


}

void AnalogDiscoveryIO::setupAnalogIn(int device_id, int channel, double freq, int buf_size, double range, double offset) {
  if (device_id < 0 || device_id >= NumDevices()) {
    std::cerr << "Device ID " << device_id << " not connected" << std::endl;
    return;
  }
  if (channel < 0 || channel >= 2) {
    std::cerr << "Channel " << channel << " is inappropriate." << std::endl;
    return;
  }

  FDwfAnalogInFrequencySet(handlerList_[device_id], freq);
  FDwfAnalogInBufferSizeSet(handlerList_[device_id], buf_size);
  FDwfAnalogInChannelEnableSet(handlerList_[device_id], channel, true);
  FDwfAnalogInChannelRangeSet(handlerList_[device_id], channel, range);
  FDwfAnalogInChannelOffsetSet(handlerList_[device_id], channel, offset);

  const bool reconfigure = false;
  const bool data_aquisition = true;
  FDwfAnalogInConfigure(device_id + 1, reconfigure, data_aquisition);
#if 0
  double range_now = 15.0;
  FDwfAnalogInChannelRangeGet(device_id + 1, channel, &range_now);
  std::cout << "range: " << range_now << std::endl;
  double rgVoltsStep[32] = {0};
  int _pnSteps;
  FDwfAnalogInChannelRangeSteps(handlerList_[device_id], rgVoltsStep, &_pnSteps);
  std::cout << "RangeSteps: ";
  for (int i = 0; i < _pnSteps; i++)
  {
    std::cout << rgVoltsStep[i] << " ";
  }
  std::cout << std::endl;
  double pvoltsMin, pvoltsMax, pnSteps;
  FDwfAnalogInChannelRangeInfo(1, &pvoltsMin, &pvoltsMax, &pnSteps);
#endif
}

void AnalogDiscoveryIO::setVoltage(int device_id, int channel, double voltage, int sleep) {
  if (device_id < 0 || device_id >= NumDevices()) {
    std::cerr << "Device ID " << device_id << " not connected" << std::endl;
    return;
  }
  if (channel < 0 || channel >= 2) {
    std::cerr << "Channel " << channel << " is inappropriate." << std::endl;
    return;
  }

  FDwfAnalogOutNodeOffsetSet(handlerList_[device_id], channel, AnalogOutNodeCarrier, voltage);
  FDwfAnalogOutConfigure(handlerList_[device_id], channel, ANALOG_OUT_APPLY);
  std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
}

// void implementORLogic(){
  
//   FDwfDigitalOutInternalClockSet(handlerList_[device_id], 100000000.0); // 100 MHz clock (10 ns resolution)
//   FDwfDigitalIOOutputEnableSet(handlerList_[device_id], 0b00000110); // Enable DIO1, DIO2, and DIO3 as output
  
//   unsigned int dioState;
//   FDwfDigitalIOStatus(handlerList_[device_id]); 
//   FDwfDigitalIOInputStatus(handlerList_[device_id], &dioState);
  

//   bool DIO1 = dioState & 0b00000010; 
//   bool DIO2 = dioState & 0b00000100; 

//   bool DIO3 = DIO1 | DIO2;

//   unsigned int outputState = (DIO3 << 3); 
//   FDwfDigitalIOOutputSet(handlerList_[device_id], outputState);
  
//   std::this_thread::sleep_for(std::chrono::microseconds(10)); 
// }


void AnalogDiscoveryIO::finalize() {
  FDwfDeviceCloseAll();
}

// void 



} /* namespace gramsballoon */

/*



*/