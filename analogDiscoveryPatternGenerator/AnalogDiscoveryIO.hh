#ifndef AnalogDiscoveryIO_H
#define AnalogDiscoveryIO_H 1

#ifdef WIN32
#include "../../inc/dwf.h"
#elif __APPLE__
#include "/Library/Frameworks/dwf.framework/Headers/dwf.h"
#else
#include <digilent/waveforms/dwf.h>
#endif

#include <iostream>
#include <vector>

#define ANALOG_OUT_STOP 0
#define ANALOG_OUT_START 1
#define ANALOG_OUT_APPLY 2

/**
 * A class to handle AnalogDiscovery
 *
 * @author Tsubasa Tamba, Shota Arai
 * @date 2023-03-15
 */

namespace gramsballoon {

class AnalogDiscoveryIO {
public:
  AnalogDiscoveryIO();
  int initialize();
  int connect(int device_id); // Connect to a device
  int connect(); // Connect to all devices
  // void setupAnalogOut(int device_id, int channel, double init_value = 0.0, bool triggerInput);
  void setupAnalogOut(int device_id, int channel, double init_value = 0.0, std::string signalType = "");
  void setupAnalogIn(int device_id, int channel, double freq, int buf_size, double range, double offset);
  void setVoltage(int device_id, int channel, double voltage, int sleep);
  // void implementORLogic();
  void finalize();

  int NumDevices() { return numDevices_; }
  const std::vector<HDWF> &HandlerList() const { return handlerList_; }

private:
  int numDevices_ = 0;
  std::vector<HDWF> handlerList_;
  char szError_[512] = {0};
  std::vector<std::vector<char>> deviceName_;
  std::vector<std::vector<char>> deviceSerialName_;
};

} /* namespace gramsballoon */

#endif /* AnalogDiscoveryIO_H */
