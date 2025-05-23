#include "WF_SDK/WF_SDK.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

using namespace wf;

/* ----------------------------------------------------- */

int main(void)
{
  Device::Data *device_data;

  device_data = device.open();

  // PPS signal 100ns pulse per second
  wavegen.generate(device_data, 1, wavegen.function.pulse, 0, 1e3, 3.3, 0.01, 0.999, 1 / 1e3, 0, std::vector<double>());
  FDwfAnalogOutIdleSet(device_data->handle, 0, 0);
  FDwfAnalogOutConfigure(device_data->handle, 0, true);

  // Train signal 10 100ns pulse per 10ms
  wavegen.generate(device_data, 2, wavegen.function.pulse, 0, 1e3, 3.3, 0.01, 0.009, 1 / 1e3, 0, std::vector<double>());
  FDwfAnalogOutIdleSet(device_data->handle, 1, 0);
  FDwfAnalogOutConfigure(device_data->handle, 1, true);

  // Failed OR gate attempt
  // FDwfDigitalOutEnableSet(device_data->handle, 0, false);
  // FDwfDigitalOutEnableSet(device_data->handle, 1, false);
  // FDwfDigitalOutEnableSet(device_data->handle, 15, true);
  // FDwfDigitalOutTypeSet(device_data->handle, 1, DwfDigitalOutTypeROM);
  // FDwfDigitalOutDividerSet(device_data->handle, 15, 1);
  // FDwfDigitalOutOutputSet(device_data->handle, 15, DwfDigitalOutOutputPushPull);
  // uint8_t truthTableOR = 0b00001110;
  // FDwfDigitalOutDataSet(device_data->handle, 15, &truthTableOR, 8);
  // FDwfDigitalOutConfigure(device_data->handle, true);

  tools.sleep(10000);
  device.close(device_data);
  return 0;
}