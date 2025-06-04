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

  FDwfAnalogOutReset(device_data->handle, 1);

  // PPS
  std::vector<std::double_t> data = {0};
  data.resize(10000);
  data[9] = 1;

  FDwfAnalogOutNodeEnableSet(device_data->handle, 0, AnalogOutNodeCarrier, true);
  FDwfAnalogOutNodeFunctionSet(device_data->handle, 0, AnalogOutNodeCarrier, funcCustom);
  FDwfAnalogOutNodeFrequencySet(device_data->handle, 0, AnalogOutNodeCarrier, 1000.0);
  FDwfAnalogOutNodeAmplitudeSet(device_data->handle, 0, AnalogOutNodeCarrier, 3.3);
  FDwfAnalogOutNodeOffsetSet(device_data->handle, 0, AnalogOutNodeCarrier, 0.0);
  FDwfAnalogOutNodePhaseSet(device_data->handle, 0, AnalogOutNodeCarrier, 0.0);
  FDwfAnalogOutNodeDataSet(device_data->handle, 0, AnalogOutNodeCarrier, data.data(), data.size());
  FDwfAnalogOutModeSet(device_data->handle, 0, DwfAnalogOutModeVoltage);
  FDwfAnalogOutIdleSet(device_data->handle, 0, DwfAnalogOutIdleOffset);
  FDwfAnalogOutRunSet(device_data->handle, 0, 0.001);
  FDwfAnalogOutWaitSet(device_data->handle, 0, 0.0);
  FDwfAnalogOutRepeatSet(device_data->handle, 0, 0);
  FDwfAnalogOutTriggerSourceSet(device_data->handle, 0, trigsrcExternal1); // check later
  FDwfAnalogOutTriggerSlopeSet(device_data->handle, 0, DwfTriggerSlopeRise);
  FDwfAnalogOutRepeatTriggerSet(device_data->handle, 0, true);
  FDwfAnalogOutConfigure(device_data->handle, 0, true);

  // PAT
  FDwfAnalogOutNodeEnableSet(device_data->handle, 1, AnalogOutNodeCarrier, true);
  FDwfAnalogOutNodeFunctionSet(device_data->handle, 1, AnalogOutNodeCarrier, funcCustom);
  FDwfAnalogOutNodeFrequencySet(device_data->handle, 1, AnalogOutNodeCarrier, 1000.0);
  FDwfAnalogOutNodeAmplitudeSet(device_data->handle, 1, AnalogOutNodeCarrier, 3.3);
  FDwfAnalogOutNodeOffsetSet(device_data->handle, 1, AnalogOutNodeCarrier, 0.0);
  FDwfAnalogOutNodePhaseSet(device_data->handle, 1, AnalogOutNodeCarrier, 0.0);
  FDwfAnalogOutNodeDataSet(device_data->handle, 1, AnalogOutNodeCarrier, data.data(), data.size());
  FDwfAnalogOutModeSet(device_data->handle, 1, DwfAnalogOutModeVoltage);
  FDwfAnalogOutIdleSet(device_data->handle, 1, DwfAnalogOutIdleOffset);
  FDwfAnalogOutRunSet(device_data->handle, 1, 0.01);
  FDwfAnalogOutWaitSet(device_data->handle, 1, 0.0);
  FDwfAnalogOutRepeatSet(device_data->handle, 1, 0);
  FDwfAnalogOutTriggerSourceSet(device_data->handle, 1, trigsrcPC); // check later
  FDwfAnalogOutTriggerSlopeSet(device_data->handle, 1, DwfTriggerSlopeRise);
  FDwfAnalogOutRepeatTriggerSet(device_data->handle, 1, true);
  FDwfAnalogOutConfigure(device_data->handle, 1, true);

  // OR
  // FDwfDigitalOutEnableSet(device_data->handle, 0, 0);
  // FDwfDigitalOutEnableSet(device_data->handle, 1, 0);
  // FDwfDigitalOutEnableSet(device_data->handle, 15, 1);
  // FDwfDigitalOutTypeSet(device_data->handle, 15, DwfDigitalOutTypeROM);
  // FDwfDigitalOutDividerSet(device_data->handle, 15, 1);
  // FDwfDigitalOutOutputSet(device_data->handle, 15, DwfDigitalOutOutputPushPull);
  // uint8_t truthTableOR = 0b00001110;
  // FDwfDigitalOutDataSet(device_data->handle, 15, &truthTableOR, 1);
  // FDwfDigitalOutIdleSet(device_data->handle, 15, 0);
  // FDwfDigitalOutConfigure(device_data->handle, true);

  // exit when q is pressed
  std::cout << "Press 't' to trigger and 'q' to quit" << std::endl;
  std::string input;
  while (true)
  {
    std::getline(std::cin, input);
    if (input == "t")
    {
      FDwfDeviceTriggerPC(device_data->handle);
    } else if (input == "q")
    {
      device.close();
      return 0;
    }
  }
}