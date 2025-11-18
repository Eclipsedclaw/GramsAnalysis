#include "WF_SDK/WF_SDK.h"
#include "digilent/waveforms/dwf.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unistd.h> // for sleep function

using namespace wf;
Device::Data *device_data;

/* ----------------------------------------------------- */

bool or_func(int iAddress);
void configure_rom(int chan, bool (*func)(int));

int main(void)
{

  device_data = device.open();

  // Set up power supply
  FDwfAnalogIOReset(device_data->handle);
  FDwfAnalogIOChannelNodeSet(device_data->handle, 0, 0, 1); // turn on ps
  FDwfAnalogIOChannelNodeSet(device_data->handle, 0, 1, 3.3); // set ps voltage to 5 volts
  FDwfAnalogIOEnableSet(device_data->handle, true); // master enable
  sleep(1); // wait a second for the power supply voltage to come up (possibly unnecessary)

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
  FDwfAnalogOutTriggerSourceSet(device_data->handle, 1, trigsrcPC);
  FDwfAnalogOutTriggerSlopeSet(device_data->handle, 1, DwfTriggerSlopeRise);
  FDwfAnalogOutRepeatTriggerSet(device_data->handle, 1, true);
  FDwfAnalogOutConfigure(device_data->handle, 1, true);

  // OR
  configure_rom(15, or_func);
  FDwfDigitalOutConfigure(device_data->handle, 1);

  // exit when q is pressed
  std::cout << "Press 't' to trigger and 'q' to quit" << std::endl;
  std::string input;
  while (true)
  {
    std::getline(std::cin, input);
    if (input == "t")
    {
      FDwfDeviceTriggerPC(device_data->handle);
    }
    else if (input == "q")
    {
      break;
    }
  }
  device.close(device_data);
  return 0;
}

void configure_rom(int chan, bool (*func)(int))
{
  unsigned int customSize = 0;
  FDwfDigitalOutDataInfo(device_data->handle, chan, &customSize);

  std::cout << "Custom size: " << customSize
            << " Address space: " << (int)log2(customSize) << std::endl;

  int bufferSizeBytes = customSize / 8;
  std::vector<uint8_t> rgbSamples(bufferSizeBytes, 0);

  for (unsigned int iAddress = 0; iAddress < customSize; ++iAddress)
  {
    if (func(iAddress))
    {
      rgbSamples[iAddress / 8] |= (1 << (iAddress % 8));
    }
  }

  for (int i = 0; i < 3; ++i)
  {
    FDwfDigitalOutEnableSet(device_data->handle, 0, 0);
  }

  FDwfDigitalOutEnableSet(device_data->handle, chan, 1); // enable output
  FDwfDigitalOutTypeSet(device_data->handle, chan, DwfDigitalOutTypeROM);
  FDwfDigitalOutDividerSet(device_data->handle, chan, 1); // set minimum delay
  FDwfDigitalOutDataSet(device_data->handle, chan, rgbSamples.data(), customSize);
}

bool or_func(int iAddress)
{
  int fDio2 = (iAddress >> 2) & 1;
  int fDio1 = (iAddress >> 1) & 1;
  return (fDio2 == 1 || fDio1 == 1);
}