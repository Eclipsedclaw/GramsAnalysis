cmake_minimum_required(VERSION 3.20)
#CMAKE_MINIMUM_REQUIRED_VERSION(VERSION 3.14)

### Initial definition of cmake variables
set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type")
set(CMAKE_CXX_FLAGS_DEBUG "-g -W -Wall" CACHE STRING "CXX_FLAGS for debug")
set(CMAKE_C_FLAGS_DEBUG "-g -W -Wall" CACHE STRING "C_FLAGS for debug")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -W -Wall" CACHE STRING "CXX_FLAGS for release")
set(CMAKE_C_FLAGS_RELEASE "-O3 -W -Wall" CACHE STRING "C_FLAGS for release")
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

### Definition of project
project(GRAMSBalloon CXX C)
set(PROJECT_VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(MY_LIBRARY ad2_test)

### options
### Waveforms ###
set(WAVEFORMS_LIB dwf)
message("-- WAVEFORMS_LIB: ${WAVEFORMS_LIB}")


add_executable(${MY_LIBRARY} main.cc AnalogDiscoveryIO.cc)
target_link_libraries(${MY_LIBRARY} ${WAVEFORMS_LIB})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})