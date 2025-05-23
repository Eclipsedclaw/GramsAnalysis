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
set(MY_EXE ad2_test)

# ## Check OS
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()

### options
### Waveforms ###
set(WAVEFORMS_LIB dwf)
set(WAVEFORMS_LIB_DIR /Library/Frameworks/dwf.framework)
set(WAVEFORMS_INC_DIR /Library/Frameworks/dwf.framework/Headers)
message("-- WAVEFORMS_LIB: ${WAVEFORMS_LIB}")
message("-- WAVEFORMS_LIB_DIR: ${WAVEFORMS_LIB_DIR}")
message("-- WAVEFORMS_INC_DIR: ${WAVEFORMS_INC_DIR}")

add_executable(${MY_EXE} main.cc)
target_include_directories(${MY_EXE} PRIVATE ${WAVEFORMS_INC_DIR})
target_link_libraries(${MY_EXE} ${WAVEFORMS_LIB})
target_link_directories(${MY_EXE} PRIVATE ${WAVEFORMS_LIB_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(MACOSX)
  set_target_properties(${MY_EXE} PROPERTIES BUILD_RPATH "/Library/Frameworks")
endif(MACOSX)