cmake_minimum_required(VERSION 3.19)

# Define the project name and language
project(multiheadAttentionExample LANGUAGES CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the executable from main.cpp
add_executable(multiheadAttentionExample main.cpp)

# Include the library headers directory
target_include_directories(multiheadAttentionExample PRIVATE ../../include)
