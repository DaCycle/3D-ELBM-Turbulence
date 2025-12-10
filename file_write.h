#pragma once
#include <string>

void createOutputFolder(const std::string& output_folder);
void writeToFile(int timestep, double* data, const std::string& dataname, const std::string& folderName, int size);