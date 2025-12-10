#include "file_write.h"

#include <iostream>
using namespace std;
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>
namespace fs = std::filesystem;

void createOutputFolder(const std::string& folderName) {
	if (!fs::exists(folderName)) {
		fs::create_directory(folderName);
	}
}

void writeToFile(int timestep, double* data, const std::string& dataname, const std::string& folderName, int size) {
	ostringstream fname;
	fname << folderName << "/" << dataname << "_" << setw(6) << setfill('0') << timestep << ".dat";
	string filename = fname.str();

	fstream fout;
	fout.open(filename, ios::out | ios::binary);
	if (fout) {
		fout.write(reinterpret_cast<char*>(data), size);
		fout.close();
	}
	else {
		cerr << "Error opening file: " << filename << endl;
	}
}