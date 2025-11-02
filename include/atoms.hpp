#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstring>

std::vector<float> read_atoms_as_vec4(std::istream &in) {
    std::unordered_map<std::string, int> atomicNumbers = {
        {"H", 1},
        {"He", 2}
    };

    std::vector<float> result;
    int numAtoms = 0;
    in >> numAtoms;
    if (!in) throw std::runtime_error("Failed to read number of atoms");

    std::string symbol;
    float x, y, z;

    for (int i = 0; i < numAtoms; ++i) {
        in >> symbol >> x >> y >> z;
        if (!in) throw std::runtime_error("Failed to read atom data");

        auto it = atomicNumbers.find(symbol);
        if (it == atomicNumbers.end())
            throw std::runtime_error("Unsupported atom symbol: " + symbol);

        float w = it->second;

        result.push_back(x);
        result.push_back(y);
        result.push_back(z);
        result.push_back(w);
    }

    return result;
}
