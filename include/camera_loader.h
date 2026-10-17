#pragma once

#include "types.h"

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class CameraLoader {
public:
    /// Load cameras from CSV file
    /// @param csv_path Path to the CSV file containing camera parameters
    /// @return Vector of Camera structures with loaded data
    static std::vector<Camera> loadFromCSV(const std::string& csv_path);
};