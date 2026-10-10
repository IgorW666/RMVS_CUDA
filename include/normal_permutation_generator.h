#pragma once

#include "types.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

class NormalPermutationGenerator
{
public:

    explicit NormalPermutationGenerator(std::size_t count)
        : m_count(count)
    {
        if (m_count < 2)
            throw std::invalid_argument(
                "NormalPermutationGenerator requires at least 2 permutations.");
    }

    std::vector<NormalPermutation> generateSingle(float coneAngleDegrees, bool dbgPrintPrmutations = false) const
    {
        constexpr float pi = 3.14159265358979323846f;

        const float coneAngleRadians =
            coneAngleDegrees * pi / 180.0f;

        const float radius =
            std::tan(coneAngleRadians);

        std::vector<NormalPermutation> permutations;
        permutations.reserve(m_count);

        // Permutation 0: unchanged parent normal.
        permutations.push_back({ 0.0f, 0.0f });

        const std::size_t childCount = m_count - 1;

        for (std::size_t i = 0; i < childCount; ++i)
        {
            const float theta =
                2.0f * pi * static_cast<float>(i)
                / static_cast<float>(childCount);

            permutations.push_back({
                radius * std::cos(theta),
                radius * std::sin(theta)
            });
        }

        if (dbgPrintPrmutations) {
            std::cout << "Normal Permutations:\n";
            for (const auto& perm : permutations) {
                std::cout << "Tangent: " << perm.tangent << ", Bitangent: " << perm.bitangent << "\n";
            }
		}

        return permutations;
    }

    // Debug: save generated normals as 3D points in an .xyz file for verification.
    // Writes 2 points per normal: origin (0,0,0) and the normalized permuted normal (x,y,z).
    // The parent normal is assumed to be (0,0,1). The permutation (tangent, bitangent)
    // is applied as the x and y components with z = 1 before normalization.
    // Format: first line = number of points, second line = comment, then one point per line "x y z".
    void saveNormalsAsXyz(const std::string& filename, float coneAngleDegrees) const
    {
        const auto permutations = generateSingle(coneAngleDegrees, false);

        const std::size_t totalPoints = permutations.size() * 2;

        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << "\n";
            return;
        }

        //out << totalPoints << "\n";
        //out << "Normal permutations exported by NormalPermutationGenerator\n";
        out << std::fixed << std::setprecision(6);

        for (const auto& perm : permutations) {
            // Point 1: origin
            out << 0.0f << " " << 0.0f << " " << 0.0f << "\n";

            // Compute permuted normal from parent (0,0,1)
            float nx = perm.tangent;
            float ny = perm.bitangent;
            float nz = 1.0f;

            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 0.0f) {
                nx /= len;
                ny /= len;
                nz /= len;
            }

            // Point 2: tip of the normal
            out << nx << " " << ny << " " << nz << "\n";
        }

        out.close();
    }

private:
    std::size_t m_count;
};