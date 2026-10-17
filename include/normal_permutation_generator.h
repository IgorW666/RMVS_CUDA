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
        std::vector<NormalPermutation> permutations;
        permutations.reserve(m_count);
        permutations.push_back({ 0.0f, 0.0f });
        const std::size_t childCount = m_count - 1;
        appendRing(permutations, coneAngleDegrees, childCount, 0.0f);
        if (dbgPrintPrmutations) {
            std::cout << "Normal Permutations:\n";
            for (const auto& perm : permutations) {
                std::cout << "Tangent: " << perm.tangent << ", Bitangent: " << perm.bitangent << "\n";
            }
        }
        return permutations;
    }

    std::vector<NormalPermutation> generateDouble(float coneAngleDegrees1, std::size_t count1,
        float coneAngleDegrees2, std::size_t count2, bool dbgPrintPermutations = false) const
    {
        if (1 + count1 + count2 != m_count)
            throw std::invalid_argument(
                "generateDouble: 1 + count1 + count2 must equal the total permutation count.");
        std::vector<NormalPermutation> permutations;
        permutations.reserve(m_count);
        permutations.push_back({ 0.0f, 0.0f });
        appendRing(permutations, coneAngleDegrees1, count1, 0.0f);
        appendRing(permutations, coneAngleDegrees2, count2, 0.5f);
        if (dbgPrintPermutations) {
            std::cout << "Normal Permutations (double):\n";
            for (const auto& perm : permutations) {
                std::cout << "Tangent: " << perm.tangent << ", Bitangent: " << perm.bitangent << "\n";
            }
        }
        return permutations;
    }

    std::vector<NormalPermutation> generateMulti(
        const std::vector<std::pair<float, int>>& rings, bool dbgPrintPermutations = false) const
    {
        std::size_t totalRequired = 1; // for parent normal at (0,0)
        for (const auto& ring : rings) {
            totalRequired += ring.second;
        }
        if (totalRequired != m_count)
            throw std::invalid_argument(
                "generateMulti: 1 + sum of all ring counts must equal the total permutation count.");
        std::vector<NormalPermutation> permutations;
        permutations.reserve(m_count);        
        permutations.push_back({ 0.0f, 0.0f });
        for (const auto& ring : rings) {
            appendRing(permutations, ring.first, ring.second, 0.0f);
        }
        if (dbgPrintPermutations) {
            std::cout << "Normal Permutations (multi):\n";
            for (const auto& perm : permutations) {
                std::cout << "Tangent: " << perm.tangent << ", Bitangent: " << perm.bitangent << "\n";
            }
        }
        return permutations;
    }

    static void saveNormalsAsXyz(const std::string& filename,
        const std::vector<NormalPermutation>& permutations)
    {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << "\n";
            return;
        }
        out << std::fixed << std::setprecision(6);
        for (const auto& perm : permutations) {            
            out << 0.0f << " " << 0.0f << " " << 0.0f << "\n";
            float nx = perm.tangent;
            float ny = perm.bitangent;
            float nz = 1.0f;
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 0.0f) {
                nx /= len;
                ny /= len;
                nz /= len;
            }            
            out << nx << " " << ny << " " << nz << "\n";
        }
        out.close();
    }

private:
    static void appendRing(std::vector<NormalPermutation>& permutations,
        float coneAngleDegrees, std::size_t count, float phase)
    {
        constexpr float pi = 3.14159265358979323846f;
        if (count == 0)
            return;
        const float radius = std::tan(coneAngleDegrees * pi / 180.0f);
        for (std::size_t i = 0; i < count; ++i)
        {
            const float theta = 2.0f * pi * (static_cast<float>(i) + phase) / static_cast<float>(count);
            permutations.push_back({ radius * std::cos(theta), radius * std::sin(theta)});
        }
    }
    std::size_t m_count;
};