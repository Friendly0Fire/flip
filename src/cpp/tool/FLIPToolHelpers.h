/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Visualizing and Communicating Errors in Rendered Images
// Ray Tracing Gems II, 2021,
// by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
// Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

// Visualizing Errors in Rendered High Dynamic Range Images
// Eurographics 2021,
// by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
// Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

// FLIP: A Difference Evaluator for Alternating Images
// High Performance Graphics 2020,
// by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
// Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
// Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

// Code by Pontus Ebelin (formerly Andersson), Jim Nilsson, and Tomas Akenine-Moller.

#pragma once
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#define FIXED_DECIMAL_DIGITS(x, d) std::fixed << std::setprecision(d) << (x)

#include "../FLIP.h"

#include "imagehelpers.h"
#include "commandline.h"
#include "filename.h"
#include "pooling.h"

namespace FLIPTool
{
    inline std::string f2s(float value, size_t decimals = 4)
    {
        std::stringstream ss;
        ss << std::string(value < 0.0f ? "m" : "p") << FIXED_DECIMAL_DIGITS(std::abs(value), decimals);
        return ss.str();
    }

    // Here follows a set of helps functions for differet setups in order to avoid clutter in main().

    static void setupDestinationDirectory(const bool useHDR, const commandline& commandLine, std::string& destinationDirectory)
    {
        if (commandLine.optionSet("directory"))
        {
            destinationDirectory = commandLine.getOptionValue("directory");
            std::replace(destinationDirectory.begin(), destinationDirectory.end(), '\\', '/');      // Replace backslash with forwardslash.
            const bool bNoExposureMap = useHDR ? commandLine.optionSet("no-exposure-map") : true;
            const bool bSaveLDRImages = useHDR ? commandLine.optionSet("save-ldr-images") : false;
            const bool bSaveLDRFLIP = useHDR ? commandLine.optionSet("save-ldrflip") : false;
            const bool willCreateOutput = (!commandLine.optionSet("no-error-map")) || (!bNoExposureMap) || bSaveLDRImages || bSaveLDRFLIP || commandLine.optionSet("histogram");

            if (!std::filesystem::exists(destinationDirectory) && willCreateOutput)     // Create directories if the parameters indicate that some files will be saved.
            {
                std::cout << "Creating new directory(s): <" << destinationDirectory << ">.\n";
                std::filesystem::create_directories(destinationDirectory);
            }
        }
    }

    static void setupPixelsPerDegree(const commandline& commandLine, FLIP::parameters& parameters)
    {
        // The default value in parameters.PPD is computed as FLIP::calculatePPD(0.7f, 3840.0f, 0.7f); in FLIP.h.
        if (commandLine.optionSet("pixels-per-degree"))
        {
            parameters.ppd = std::stof(commandLine.getOptionValue("pixels-per-degree"));
        }
        else if (commandLine.optionSet("viewing-conditions"))
        {
            const float monitorDistance = std::stof(commandLine.getOptionValue("viewing-conditions", 0));
            const float monitorWidth = std::stof(commandLine.getOptionValue("viewing-conditions", 1));
            const float monitorResolutionX = std::stof(commandLine.getOptionValue("viewing-conditions", 2));
            parameters.ppd = FLIP::calculate_ppd(monitorDistance, monitorResolutionX, monitorWidth);
        }
    }

    static void getExposureParameters(const bool useHDR, const commandline& commandLine, FLIP::parameters& parameters, bool& returnLDRFLIPImages, bool& returnLDRImages)
    {
        if (useHDR)
        {
            if (commandLine.optionSet("tone-mapper"))   // The default in FLIP::Parameters.tonemapper is "aces".
            {
                std::string tonemapper = commandLine.getOptionValue("tone-mapper");
                std::ranges::transform(tonemapper, tonemapper.begin(), [](unsigned char c) { return std::tolower(c); });
                if (tonemapper != "aces" && tonemapper != "reinhard" && tonemapper != "hable")
                {
                    std::cout << "\nError: unknown tonemapper, should be one of \"ACES\", \"Reinhard\", or \"Hable\"\n";
                    std::exit(-1);
                }
                if(tonemapper == "aces")
                    parameters.tonemapper = FLIP::tonemapper::aces;
                else if(tonemapper == "reinhard")
                    parameters.tonemapper = FLIP::tonemapper::reinhard;
                else if(tonemapper == "hable")
                    parameters.tonemapper = FLIP::tonemapper::hable;
            }
            if (commandLine.optionSet("start-exposure"))
            {
                parameters.exposure.min = std::stof(commandLine.getOptionValue("start-exposure"));
            }
            if (commandLine.optionSet("stop-exposure"))
            {
                parameters.exposure.max = std::stof(commandLine.getOptionValue("stop-exposure"));
            }
            if (commandLine.optionSet("num-exposures"))
            {
                parameters.num_exposures = atoi(commandLine.getOptionValue("num-exposures").c_str());
            }
            returnLDRFLIPImages = commandLine.optionSet("save-ldrflip");
            returnLDRImages = commandLine.optionSet("save-ldr-images");
        }
    }

    static void saveErrorAndExposureMaps(const bool useHDR, commandline& commandLine, const FLIP::parameters& parameters, const std::string basename,
        FLIP::image<float>& errorMapFLIP, FLIP::image<float>& maxErrorExposureMap, const std::string& destinationDirectory,
        FLIP::filename& referenceFileName, FLIP::filename& testFileName, FLIP::filename& histogramFileName, FLIP::filename& txtFileName,
        FLIP::filename& flipFileName, FLIP::filename& exposureFileName, const size_t verbosity, const size_t testFileCount)
    {
        std::string logString = commandLine.optionSet("log") ? "log_" : "";
        if ((basename != "" && commandLine.getOptionValues("test").size() == 1))
        {
            flipFileName.setName(basename);
            histogramFileName.setName(logString + basename);
            txtFileName.setName(basename);
            exposureFileName.setName(basename + ".exposure_map");
        }
        else
        {
            flipFileName.setName(referenceFileName.getName() + "." + testFileName.getName() + "." + std::to_string(int(std::round(parameters.ppd))) + "ppd");
            if (!useHDR)
            {
                flipFileName.setName(flipFileName.getName() + ".ldr");  // Note that the HDR filename is not complete until after FLIP has been computed, since FLIP may update the exposure parameters.
            }
        }

        if (useHDR) // Updating the flipFileName here, since the computation of FLIP may have updated the exposure parameters.
        {
            if (verbosity > 1 && testFileCount == 0)
            {
                std::cout << "     Assumed tone mapper: " << to_string(parameters.tonemapper) << "\n";
                std::cout << "     Start exposure: " << FIXED_DECIMAL_DIGITS(parameters.exposure.min, 4) << "\n";
                std::cout << "     Stop exposure: " << FIXED_DECIMAL_DIGITS(parameters.exposure.max, 4) << "\n";
                std::cout << "     Number of exposures: " << parameters.num_exposures << "\n\n";
            }

            flipFileName.setName(std::format("{}.hdr.{}.{}_to_{}.{}",
                flipFileName.getName(), to_string(parameters.tonemapper), parameters.exposure.min, parameters.exposure.max, parameters.num_exposures));
            exposureFileName.setName("exposure_map." + flipFileName.getName());
        }

        if (!basename.empty() && commandLine.getOptionValues("test").size() == 1)
        {
            flipFileName.setName(basename);
            exposureFileName.setName(basename + ".exposure_map");
            histogramFileName.setName(basename + "." + logString + "weighted_histogram");
            testFileName.setName(basename);
        }
        else
        {
            flipFileName.setName("flip." + flipFileName.getName());
            histogramFileName.setName(logString + "weighted_histogram." + flipFileName.getName());
            txtFileName.setName("pooled_values." + flipFileName.getName());
        }

        if (!commandLine.optionSet("no-error-map"))
        {
            FLIP::image<FLIP::float3> pngResult(errorMapFLIP.get_width(), errorMapFLIP.get_height());
            if (!commandLine.optionSet("no-magma"))
            {
                pngResult.apply_color_map(errorMapFLIP, FLIP::MagmaMap);
            }
            else
            {
                pngResult.expand_grayscale(errorMapFLIP);
            }
            ImageHelpers::pngSave(destinationDirectory + "/" + flipFileName.toString(), pngResult);
        }

        if (useHDR)
        {
            if (!commandLine.optionSet("no-exposure-map"))
            {
                FLIP::image<FLIP::float3> pngMaxErrorExposureMap(maxErrorExposureMap.get_width(), maxErrorExposureMap.get_height());
                pngMaxErrorExposureMap.apply_color_map(maxErrorExposureMap, FLIP::ViridisMap);
                ImageHelpers::pngSave(destinationDirectory + "/" + exposureFileName.toString(), pngMaxErrorExposureMap);
            }
        }
    }

    static void setExposureStrings(const int exposureCount, const float exposure, std::string& expCount, std::string& expString)
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(3) << exposureCount;
        expCount = ss.str();
        ss.str(std::string());
        ss << std::string(exposure < 0.0f ? "m" : "p") << std::to_string(std::abs(exposure));
        expString = ss.str();
    }

    // Optionally store the intermediate LDR images and LDR-FLIP error maps produced during the evaluation of HDR-FLIP.
    static void saveIntermediateHDRFLIPOutput(commandline& commandLine, const FLIP::parameters& parameters, const std::string& basename, const FLIP::filename& flipFileName,
        const FLIP::filename& referenceFileName, const FLIP::filename& testFileName, const std::string& destinationDirectory,
        std::vector<FLIP::image<float>*> intermediateLDRFLIPImages, std::vector<FLIP::image<FLIP::float3>*> intermediateLDRImages)
    {
        if (intermediateLDRImages.size() > 0)
        {
            FLIP::filename rFileName(".png");
            FLIP::filename tFileName(".png");
            if (intermediateLDRImages.size() != static_cast<size_t>(parameters.num_exposures * 2))
            {
                std::cout << "FLIP tool error: the number of LDR images from HDR-FLIP is not the expected number.\nExiting.\n";
                std::exit(EXIT_FAILURE);
            }

            const float exposureStepSize = parameters.exposure.range() / static_cast<float>(parameters.num_exposures - 1);
            for (int i = 0; i < parameters.num_exposures; i++)
            {
                std::string expCount, expString;
                setExposureStrings(i, parameters.exposure.min + i * exposureStepSize, expCount, expString);

                if (basename.empty())
                {
                    rFileName.setName(std::format("{}.{}.{}.{}",
                        referenceFileName.getName(), parameters.tonemapper, expCount, expString));
                    tFileName.setName(std::format("{}.{}.{}.{}",
                        testFileName.getName(), parameters.tonemapper, expCount, expString));
                }
                else
                {
                    rFileName.setName(std::format("{}.reference.{}", basename, expCount));
                    tFileName.setName(std::format("{}.test.{}", basename, expCount));
                }
                FLIP::image<FLIP::float3>* rImage = intermediateLDRImages[0];
                FLIP::image<FLIP::float3>* tImage = intermediateLDRImages[1];
                intermediateLDRImages.erase(intermediateLDRImages.begin());
                intermediateLDRImages.erase(intermediateLDRImages.begin());
                rImage->linear_rgb_to_srgb();
                tImage->linear_rgb_to_srgb();
                ImageHelpers::pngSave(destinationDirectory + "/" + rFileName.toString(), *rImage);
                ImageHelpers::pngSave(destinationDirectory + "/" + tFileName.toString(), *tImage);
                delete rImage;
                delete tImage;
            }
        }
        if (intermediateLDRFLIPImages.size() > 0)
        {
            if (intermediateLDRFLIPImages.size() != size_t(parameters.num_exposures))
            {
                std::cout << "FLIP tool error: the number of FLIP LDR images from HDR-FLIP is not the expected number.\nExiting.\n";
                exit(EXIT_FAILURE);
            }

            const float exposureStepSize = parameters.exposure.range() / (parameters.num_exposures - 1);
            for (int i = 0; i < parameters.num_exposures; i++)
            {
                std::string expCount, expString;
                setExposureStrings(i, parameters.exposure.min + i * exposureStepSize, expCount, expString);

                FLIP::image<float>* flipImage = intermediateLDRFLIPImages[0];
                intermediateLDRFLIPImages.erase(intermediateLDRFLIPImages.begin());

                FLIP::image<FLIP::float3> pngResult(flipImage->get_width(), flipImage->get_height());

                if (!commandLine.optionSet("no-magma"))
                {
                    pngResult.apply_color_map(*flipImage, FLIP::MagmaMap);
                }
                else
                {
                    pngResult.expand_grayscale(*flipImage);
                }
                if (basename.empty())
                {
                    ImageHelpers::pngSave(std::format("{}/flip.{}.{}.{}ppd.ldr.{}.{}.{}.png",
                        destinationDirectory, referenceFileName.getName(), testFileName.getName(), static_cast<int>(std::round(parameters.ppd)), parameters.tonemapper, expCount, expString), pngResult);
                }
                else
                {
                    ImageHelpers::pngSave(std::format("{}/{}.{}.png",
                        destinationDirectory, basename, expCount), pngResult);
                }
                ImageHelpers::pngSave(std::format("{}/{}",
                    destinationDirectory, flipFileName.toString()), pngResult);
                delete flipImage;
            }
        }
    }

    static void gatherStatisticsAndSaveOutput(commandline& commandLine, FLIP::image<float>& errorMapFLIP, FLIPPooling::pooling<float>& pooledValues,
        const std::string& destinationDirectory, const FLIP::filename& referenceFileName, const FLIP::filename& testFileName, const FLIP::filename& histogramFileName,
        const FLIP::filename& txtFileName, const FLIP::filename& flipFileName, const FLIP::filename& exposureFileName, const std::string& FLIPString, const float time,
        const uint32_t testFileCount, const bool saveOverlappedHistogram, const bool useHDR, const size_t verbosity)
    {
        for (int y = 0; y < errorMapFLIP.get_height(); y++)
        {
            for (int x = 0; x < errorMapFLIP.get_width(); x++)
            {
                pooledValues.update(x, y, errorMapFLIP.get(x, y));
            }
        }

        if (commandLine.optionSet("histogram") && !saveOverlappedHistogram)
        {
            bool optionLog = commandLine.optionSet("log");
            bool optionExcludeValues = commandLine.optionSet("exclude-pooled-values");
            float yMax = (commandLine.optionSet("y-max") ? std::stof(commandLine.getOptionValue("y-max")) : 0.0f);
            pooledValues.save(destinationDirectory + "/" + histogramFileName.toString(), errorMapFLIP.get_width(), errorMapFLIP.get_height(), optionLog, !optionExcludeValues, yMax);
        }

        // Collect pooled values and elapsed time.
        float mean = pooledValues.getMean();
        float weightedMedian = pooledValues.getPercentile(0.5f, true);
        float firstWeightedQuartile = pooledValues.getPercentile(0.25f, true);
        float thirdWeightedQuartile = pooledValues.getPercentile(0.75f, true);
        float minValue = pooledValues.getMinValue();
        float maxValue = pooledValues.getMaxValue();

        if (commandLine.optionSet("textfile"))
        {
            FLIP::filename textFileName(destinationDirectory + "/" + txtFileName.toString());

            std::ofstream txt;
            txt.open(textFileName.toString());
            if (txt.is_open())
            {
                txt.seekp(0, std::ios_base::end);

                txt << "Mean: " << FIXED_DECIMAL_DIGITS(mean, 6) << "\n";
                txt << "Weighted median: " << FIXED_DECIMAL_DIGITS(weightedMedian, 6) << "\n";
                txt << "1st weighted quartile: " << FIXED_DECIMAL_DIGITS(firstWeightedQuartile, 6) << "\n";
                txt << "3rd weighted quartile: " << FIXED_DECIMAL_DIGITS(thirdWeightedQuartile, 6) << "\n";
                txt << "Min: " << FIXED_DECIMAL_DIGITS(minValue, 6) << "\n";
                txt << "Max: " << FIXED_DECIMAL_DIGITS(maxValue, 6) << "\n";

                txt.close();
            }
            else
            {
                std::cout << "\nError: Could not write txt file " << textFileName.toString() << "\n";
            }
        }

        if (commandLine.optionSet("csv"))
        {
            FLIP::filename csvFileName(commandLine.getOptionValue("csv"));

            std::fstream csv;
            csv.open(csvFileName.toString(), std::ios::app);
            if (csv.is_open())
            {
                csv.seekp(0, std::ios_base::end);

                if (csv.tellp() <= 0)
                    csv << "\"Reference\",\"Test\",\"Mean\",\"Weighted median\",\"1st weighted quartile\",\"3rd weighted quartile\",\"Min\",\"Max\",\"Evaluation time\"\n";

                csv << "\"" << referenceFileName.toString() << "\",";
                csv << "\"" << testFileName.toString() << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(mean, 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(weightedMedian, 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(firstWeightedQuartile, 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(thirdWeightedQuartile, 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(minValue, 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(maxValue, 6) << "\",";
                csv << "\"" << FIXED_DECIMAL_DIGITS(time, 4) << "\"\n";

                csv.close();
            }
            else
            {
                std::cout << "\nError: Could not write csv file " << csvFileName.toString() << "\n";
            }
        }

        if (verbosity > 0)
        {
            std::cout << FLIPString << " between reference image <" << referenceFileName.toString() << "> and test image <" << testFileName.toString() << ">\n";
            std::cout << "     Mean: " << FIXED_DECIMAL_DIGITS(mean, 6) << "\n";
            std::cout << ((testFileCount < commandLine.getOptionValues("test").size() && verbosity == 1) ? "\n" : "");
            if (verbosity > 1)
            {
                std::cout << "     Weighted median: " << FIXED_DECIMAL_DIGITS(weightedMedian, 6) << "\n";
                std::cout << "     1st weighted quartile: " << FIXED_DECIMAL_DIGITS(firstWeightedQuartile, 6) << "\n";
                std::cout << "     3rd weighted quartile: " << FIXED_DECIMAL_DIGITS(thirdWeightedQuartile, 6) << "\n";
                std::cout << "     Min: " << FIXED_DECIMAL_DIGITS(minValue, 6) << "\n";
                std::cout << "     Max: " << FIXED_DECIMAL_DIGITS(maxValue, 6) << "\n";
                std::cout << "     Evaluation time: " << FIXED_DECIMAL_DIGITS(time, 4) << " seconds\n";
                if (!commandLine.optionSet("no-error-map"))
                {
                    std::cout << "     FLIP error map location: " << destinationDirectory + "/" + flipFileName.toString() << "\n";
                }
                if (!commandLine.optionSet("no-exposure-map") && useHDR)
                {
                    std::cout << "     FLIP exposure map location: " << destinationDirectory + "/" + exposureFileName.toString() << "\n";
                }
                std::cout << ((testFileCount < commandLine.getOptionValues("test").size()) ? "\n" : "");
            }
        }

        if (commandLine.optionSet("exit-on-test"))
        {
            std::string exitOnTestQuantity = "mean";
            float exitOnTestThresholdValue = 0.05f;
            if (commandLine.optionSet("exit-test-parameters"))
            {
                exitOnTestQuantity = commandLine.getOptionValue("exit-test-parameters", 0);
                std::transform(exitOnTestQuantity.begin(), exitOnTestQuantity.end(), exitOnTestQuantity.begin(), [](unsigned char c) { return std::tolower(c); });
                if (exitOnTestQuantity != "mean" && exitOnTestQuantity != "weighted-median" && exitOnTestQuantity != "max")
                {
                    std::cout << "For --exit-test-parameters / -etp, the first paramter needs to be {MEAN | WEIGHTED-MEDIAN | MAX}\n";
                    exit(EXIT_FAILURE);
                }
                exitOnTestThresholdValue = std::stof(commandLine.getOptionValue("exit-test-parameters", 1));
                if (exitOnTestThresholdValue < 0.0f || exitOnTestThresholdValue > 1.0f)
                {
                    std::cout << "For --exit-test-parameters / -etp, the second paramter needs to be in [0,1]\n";
                    exit(EXIT_FAILURE);
                }
            }

            float testQuantity;
            if (exitOnTestQuantity == "mean")
            {
                testQuantity = mean;
            }
            else if (exitOnTestQuantity == "weighted-median")
            {
                testQuantity = weightedMedian;
            }
            else if (exitOnTestQuantity == "max")
            {
                testQuantity = maxValue;
            }
            else
            {
                std::cout << "Exiting with failure code because exit-on-text-quantity was " << exitOnTestQuantity << ", and expects to be mean, weighted-median, or max.\n";
                exit(EXIT_FAILURE);     // From stdlib.h: equal to 1.
            }

            if (testQuantity > exitOnTestThresholdValue)
            {
                std::cout << "Exiting with failure code because the " << exitOnTestQuantity << " of the FLIP error map is " << FIXED_DECIMAL_DIGITS(testQuantity, 6) << ", while the threshold for success is " << FIXED_DECIMAL_DIGITS(exitOnTestThresholdValue, 6) << ".\n";
                exit(EXIT_FAILURE);     // From stdlib.h: equal to 1.
            }
        }
    }

    int execute(commandline commandLine)
    {
        auto timeStart = std::chrono::high_resolution_clock::now();

        std::string FLIPString = "FLIP";
        int MajorVersion = 1;
        int MinorVersion = 7;

        if (commandLine.optionSet("help"))
        {
            std::cout << "FLIP v" << MajorVersion << "." << MinorVersion << ".\n";
            commandLine.print();
            exit(EXIT_SUCCESS);
        }
        if (!commandLine.optionSet("reference"))
        {
            std::cout << "Error: you need to set a reference image filename.\n  Typically done with '-r refimg.{png,exr}' or '--reference refimg.{png,exr}'.\n  Use -h or --help for help message. Exiting.\n";
            exit(EXIT_FAILURE);
        }
        if (!std::filesystem::exists(commandLine.getOptionValue("reference")))  // Reference does not exist?
        {
            std::cout << "Error: reference file <" << commandLine.getOptionValue("reference") << "> does not exist. Exiting.\n";
            exit(EXIT_FAILURE);
        } 
        if (!commandLine.optionSet("test"))
        {
            std::cout << "Error: you need to set a test image filename.\n  Typically done with '-t testimg.{png,exr}' or '--test testimg.{png,exr}'.\n  Use -h or --help for help message. Exiting.\n";
            exit(EXIT_FAILURE);
        }
        if ((commandLine.optionSet("basename") && commandLine.getOptionValues("test").size() != 1) || commandLine.getError())
        {
            if (commandLine.getError())
            {
                std::cout << commandLine.getErrorString() << "\n";
            }
            std::cout << "FLIP v" << MajorVersion << "." << MinorVersion << ".\n";
            commandLine.print();
            exit(EXIT_FAILURE);
        }

        size_t verbosity = commandLine.optionSet("verbosity") ? std::stoi(commandLine.getOptionValue("verbosity")) : 2;

        FLIP::parameters parameters;
        FLIP::filename referenceFileName(commandLine.getOptionValue("reference"));
        bool bUseHDR = (referenceFileName.getExtension() == "exr");
        std::string destinationDirectory = ".";
        std::string basename = (commandLine.optionSet("basename") ? commandLine.getOptionValue("basename") : "");
        FLIP::filename flipFileName(".png");
        FLIP::filename histogramFileName(".py");
        FLIP::filename exposureFileName(".png");
        FLIP::filename txtFileName(".txt");
        FLIP::filename testFileName;
        bool returnLDRImages = false;                                   // Can only happen for HDR.
        bool returnLDRFLIPImages = false;                               // Can only happen for HDR.

        setupDestinationDirectory(bUseHDR, commandLine, destinationDirectory);
        setupPixelsPerDegree(commandLine, parameters);
        getExposureParameters(bUseHDR, commandLine, parameters, returnLDRFLIPImages, returnLDRImages);

        if (verbosity > 1)
        {
            std::cout << "Invoking " << (bUseHDR ? "HDR" : "LDR") << "-FLIP\n";
            std::cout << "     Pixels per degree: " << int(std::round(parameters.ppd)) << "\n" << (!bUseHDR ? "\n" : "");
        }

        auto referenceImageOpt = ImageHelpers::load(referenceFileName.toString());   // Load reference image.
        if (referenceImageOpt.index() == 0)
        {
            std::cout << "Error: could not read reference image file <" << referenceFileName.toString() << ">. Note that FLIP only loads png, bmp, tga, and exr images. Exiting.\n";
            exit(EXIT_FAILURE);
        }

        using img3 = FLIP::image<FLIP::float3>;
        using img4 = FLIP::image<FLIP::float4>;
        auto parseImage = [bUseHDR](std::variant<std::monostate, img3, img4>& img) {
            if(auto* f3 = std::get_if<img3>(&img)) {
                if(!bUseHDR) f3->srgb_to_linear_rgb();
                return f3->get_dimensions();
            }
            if(auto* f4 = std::get_if<img4>(&img)) {
                if(!bUseHDR) f4->srgb_to_linear_rgb();
                return f4->get_dimensions();
            }

            return FLIP::int3{ 0,0,0 };
        };

        const auto referenceImageDims = parseImage(referenceImageOpt);

        // Save firstTestFileName and firstPooledValue for optional overlapped histogram.
        const bool saveOverlappedHistogram = commandLine.optionSet("histogram") && commandLine.getOptionValues("test").size() == 2;
        FLIP::filename firstTestFileName;
        FLIPPooling::pooling<float> pooledValues;
        FLIPPooling::pooling<float> firstPooledValues;

        uint32_t testFileCount = 0;
        // Loop over the test images files to be FLIP:ed against the reference image.
        for (auto& testFileNameString : commandLine.getOptionValues("test"))
        {
            pooledValues = FLIPPooling::pooling<float>(100); // Reset pooledValues to remove accumulation issues.
            testFileName = testFileNameString;

            if (!std::filesystem::exists(testFileName.toString()))
            {
                std::cout << "Error: test image file <" << testFileName.toString() << "> does not exist. Exiting.\n";
                exit(EXIT_FAILURE);
            }
            
            auto testImageOpt = ImageHelpers::load(testFileName.toString());     // Load test image.
            if (testImageOpt.index() == 0)
            {
                std::cout << "Error: could not read test file <" << testFileName.toString() << ">. Note that FLIP only loads png, bmp, tga, and exr images. Exiting.\n";
                exit(EXIT_FAILURE);
            }
            const auto testImageDims = parseImage(testImageOpt);
            if (referenceImageDims != testImageDims)
            {
                std::cout << "Error: reference <" << referenceImageDims[0] << "x" << referenceImageDims[1] << "> and test <" << testImageDims[0] << "x" << testImageDims[1] << "> images must be of equal dimensions. Exiting.\n";
                exit(EXIT_FAILURE);
            }

            FLIP::image<float> errorMapFLIP(referenceImageDims, 0.0f);
            FLIP::image<float> maxErrorExposureMap(referenceImageDims);

            auto t0 = std::chrono::high_resolution_clock::now();
            if(std::holds_alternative<img3>(referenceImageOpt))
                FLIP::evaluate(std::get<img3>(referenceImageOpt), std::get<img3>(testImageOpt), bUseHDR, parameters, errorMapFLIP, maxErrorExposureMap);
            else
                FLIP::evaluate(std::get<img4>(referenceImageOpt), std::get<img4>(testImageOpt), bUseHDR, parameters, errorMapFLIP, maxErrorExposureMap);
            float time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t0).count() / 1000000.0f;

            saveErrorAndExposureMaps(bUseHDR, commandLine, parameters, basename, errorMapFLIP, maxErrorExposureMap, destinationDirectory, referenceFileName, testFileName, histogramFileName, txtFileName, flipFileName, exposureFileName, verbosity, testFileCount);
            gatherStatisticsAndSaveOutput(commandLine, errorMapFLIP, pooledValues, destinationDirectory, referenceFileName, testFileName, histogramFileName, txtFileName, flipFileName, exposureFileName, FLIPString, time, ++testFileCount, saveOverlappedHistogram, bUseHDR, verbosity);

            // Save first set of results for overlapped histogram.
            if (saveOverlappedHistogram && testFileCount == 1)
            {
                firstPooledValues = pooledValues;
                firstTestFileName = testFileName;
            }
        }

        if (saveOverlappedHistogram)
        {
            bool optionLog = commandLine.optionSet("log");
            bool optionExcludeValues = commandLine.optionSet("exclude-pooled-values");
            float yMax = (commandLine.optionSet("y-max") ? std::stof(commandLine.getOptionValue("y-max")) : 0.0f);
            pooledValues.saveOverlappedHistogram(firstPooledValues, destinationDirectory + "/" + histogramFileName.toString(), referenceImageDims[0], referenceImageDims[1], optionLog, referenceFileName.getName(), firstTestFileName.getName(), testFileName.getName(), !optionExcludeValues, yMax);
        }

        float timeTotal = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timeStart).count() / 1000000.0f;
        std::cout << "\nTotal time: " << FIXED_DECIMAL_DIGITS(timeTotal, 4) << " seconds\n";
        exit(EXIT_SUCCESS);                 // From stdlib.h: equal to 0.
    }
}