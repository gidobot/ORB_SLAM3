/**
 * @file
 * @brief The ContextDesc class wraps the ContextDesc Tensorflow network for feature description.
 */

// #pragma once

#ifndef DESCMODEL_H
#define DESCMODEL_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>

#include "engine.h" // tensorrt-cpp-api

using namespace std;

namespace descnet {

class Model
{
public:
	// Create trt engine from engine path
	Model(const string &engine_path);

	Model() = delete;

	~Model();

	void run(cv::Mat &input, cv::Mat &output);
	void runCUDA(const int batchSize, float *gpuPatches, cv::Mat &output);

	int32_t getBatchSize() const noexcept;

private:
    std::unique_ptr<Engine<float>> _engine;
    Options _options;
};

} // namespace descnet

#endif // DESCMODEL_H