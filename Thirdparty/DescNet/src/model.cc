#include "model.h"

using namespace std;

namespace descnet {

Model::Model(const string &engine_path) {
    // Specify our GPU inference configuration options
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    _options.precision = Precision::FP32;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    _options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    _options.optBatchSize = 2000;
    // Specify the maximum batch size we plan on running.
    _options.maxBatchSize = 6000;

    // Set Engine pointer equal to new engine class
    _engine = std::make_unique<Engine<float>>(_options);

    // Normalize values between
    // [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = false;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    // note should be onnx model path
    if (!engine_path.empty()) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = _engine->buildLoadNetwork(engine_path, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    } else {
        throw std::runtime_error("TensorRT engine path is empty.");
    }

    // } else {
    //     // Load the TensorRT engine file directly
    //     bool succ = engine.loadNetwork(arguments.trtModelPath, subVals, divVals, normalize);
    //     if (!succ) {
    //         throw std::runtime_error("Unable to load TensorRT engine.");
    //     }
    // }
}

Model::~Model() {
}

int32_t Model::getBatchSize() const noexcept {
    return _options.optBatchSize;
}

void Model::run(cv::Mat &input, cv::Mat &output){
    // In the following section we populate the input vectors to later pass for inference
    // std::vector<std::vector<cv::cuda::GpuMat>> inputsGpu;
    // std::vector<cv::Mat> inputs;
    cv::Mat inputTensor(cv::Mat({input.size[0], input.size[1], input.size[2], 1}, input.type(), input.data));

    // inputs.emplace_back(cv::Mat({input.size[0], input.size[1], input.size[2], 1}, input.type(), input.data));

    // Let's use a batch size which matches that which we set the
    // Options.optBatchSize option
    // size_t batchSize = _options.optBatchSize;

    // reshape cv matrix to 32x32x1
    // cv::Mat tmp(32, 32, CV_32FC(1), input({cv::Range(0, 1), cv::Range::all(), cv::Range::all()}).data);
    // cv::Mat tmp(32,1, CV_32FC(32), input({cv::Range(0, 1), cv::Range::all(), cv::Range::all()}).data);
    // tmp = tmp.reshape(32,32);
    // cout << "input shape: " << tmp.size << endl;
    cout << "Batch size: " << inputTensor.size[0] << endl;
    cout << "Channels: " << inputTensor.size[3] << endl;
    cout << "rows: " << inputTensor.size[1] << endl;
    cout << "cols: " << inputTensor.size[2] << endl;

    // Populate gpu inputs appropriately.
    // std::vector<cv::cuda::GpuMat> inputGpu;
    // for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
    //     // assign patch to batch j of cv::Mat patch
    //     cv::cuda::GpuMat patch(32, 32, CV_32FC(1), input({cv::Range(j, j+1), cv::Range::all(), cv::Range::all()}).data);
    //     // cv::cuda::GpuMat patch(32, 1, CV_32FC(32), input({cv::Range(j, j+1), cv::Range::all(), cv::Range::all()}).data);
    //     inputGpu.emplace_back(std::move(patch));
    // }
    // inputsGpu.emplace_back(std::move(inputGpu));

    // bool succ = _engine->runInference(inputsGpu, outputVector);
    bool succ = _engine->runInference(inputTensor, output);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    // output = cv::Mat(batchSize, 128, CV_32F);
    // for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...
        // output.row(j) = cv::Mat(1, 128, CV_32F, outputVector[0][j].data());
    // }
}

void Model::runCUDA(const int batchSize, float *gpuPatches, cv::Mat &output){
    // Options.optBatchSize option
    // size_t batchSize = _options.optBatchSize;

    std::vector<int> inputDims = {batchSize, 32, 32, 1};
    bool succ = _engine->runInferenceCUDA(inputDims, gpuPatches, output);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }
}

} // namespace descnet