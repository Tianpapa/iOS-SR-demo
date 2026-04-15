// SuperResolutionModel.cpp
#include "SuperResolutionModel.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ErrorCode.hpp>
#ifdef MNN_METAL
#include <MNN/MNNSharedContext.h>
#endif
#include <chrono>
#include <cstring>
#include <iostream>

SuperResolutionModel::SuperResolutionModel() = default;

SuperResolutionModel::~SuperResolutionModel() {
    releaseSession();
}

void SuperResolutionModel::releaseSession() {
    if (session_ && net_) {
        net_->releaseSession(session_);
        session_ = nullptr;
    }
    hostInput_.reset();
    hostOutput_.reset();
}

bool SuperResolutionModel::init(const std::string& modelPath,
                                MNNForwardType type,
                                int numThreads) {
    std::lock_guard<std::mutex> lock(mutex_);

    net_.reset(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!net_) {
        std::cerr << "[SRModel] Failed to load model from: " << modelPath << std::endl;
        return false;
    }
    std::cout << "[SRModel] Model loaded: " << modelPath << std::endl;

    backendType_ = type;
    numThreads_  = numThreads;

    return createSession();
}

void SuperResolutionModel::setBackend(MNNForwardType type, int numThreads) {
    std::lock_guard<std::mutex> lock(mutex_);
    backendType_ = type;
    numThreads_  = numThreads;
    createSession();
}

bool SuperResolutionModel::createSession() {
    if (!net_) return false;

    releaseSession();

    MNN::ScheduleConfig config;
    config.type      = backendType_;
    config.numThread = numThreads_;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    config.backendConfig = &backendConfig;

    session_ = net_->createSession(config);
    if (!session_) {
        std::cerr << "[SRModel] Failed to create session for backend " << backendType_ << std::endl;
        return false;
    }

    // 获取 MNN 内部张量指针，用于构建 host tensor
    MNN::Tensor* inputTensor  = net_->getSessionInput(session_, nullptr);
    MNN::Tensor* outputTensor = net_->getSessionOutput(session_, nullptr);
    if (!inputTensor || !outputTensor) {
        std::cerr << "[SRModel] Failed to get input/output tensors" << std::endl;
        return false;
    }

    // 预分配 host tensor，复用内存
    hostInput_  = std::make_unique<MNN::Tensor>(inputTensor, MNN::Tensor::CAFFE);
    hostOutput_ = std::make_unique<MNN::Tensor>(outputTensor, MNN::Tensor::CAFFE);

    // 可选维度校验
    auto shape = inputTensor->shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != kChannels ||
        shape[2] != kInputHeight || shape[3] != kInputWidth) {
        std::cerr << "[SRModel] Warning: input tensor shape mismatch. Expected 1x"
                  << kChannels << "x" << kInputHeight << "x" << kInputWidth << std::endl;
    }

    std::cout << "[SRModel] Session created. Backend: " << backendType_
              << ", Threads: " << numThreads_ << std::endl;
    return true;
}

bool SuperResolutionModel::upscaleFloat(const float* inputData, float* outputData) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!isReady()) {
        std::cerr << "[SRModel] Model not ready" << std::endl;
        return false;
    }
    if (!inputData || !outputData) {
        std::cerr << "[SRModel] inputData or outputData is null" << std::endl;
        return false;
    }

    using Clock = std::chrono::high_resolution_clock;
    auto t_total = Clock::now();

    // 1. 输入拷贝：直接使用预分配的 hostInput_
    auto t_input = Clock::now();
    float* hostInPtr = hostInput_->host<float>();
    if (!hostInPtr) return false;
    std::memcpy(hostInPtr, inputData, inputSize() * sizeof(float));
    auto t_input_done = Clock::now();

    // 2. 将 host 数据拷贝到设备
    MNN::Tensor* inputTensor = net_->getSessionInput(session_, nullptr);
    inputTensor->copyFromHostTensor(hostInput_.get());

    // 3. 推理
    auto t_infer = Clock::now();
    if (net_->runSession(session_) != MNN::NO_ERROR) {
        std::cerr << "[SRModel] Inference error" << std::endl;
        return false;
    }
    auto t_infer_done = Clock::now();

    // 4. 输出拷贝：从设备拷出到 hostOutput_
    MNN::Tensor* outputTensor = net_->getSessionOutput(session_, nullptr);
    outputTensor->copyToHostTensor(hostOutput_.get());

    auto t_output = Clock::now();
    float* hostOutPtr = hostOutput_->host<float>();
    if (!hostOutPtr) return false;
    std::memcpy(outputData, hostOutPtr, outputSize() * sizeof(float));
    auto t_output_done = Clock::now();

    auto t_total_done = Clock::now();

    // 计时打印
    using ms = std::chrono::duration<double, std::milli>;
    double input_ms  = std::chrono::duration_cast<ms>(t_input_done - t_input).count();
    double infer_ms  = std::chrono::duration_cast<ms>(t_infer_done - t_infer).count();
    double output_ms = std::chrono::duration_cast<ms>(t_output_done - t_output).count();
    double total_ms  = std::chrono::duration_cast<ms>(t_total_done - t_total).count();

    std::cout << "[SRModel] Input: " << input_ms << " ms, "
              << "Inference: " << infer_ms << " ms, "
              << "Output: " << output_ms << " ms, "
              << "Total: " << total_ms << " ms" << std::endl;

    return true;
}

bool SuperResolutionModel::upscaleImage(const unsigned char* inputImage, unsigned char* outputImage) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!isReady()) {
        std::cerr << "[SRModel] Model not ready" << std::endl;
        return false;
    }
    if (!inputImage || !outputImage) {
        std::cerr << "[SRModel] inputImage or outputImage is null" << std::endl;
        return false;
    }

    using Clock = std::chrono::high_resolution_clock;
    auto t_total_start = Clock::now();

    // ---------- 1. 输入转换：unsigned char -> float ----------
    auto t_conv_in_start = Clock::now();
    float* hostInPtr = hostInput_->host<float>();
    if (!hostInPtr) return false;
    for (size_t i = 0; i < inputSize(); ++i) {
        hostInPtr[i] = static_cast<float>(inputImage[i]);
    }
    auto t_conv_in_end = Clock::now();

    // ---------- 2. 拷贝到设备 ----------
    auto t_copy_in_start = Clock::now();
    MNN::Tensor* inputTensor = net_->getSessionInput(session_, nullptr);
    inputTensor->copyFromHostTensor(hostInput_.get());
    auto t_copy_in_end = Clock::now();

    // ---------- 3. 推理 ----------
    auto t_infer_start = Clock::now();
    if (net_->runSession(session_) != MNN::NO_ERROR) {
        std::cerr << "[SRModel] Inference error" << std::endl;
        return false;
    }
    auto t_infer_end = Clock::now();

    // ---------- 4. 从设备拷出 ----------
    auto t_copy_out_start = Clock::now();
    MNN::Tensor* outputTensor = net_->getSessionOutput(session_, nullptr);
    outputTensor->copyToHostTensor(hostOutput_.get());
    auto t_copy_out_end = Clock::now();

    // ---------- 5. 输出转换：float -> unsigned char ----------
    auto t_conv_out_start = Clock::now();
    float* hostOutPtr = hostOutput_->host<float>();
    if (!hostOutPtr) return false;
    for (size_t i = 0; i < outputSize(); ++i) {
        outputImage[i] = static_cast<unsigned char>(hostOutPtr[i]);
    }
    auto t_conv_out_end = Clock::now();

    auto t_total_end = Clock::now();

    // ---------- 耗时计算（毫秒）----------
    using ms = std::chrono::duration<double, std::milli>;
    double conv_in_ms  = std::chrono::duration_cast<ms>(t_conv_in_end  - t_conv_in_start).count();
    double copy_in_ms  = std::chrono::duration_cast<ms>(t_copy_in_end  - t_copy_in_start).count();
    double infer_ms    = std::chrono::duration_cast<ms>(t_infer_end    - t_infer_start).count();
    double copy_out_ms = std::chrono::duration_cast<ms>(t_copy_out_end - t_copy_out_start).count();
    double conv_out_ms = std::chrono::duration_cast<ms>(t_conv_out_end - t_conv_out_start).count();
    double total_ms    = std::chrono::duration_cast<ms>(t_total_end    - t_total_start).count();

    std::cout << "[SRModel] ConvIn: " << conv_in_ms << " ms, "
              << "CopyIn: " << copy_in_ms << " ms, "
              << "Infer: " << infer_ms << " ms, "
              << "CopyOut: " << copy_out_ms << " ms, "
              << "ConvOut: " << conv_out_ms << " ms, "
              << "Total: " << total_ms << " ms" << std::endl;

    return true;
}
