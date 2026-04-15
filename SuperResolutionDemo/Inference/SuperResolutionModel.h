//
//  SuperResolutionModel.h
//
//  Created by Rick.Liang on 2026/4/10.
//
#pragma once

#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>

// 前向声明 MNN 类型
namespace MNN {
    class Interpreter;
    class Session;
    class Tensor;
}


/**
 * 超分辨率模型推理类（基于 MNN Interpreter）
 *
 * 使用方式：
 *   1. SuperResolutionModel model;
 *   2. model.init("path/to/model.mnn", MNN_FORWARD_NN, 4);
 *   3. model.setBackend(MNN_FORWARD_METAL, 4);
 *   4. std::vector<float> output = model.upscale(inputFloatArray);
 */
class SuperResolutionModel {
public:
    SuperResolutionModel();
    ~SuperResolutionModel();

    // 禁止拷贝
    SuperResolutionModel(const SuperResolutionModel&) = delete;
    SuperResolutionModel& operator=(const SuperResolutionModel&) = delete;

    /**
     * 从 .mnn 文件初始化模型，并直接指定后端与线程数
     * @param modelPath  模型文件路径
     * @param type       MNNForwardType 枚举值（可选: CPU/Metal/CoreML，默认为CoreML）
     * @param numThreads 线程数（默认为4，可设为CPU大核数）
     * @return 成功返回 true
     */
    bool init(const std::string& modelPath,
              MNNForwardType type = MNN_FORWARD_NN,
              int numThreads = 4);
    /**
     * 设置 / 修改 推理后端及线程数
     * @param type       MNNForwardType 枚举值
     * @param numThreads 线程数
     */
    void setBackend(MNNForwardType type, int numThreads);

    /**
     * 执行超分辨率推理，结果写入预分配的缓冲区（零额外分配）
     * @param inputData  输入浮点数组，长度必须为 inputSize()
     * @param outputData 输出缓冲区指针，必须预先分配好 outputSize() 个 float
     * @return 成功返回 true，失败返回 false
     */
    bool upscaleFloat(const float* inputData, float* outputData);

    /**
     * 执行超分辨率推理（字节版本，推荐使用）
     * @param inputImage  输入图像数据（unsigned char 数组，0~255），长度 = inputSize()
     * @param outputImage 输出图像缓冲区（unsigned char 数组，0~255），长度 = outputSize()
     * @return 成功返回 true
     */
    bool upscaleImage(const unsigned char* inputImage, unsigned char* outputImage);
    
    // 获取模型信息
    // 静态尺寸访问函数（编译期求值）
    static constexpr int inputWidth()   { return kInputWidth; }
    static constexpr int inputHeight()  { return kInputHeight; }
    static constexpr int outputWidth()  { return kOutputWidth; }
    static constexpr int outputHeight() { return kOutputHeight; }
    static constexpr int channels()     { return kChannels; }

    // 输入/输出缓冲区大小（浮点数个数）
    static constexpr size_t inputSize()  { return kInputWidth * kInputHeight * kChannels; }
    static constexpr size_t outputSize() { return kOutputWidth * kOutputHeight * kChannels; }
    
    // 状态查询
    bool isReady() const { return session_ != nullptr && hostInput_ != nullptr && hostOutput_ != nullptr; }
    MNNForwardType currentBackend() const { return backendType_; }

private:
    static constexpr int kInputWidth  = 384;
    static constexpr int kInputHeight = 512;
    static constexpr int kOutputWidth  = 768;
    static constexpr int kOutputHeight = 1024;
    static constexpr int kChannels     = 1;
    
    std::unique_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;  // MNN Session 是裸指针，由 Interpreter 管理
    std::unique_ptr<MNN::Tensor> hostInput_;
    std::unique_ptr<MNN::Tensor> hostOutput_;
    // 用于 upscaleImage 的内部临时缓冲区
    std::vector<float> inputFloatBuffer_;
    std::vector<float> outputFloatBuffer_;

    MNNForwardType backendType_ = MNN_FORWARD_NN;
    int numThreads_ = 4;

    mutable std::mutex mutex_;

    void releaseSession();
    bool createSession();
    bool copyInput(const float* data);
    std::vector<float> copyOutput();
};
