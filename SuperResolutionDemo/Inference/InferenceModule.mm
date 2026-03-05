// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import "InferenceModule.h"
#import <Libtorch-Lite/Libtorch-Lite.h>
#import <Metal/Metal.h>

#import <AVFoundation/AVFoundation.h>
#import <MNN/HalideRuntime.h>
#import <MNN/MNNDefine.h>
#import <MNN/ErrorCode.hpp>
#import <MNN/ImageProcess.hpp>
#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#define MNN_METAL
#import <MNN/MNNSharedContext.h>
#import <MNN/expr/Module.hpp>
#import <MNN/expr/Executor.hpp>
#import <MNN/expr/ExprCreator.hpp>
#import <MNN/AutoTime.hpp>

struct PretreatInfo {
    int outputSize[4];
    float mean[4];
    float normal[4];
    float inputSize[4];
    float matrix[16];
};

struct GpuCache {
    CVMetalTextureCacheRef _textureCache;
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _pretreat;
    id<MTLFunction> _function;
    id<MTLBuffer> _constant;
    id<MTLCommandQueue> _queue;
    GpuCache() {
        _device = MTLCreateSystemDefaultDevice();
        CVReturn res = CVMetalTextureCacheCreate(nil, nil, _device, nil, &_textureCache);
        FUNC_PRINT(res);
        id<MTLLibrary> library = [_device newDefaultLibrary];
        _function = [library newFunctionWithName:@"pretreat"];
        NSError* error = nil;
        _pretreat = [_device newComputePipelineStateWithFunction:_function error:&error];
        _constant = [_device newBufferWithLength:sizeof(PretreatInfo) options:MTLCPUCacheModeDefaultCache];
        _queue = [_device newCommandQueue];
    }
    ~ GpuCache() {
        
    }
};

// Objective-C 类扩展，声明私有属性（非 C++ 类型）
@interface SRModel () {
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> _rtmgr;
    std::shared_ptr<MNN::Express::Module> _module;
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session *_session;
    std::mutex _mutex;
    MNNForwardType _type;
    int _threads;
    MNN::Tensor* _input;
    MNN::Tensor* _output;
    std::shared_ptr<GpuCache> _cache;
    
}
@end

@implementation SRModel
- (nullable instancetype)initMNN {
    self = [super init];
    if (self) {
        try {
            //        NSString *labels  = [[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"];
            //        NSString *lines   = [NSString stringWithContentsOfFile:labels encoding:NSUTF8StringEncoding error:nil];
            //        self.labels       = [lines componentsSeparatedByString:@"\n"];
            //        self.defaultImage = [UIImage imageNamed:@"testcat.jpg"];
            
            NSString *model = [[NSBundle mainBundle] pathForResource:@"SPAN_x2_c32_e4495_384x512" ofType:@"mnn"];
            _net            = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.UTF8String));
            NSLog(@"Successfully load model from %s", model.UTF8String);
            
            // ★ 在这里调用 setType 设置默认后端和线程数
            _type = MNN_FORWARD_NN;
            _threads = 4;
            [self setType:_type threads:_threads];
            NSLog(@"Successfully init backend with type: %d", _type);
            
            
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (void)setType:(MNNForwardType)type threads:(NSUInteger)threads {
    std::unique_lock<std::mutex> _l(_mutex);
    if (_session) {
        _net->releaseSession(_session);
    }
//    if (nullptr == _cache) {
//        _cache.reset(new GpuCache);
//    }
    MNN::ScheduleConfig config;
    config.type      = type;
    config.numThread = (int)threads;
    if (type == MNN_FORWARD_METAL) {
//        MNN::BackendConfig bnConfig;
//        MNNMetalSharedContext context;
//        context.device = _cache->_device;
//        context.queue = _cache->_queue;
//        bnConfig.sharedContext = &context;
//        config.backendConfig = &bnConfig;
        _session = _net->createSession(config);
    } else if (type == MNN_FORWARD_NN) {
        _session = _net->createSession(config);
    } else {
        _session = _net->createSession(config);
    }
    _input = _net->getSessionInput(_session, nullptr);
    _output = _net->getSessionOutput(_session, nullptr);
    _type = type;
}

- (nullable NSArray<NSNumber*>*)upscaleImage:(void*)imageBuffer {
    // 使用 C++ 的 lock_guard 加锁
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_session || !_input || !_output) {
        NSLog(@"Session or input/output not ready");
        return nil;
    }

    // 记录总开始时间
    CFTimeInterval totalStart = CACurrentMediaTime();
    CFTimeInterval inputStart, inputEnd, inferenceStart, inferenceEnd, outputStart, outputEnd;

    try {
        // ---- 输入处理 ----
        inputStart = CACurrentMediaTime();
        int inputSize = _input->elementSize();
        MNN::Tensor hostInput(_input, MNN::Tensor::CAFFE);
        float* hostInputData = hostInput.host<float>();
        if (!hostInputData) {
            NSLog(@"Failed to get host input data");
            return nil;
        }
        memcpy(hostInputData, imageBuffer, inputSize * sizeof(float));
        _input->copyFromHostTensor(&hostInput);
        inputEnd = CACurrentMediaTime();

        // ---- 推理 ----
        inferenceStart = CACurrentMediaTime();
        _net->runSession(_session);
        inferenceEnd = CACurrentMediaTime();

        // ---- 输出处理 ----
        outputStart = CACurrentMediaTime();
        MNN::Tensor hostOutput(_output, MNN::Tensor::CAFFE);
        _output->copyToHostTensor(&hostOutput);
        float* outputData = hostOutput.host<float>();
        if (!outputData) {
            NSLog(@"Failed to get host output data");
            return nil;
        }

        // 转换为 NSArray
        int outputSize = hostOutput.elementSize();
        NSMutableArray* results = [NSMutableArray arrayWithCapacity:outputSize];
        for (int i = 0; i < outputSize; i++) {
            [results addObject:@(outputData[i])];
        }
        outputEnd = CACurrentMediaTime();

        CFTimeInterval totalEnd = CACurrentMediaTime();

        // 计算耗时（毫秒）
        double inputTime = (inputEnd - inputStart) * 1000.0;
        double inferenceTime = (inferenceEnd - inferenceStart) * 1000.0;
        double outputTime = (outputEnd - outputStart) * 1000.0;
        double totalTime = (totalEnd - totalStart) * 1000.0;

        // 在一行输出
        NSLog(@"[Performance] input: %.3f ms, inference: %.3f ms, output: %.3f ms, total: %.3f ms",
              inputTime, inferenceTime, outputTime, totalTime);

        return [results copy];

    } catch (const std::exception& exception) {
        NSLog(@"C++ exception in upscaleImage: %s", exception.what());
        return nil;
    } catch (...) {
        NSLog(@"Unknown exception in upscaleImage");
        return nil;
    }
}

@end


// 640x640 is the default image size used in the export.py in the yolov5 repo to export the TorchScript model, 25200*85 is the model output size
// const int input_width = 640;
// const int input_height = 640;
// const int output_size = 25200*85;

// 超分模型尺寸
const int input_width = 384;
const int input_height = 512;
const int output_width = 768;
const int output_height = 1024;
const int output_size = output_width * output_height;  // 786432


@implementation InferenceModule {
    @protected torch::jit::mobile::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
    self = [super init];
    if (self) {
        try {
            _impl = torch::jit::_load_for_mobile(filePath.UTF8String);
        } catch (const std::exception& exception) {
            NSLog(@"%s", exception.what());
            return nil;
        }
    }
    return self;
}

- (NSArray<NSNumber*>*)detectImage:(void*)imageBuffer {
    try {
        at::Tensor tensor = torch::from_blob(imageBuffer, { 1, 3, input_height, input_width }, at::kFloat);

        c10::InferenceMode guard;
        CFTimeInterval startTime = CACurrentMediaTime();
        auto outputTuple = _impl.forward({ tensor }).toTuple();
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);

        auto outputTensor = outputTuple->elements()[0].toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }
        
        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < output_size; i++) {
          [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
    }
    return nil;
}

- (NSArray<NSNumber*>*)upscaleImage:(void*)imageBuffer {
    CFTimeInterval totalStart = CACurrentMediaTime();
    CFTimeInterval inputStart, inputEnd, inferenceStart, inferenceEnd, outputStart, outputEnd;

    try {
        // ---- 输入处理 ----
        inputStart = CACurrentMediaTime();
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 1, input_height, input_width}, at::kFloat);
        inputEnd = CACurrentMediaTime();

        c10::InferenceMode guard;

        // ---- 推理 ----
        inferenceStart = CACurrentMediaTime();
        auto outputIValue = _impl.forward({ tensor });
        inferenceEnd = CACurrentMediaTime();

        // ---- 输出处理 ----
        outputStart = CACurrentMediaTime();
        auto outputTensor = outputIValue.toTensor();
        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) return nil;

        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < output_size; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        outputEnd = CACurrentMediaTime();

        CFTimeInterval totalEnd = CACurrentMediaTime();

        // 计算耗时（毫秒）
        double inputTime = (inputEnd - inputStart) * 1000.0;
        double inferenceTime = (inferenceEnd - inferenceStart) * 1000.0;
        double outputTime = (outputEnd - outputStart) * 1000.0;
        double totalTime = (totalEnd - totalStart) * 1000.0;

        NSLog(@"[Performance LibTorch] input: %.3f ms, inference: %.3f ms, output: %.3f ms, total: %.3f ms",
              inputTime, inferenceTime, outputTime, totalTime);

        return [results copy];

    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
        return nil;
    }
}

@end
