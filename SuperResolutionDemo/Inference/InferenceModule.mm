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

@interface Model : NSObject {
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> _rtmgr;
    std::shared_ptr<MNN::Express::Module> _module;
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session *_session;
    std::mutex _mutex;
    MNNForwardType _type;
    MNN::Tensor* _input;
    MNN::Tensor* _output;
    std::shared_ptr<GpuCache> _cache;
}
@property (strong, nonatomic) UIImage *defaultImage;
@property (strong, nonatomic) NSArray<NSString *> *labels;

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
    try {
        // 输入张量：形状 {1, 1, input_height, input_width}
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 1, input_height, input_width}, at::kFloat);

        c10::InferenceMode guard;
        CFTimeInterval startTime = CACurrentMediaTime();
        
        // 前向传播，返回 IValue（此处应为 Tensor 类型）
        auto outputIValue = _impl.forward({ tensor });
        
        CFTimeInterval elapsedTime = CACurrentMediaTime() - startTime;
        NSLog(@"inference time:%f", elapsedTime);

        // 直接转换为 Tensor
        auto outputTensor = outputIValue.toTensor();

        float* floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) return nil;

        NSMutableArray* results = [[NSMutableArray alloc] init];
        for (int i = 0; i < output_size; i++) {
            [results addObject:@(floatBuffer[i])];
        }
        return [results copy];
        
    } catch (const std::exception& exception) {
        NSLog(@"%s", exception.what());
        return nil;
    }
}

@end
