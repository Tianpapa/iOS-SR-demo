//
//  SRModelBridge.mm
//  SuperResolutionDemo
//
//  Created by Rick.Liang on 2026/4/13.
//
#import "SRModelBridge.h"
#import "SuperResolutionModel.h"

@interface SRModelBridge () {
    std::unique_ptr<SuperResolutionModel> _model;
}
@end

@implementation SRModelBridge

- (instancetype)initWithModelPath:(NSString *)modelPath
                          backend:(NSString *)backend
                          threads:(NSInteger)threads
                            error:(NSError **)error {
    self = [super init];
    if (self) {
        // 映射后端
        MNNForwardType type = MNN_FORWARD_NN;
        NSString *lower = backend.lowercaseString;
        if ([lower isEqualToString:@"cpu"]) {
            type = MNN_FORWARD_CPU;
        } else if ([lower isEqualToString:@"metal"]) {
            type = MNN_FORWARD_METAL;
        } else if ([lower isEqualToString:@"coreml"]) {
            type = MNN_FORWARD_NN;
        }
        
        _model = std::make_unique<SuperResolutionModel>();
        if (!_model->init(modelPath.UTF8String, type, (int)threads)) {
            if (error) {
                *error = [NSError errorWithDomain:@"SRModelBridge"
                                             code:-1
                                         userInfo:@{NSLocalizedDescriptionKey: @"Failed to initialize model"}];
            }
            return nil;
        }
    }
    return self;
}

- (BOOL)processImage:(NSData *)inputImageData
         outputImage:(NSMutableData *)outputImageData
               error:(NSError **)error {
    // 长度校验
    if (inputImageData.length != self.inputSize) {
        if (error) {
            *error = [NSError errorWithDomain:@"SRModelBridge"
                                         code:-2
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                [NSString stringWithFormat:@"Input size mismatch. Expected %ld bytes, got %lu",
                                                 (long)self.inputSize, (unsigned long)inputImageData.length]}];
        }
        return NO;
    }
    if (outputImageData.length != self.outputSize) {
        if (error) {
            *error = [NSError errorWithDomain:@"SRModelBridge"
                                         code:-3
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                [NSString stringWithFormat:@"Output size mismatch. Expected %ld bytes, got %lu",
                                                 (long)self.outputSize, (unsigned long)outputImageData.length]}];
        }
        return NO;
    }

    const unsigned char *inBytes = (const unsigned char *)inputImageData.bytes;
    unsigned char *outBytes = (unsigned char *)outputImageData.mutableBytes;

    if (!_model->upscaleImage(inBytes, outBytes)) {
        if (error) {
            *error = [NSError errorWithDomain:@"SRModelBridge"
                                         code:-4
                                     userInfo:@{NSLocalizedDescriptionKey: @"Inference failed"}];
        }
        return NO;
    }
    return YES;
}

#pragma mark - Dimensions

- (NSInteger)inputWidth   { return SuperResolutionModel::inputWidth(); }
- (NSInteger)inputHeight  { return SuperResolutionModel::inputHeight(); }
- (NSInteger)outputWidth  { return SuperResolutionModel::outputWidth(); }
- (NSInteger)outputHeight { return SuperResolutionModel::outputHeight(); }
- (NSInteger)channels     { return SuperResolutionModel::channels(); }
- (NSInteger)inputSize    { return (NSInteger)SuperResolutionModel::inputSize(); }
- (NSInteger)outputSize   { return (NSInteger)SuperResolutionModel::outputSize(); }

@end
