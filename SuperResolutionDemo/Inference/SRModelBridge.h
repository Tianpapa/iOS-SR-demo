//
//  SRModelBridge.h
//  SuperResolutionDemo
//
//  Created by Rick.Liang on 2026/4/13.
//
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SRModelBridge : NSObject

/// 模型尺寸（只读）
@property (nonatomic, readonly) NSInteger inputWidth;
@property (nonatomic, readonly) NSInteger inputHeight;
@property (nonatomic, readonly) NSInteger outputWidth;
@property (nonatomic, readonly) NSInteger outputHeight;
@property (nonatomic, readonly) NSInteger channels;
@property (nonatomic, readonly) NSInteger inputSize;   // 字节数
@property (nonatomic, readonly) NSInteger outputSize;  // 字节数

/// 初始化模型
/// @param modelPath .mnn 文件路径
/// @param backend 后端名称 ("CPU", "Metal", "OpenCL")
/// @param threads 线程数
/// @param error 错误信息
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   backend:(NSString *)backend
                                   threads:(NSInteger)threads
                                     error:(NSError **)error;

/// 执行超分推理（字节版本）
/// @param inputImageData 输入图像数据 (unsigned char, 长度 = inputSize)
/// @param outputImageData 输出图像缓冲区 (长度 = outputSize，调用前必须分配好)
/// @param error 错误信息
/// @return 成功返回 YES
- (BOOL)processImage:(NSData *)inputImageData
         outputImage:(NSMutableData *)outputImageData
               error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
