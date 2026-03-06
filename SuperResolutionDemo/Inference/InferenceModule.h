// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface InferenceModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)detectImage:(void*)imageBuffer NS_SWIFT_NAME(detect(image:));
- (nullable NSArray<NSNumber*>*)upscaleImage:(void*)imageBuffer NS_SWIFT_NAME(upscale(image:));

@end


@interface SRModel : NSObject

- (nullable instancetype)initMNN
    NS_SWIFT_NAME(init())NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSData *)upscaleImage:(void*)imageBuffer NS_SWIFT_NAME(upscale(image:));

@end

NS_ASSUME_NONNULL_END
