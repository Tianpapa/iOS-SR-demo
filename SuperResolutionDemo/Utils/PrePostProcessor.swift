//// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//
//  PrePostProcessor.swift
//  用于超分辨率模型的简单后处理工具
//

import UIKit

class PrePostProcessor : NSObject {
    
    // 超分模型输入输出尺寸（根据你的模型设定）
    static let inputWidth = 384
    static let inputHeight = 512
    static let outputWidth = 768
    static let outputHeight = 1024
    
    // 将模型输出的 [NSNumber] 数组转换为灰度 UIImage
    static func outputsToUIImage(outputs: [NSNumber], width: Int, height: Int) -> UIImage? {
        guard outputs.count == width * height else {
            print("输出数组大小不匹配：期望 \(width*height)，实际 \(outputs.count)")
            return nil
        }
        
        // 将 Float 值（0~1）转换为 UInt8（0~255）
        var pixels = [UInt8](repeating: 0, count: width * height)
        for i in 0..<outputs.count {
            let val = outputs[i].floatValue * 255.0
            pixels[i] = UInt8(max(0, min(255, val)))
        }
        
        // 创建灰度图 CGImage
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bytesPerRow = width
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        
        guard let dataProvider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: dataProvider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        
        return UIImage(cgImage: cgImage)
    }
    
    // 清理之前绘制的图像（如果之前有叠加层，可以保留此方法用于重置）
    static func cleanDetection(imageView: UIImageView) {
        // 对于超分，通常不需要清理，但如果你在 imageView 上叠加了其他视图，可以保留
        // 例如移除所有子视图（如果之前添加过）
        imageView.subviews.forEach { $0.removeFromSuperview() }
        imageView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
    }
}
