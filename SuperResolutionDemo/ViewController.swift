// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//
//  ViewController.swift
//  超分辨率 App
//

import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var btnRun: UIButton!
    @IBOutlet weak var btnNext: UIButton!
    
    private let testImages = ["test1.png", "test2.jpg", "test3.png"]
    private var imgIndex = 0
    private var image: UIImage?
    private var inferencer = SuperResolutionModel()  // 注意：ObjectDetector 类名仍可沿用，但其内部加载的模型已改为超分模型
    
    override func viewDidLoad() {
        super.viewDidLoad()
        image = UIImage(named: testImages[imgIndex])!
        if let iv = imageView {
            iv.image = image
            // 按钮标题改为 "Run SR"
            btnRun.setTitle("Run SR", for: .normal)
        }
    }
    
    @IBAction func runTapped(_ sender: Any) {
        btnRun.isEnabled = false
        btnRun.setTitle("Running SR...", for: .normal)
        
        // 1. 预处理：将 UIImage 转换为模型所需的输入 Float 数组
        guard let inputArray = image?.preprocessForSR() else {
            print("预处理失败")
            btnRun.isEnabled = true
            btnRun.setTitle("Run SR", for: .normal)
            return
        }
        
        // 2. 将 Swift 数组转换为 UnsafeMutableRawPointer 供 ObjC++ 方法使用
        inputArray.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            // 注意：detect(image:) 期望的是 UnsafeMutableRawPointer?，这里使用 UnsafeMutableRawPointer(mutating:)
            let pointer = UnsafeMutableRawPointer(mutating: baseAddress)
            
            DispatchQueue.global().async {
                // 调用模型推理
                guard let outputs = self.inferencer.module.upscale(image: pointer) else {
                    DispatchQueue.main.async {
                        self.btnRun.isEnabled = true
                        self.btnRun.setTitle("Run SR", for: .normal)
                    }
                    return
                }
                
                // 3. 后处理：将输出的 [NSNumber] 数组转换为 UIImage
                let outputImage = PrePostProcessor.outputsToUIImage(
                    outputs: outputs,
                    width: PrePostProcessor.outputWidth,
                    height: PrePostProcessor.outputHeight
                )
                
                DispatchQueue.main.async {
                    // 显示超分结果
                    self.imageView.image = outputImage
                    self.btnRun.isEnabled = true
                    self.btnRun.setTitle("Run SR", for: .normal)
                }
            }
        }
    }
    
    @IBAction func nextTapped(_ sender: Any) {
        // 切换下一张测试图片，清除之前的超分结果（直接设置新图即可）
        imgIndex = (imgIndex + 1) % testImages.count
        btnNext.setTitle(String(format: "Test Image %d/%d", imgIndex + 1, testImages.count), for: .normal)
        image = UIImage(named: testImages[imgIndex])!
        imageView.image = image
    }
    
    @IBAction func photosTapped(_ sender: Any) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true, completion: nil)
    }
    
    @IBAction func cameraTapped(_ sender: Any) {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePickerController = UIImagePickerController()
            imagePickerController.delegate = self
            imagePickerController.sourceType = .camera
            present(imagePickerController, animated: true, completion: nil)
        }
    }
    
    // MARK: - UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        // 获取选中的图片，不做任何缩放，保留原图显示
        image = info[.originalImage] as? UIImage
        imageView.image = image
        dismiss(animated: true, completion: nil)
    }
}

// MARK: - UIImage 扩展，用于超分预处理
extension UIImage {
    /// 将 UIImage 预处理为超分模型所需的 Float 数组（灰度，归一化到 0~1，尺寸 384x512）
    func preprocessForSR() -> [Float]? {
        let targetWidth = 384
        let targetHeight = 512
        
        // 1. 缩放到目标尺寸
        guard let resized = self.resized(to: CGSize(width: targetWidth, height: targetHeight)) else {
            return nil
        }
        
        // 2. 转换为灰度图并提取像素值
        guard let cgImage = resized.cgImage else { return nil }
        
        let width = targetWidth
        let height = targetHeight
        let bytesPerRow = width
        var pixelData = [UInt8](repeating: 0, count: width * height)
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // 3. 转换为 Float 并归一化到 0~1
        return pixelData.map { Float($0) / 255.0 }
    }
    
    /// 原有 resized 方法（假设已存在，若没有则需自行实现）
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}


