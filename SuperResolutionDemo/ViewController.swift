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
    @IBOutlet weak var btnPreprocess: UIButton!
    @IBOutlet weak var btnRun: UIButton!
    @IBOutlet weak var btnNext: UIButton!
    
    private let testImages = ["test1.png", "test2.jpg", "test3.png"]
    private var imgIndex = 0
    private var image: UIImage?
    private var inferencer = SuperResolutionModel()  // 注意：ObjectDetector 类名仍可沿用，但其内部加载的模型已改为超分模型
    private var isPreprocessed = false          // 标记是否已完成预处理
    private var preprocessedInputArray: [Float]? // 缓存预处理后的输入数组
    
    override func viewDidLoad() {
        super.viewDidLoad()
        image = UIImage(named: testImages[imgIndex])!
        if let iv = imageView {
            iv.image = image
            btnRun.setTitle("Run SR", for: .normal)
            btnPreprocess.setTitle("Pseudo IR", for: .normal)
            btnRun.isEnabled = false   // 初始禁用
        }
    }
    
    @IBAction func pseudoIRTapped(_ sender: Any) {
        guard let originalImage = image else { return }
        
        // 1. 执行预处理（复用 UIImage 扩展中的方法）
        guard let inputArray = originalImage.preprocessForSR() else {
            print("预处理失败")
            return
        }
        
        // 2. 将预处理结果显示为灰度图（需要将 UInt8 像素数组转为 UIImage）
        if let grayImage = originalImage.convertToGrayscale(targetSize: CGSize(width: 384, height: 512)) {
            imageView.image = grayImage
        }
        
        // 3. 缓存输入数组并更新状态
        preprocessedInputArray = inputArray
        isPreprocessed = true
        btnRun.isEnabled = true
    }
    
    @IBAction func runTapped(_ sender: Any) {
        guard isPreprocessed, let inputArray = preprocessedInputArray else {
            // 提示用户先点击 Pseudo IR
            let alert = UIAlertController(title: "提示", message: "请先点击「Pseudo IR」进行预处理", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
            return
        }
        
        btnRun.isEnabled = false
        btnRun.setTitle("Running SR...", for: .normal)
        
        // 其余推理代码不变，但使用缓存的 inputArray
        inputArray.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            let pointer = UnsafeMutableRawPointer(mutating: baseAddress)
            
            DispatchQueue.global().async {
                guard let outputNSData = self.inferencer.model.upscale(image: pointer) else {
                    DispatchQueue.main.async {
                        self.btnRun.isEnabled = true
                        self.btnRun.setTitle("Run SR", for: .normal)
                    }
                    return
                }
                // 将 NSData 转换为 Data
                let outputData = outputNSData as Data
                
                let outputImage = PrePostProcessor.outputsToUIImage(
                    data: outputData,
                    width: PrePostProcessor.outputWidth,
                    height: PrePostProcessor.outputHeight		
                )
                
                DispatchQueue.main.async {
                    self.imageView.image = outputImage
                    self.btnRun.isEnabled = true
                    self.btnRun.setTitle("Run SR", for: .normal)
                }
            }
        }
    }
    
    // 添加一个重置状态的方法
    private func resetPreprocessingState() {
        isPreprocessed = false
        preprocessedInputArray = nil
        btnRun.isEnabled = false
        // 可选：恢复显示原图（如果当前显示的是灰度图）
        imageView.image = image
    }

    // 在每个更换图片的地方调用
    @IBAction func nextTapped(_ sender: Any) {
        imgIndex = (imgIndex + 1) % testImages.count
        btnNext.setTitle(String(format: "Test Image %d/%d", imgIndex + 1, testImages.count), for: .normal)
        image = UIImage(named: testImages[imgIndex])!
        resetPreprocessingState()
    }

    @IBAction func photosTapped(_ sender: Any) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true) {
            // 注意：此时尚未获取新图片，重置状态在 delegate 中处理
        }
    }

    @IBAction func cameraTapped(_ sender: Any) {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePickerController = UIImagePickerController()
            imagePickerController.delegate = self
            imagePickerController.sourceType = .camera
            present(imagePickerController, animated: true, completion: nil)
        }
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        image = info[.originalImage] as? UIImage
        resetPreprocessingState()
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
    
    /// 将图片缩放到指定尺寸并转为灰度图（直接返回 UIImage）
    func convertToGrayscale(targetSize: CGSize) -> UIImage? {
        // 1. 缩放到目标尺寸
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        let scaledImage = renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        // 2. 转为灰度图
        guard let cgImage = scaledImage.cgImage else { return nil }
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let grayCGImage = context.makeImage() else { return nil }
        return UIImage(cgImage: grayCGImage)
    }
    
    /// 原有 resized 方法（假设已存在，若没有则需自行实现）
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}


