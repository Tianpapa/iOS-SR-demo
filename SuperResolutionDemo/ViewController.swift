import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var btnPreprocess: UIButton!
    @IBOutlet weak var btnRun: UIButton!
    @IBOutlet weak var btnNext: UIButton!
    
    private let testImages = ["test1.png", "test2.jpg", "test3.png"]
    private var imgIndex = 0
    private var image: UIImage?
    private var inferencer: SRModelBridge?
    private var isPreprocessed = false
    private var preprocessedInputBytes: [UInt8]?   // 改为直接缓存字节
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 1. 加载模型
        guard let modelPath = Bundle.main.path(forResource: "SPAN_tiny_c8_e2217_384x512x2_smp_int8", ofType: "mnn") else {
            fatalError("❌ Model file not found")
        }
        
        let preferredBackend = "CoreML"  // 可选: "CPU", "Metal", "CoreML"
        let threads = 4

        // 显示加载状态（可选）
        btnRun.isEnabled = false
        btnPreprocess.isEnabled = false

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let model = try SRModelBridge(modelPath: modelPath, backend: preferredBackend, threads: threads)
                print("✅ Model initialized with \(preferredBackend) backend")

                DispatchQueue.main.async {
                    self?.inferencer = model
                    self?.setupUIAfterModelLoad()
                }
            } catch {
                print("⚠️ \(preferredBackend) backend failed: \(error.localizedDescription)")
                do {
                    let model = try SRModelBridge(modelPath: modelPath, backend: "CPU", threads: threads)
                    print("✅ Model initialized with CPU backend (fallback)")

                    DispatchQueue.main.async {
                        self?.inferencer = model
                        self?.setupUIAfterModelLoad()
                    }
                } catch {
                    DispatchQueue.main.async {
                        self?.showModelLoadError(error)
                    }
                }
            }
        }
    }

    private func setupUIAfterModelLoad() {
        image = UIImage(named: testImages[imgIndex])!
        imageView.image = image
        btnRun.setTitle("Run SR", for: .normal)
        btnPreprocess.setTitle("Pseudo IR", for: .normal)
        btnPreprocess.isEnabled = true
        // btnRun 仍需等待预处理后才启用
        btnRun.isEnabled = false
    }

    private func showModelLoadError(_ error: Error) {
        let alert = UIAlertController(title: "模型加载失败",
                                      message: error.localizedDescription,
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "退出", style: .default) { _ in
            fatalError("Model init failed")
        })
        present(alert, animated: true)
    }
    
    // MARK: - Actions
    
    @IBAction func pseudoIRTapped(_ sender: Any) {
        guard let originalImage = image else { return }
        
        // 预处理：将 UIImage 转为 384x512 灰度字节数组（0~255）
        guard let grayBytes = originalImage.grayscaleBytes(width: 384, height: 512) else {
            print("预处理失败")
            return
        }
        
        // 显示灰度预览图
        if let grayImage = originalImage.convertToGrayscale(targetSize: CGSize(width: 384, height: 512)) {
            imageView.image = grayImage
        }
        
        // 缓存输入字节
        preprocessedInputBytes = grayBytes
        isPreprocessed = true
        btnRun.isEnabled = true
    }
    
    @IBAction func runTapped(_ sender: Any) {
        guard let inferencer = inferencer else {
            print("Model not ready")
            return
        }
        guard isPreprocessed, let inputBytes = preprocessedInputBytes else {
            showAlert(message: "请先点击「Pseudo IR」进行预处理")
            return
        }
        
        btnRun.isEnabled = false
        btnRun.setTitle("Running SR...", for: .normal)
        
        let inputData = Data(inputBytes)
        let outputSize = inferencer.outputSize
        let mutableOutput = NSMutableData(length: outputSize)!
        
        DispatchQueue.global().async {
            do {
                // 调用桥接推理方法
                try inferencer.processImage(inputData, outputImage: mutableOutput)
                
                let outputData = mutableOutput as Data
                guard let outputImage = self.imageFromSRBytes(outputData,
                                                              width: inferencer.outputWidth,
                                                              height: inferencer.outputHeight) else {
                    throw NSError(domain: "SR", code: -1, userInfo: nil)
                }
                
                DispatchQueue.main.async {
                    self.imageView.image = outputImage
                    self.btnRun.isEnabled = true
                    self.btnRun.setTitle("Run SR", for: .normal)
                }
            } catch {
                DispatchQueue.main.async {
                    self.btnRun.isEnabled = true
                    self.btnRun.setTitle("Run SR", for: .normal)
                    self.showAlert(message: "推理失败: \(error.localizedDescription)")
                }
            }
        }
    }
    
    @IBAction func nextTapped(_ sender: Any) {
        imgIndex = (imgIndex + 1) % testImages.count
        btnNext.setTitle(String(format: "Test Image %d/%d", imgIndex + 1, testImages.count), for: .normal)
        image = UIImage(named: testImages[imgIndex])!
        resetPreprocessingState()
    }
    
    @IBAction func photosTapped(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }
    
    @IBAction func cameraTapped(_ sender: Any) {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else { return }
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .camera
        present(picker, animated: true)
    }
    
    // MARK: - UIImagePickerControllerDelegate
    
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        image = info[.originalImage] as? UIImage
        resetPreprocessingState()
        dismiss(animated: true)
    }
    
    // MARK: - Helpers
    
    private func resetPreprocessingState() {
        isPreprocessed = false
        preprocessedInputBytes = nil
        btnRun.isEnabled = false
        imageView.image = image
    }
    
    private func showAlert(message: String) {
        let alert = UIAlertController(title: "提示", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    /// 将模型输出的单通道字节数组（0~255）转换为 UIImage
    private func imageFromSRBytes(_ data: Data, width: Int, height: Int) -> UIImage? {
        guard data.count == width * height else { return nil }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: data as CFData),
              let cgImage = CGImage(width: width,
                                    height: height,
                                    bitsPerComponent: 8,
                                    bitsPerPixel: 8,
                                    bytesPerRow: width,
                                    space: colorSpace,
                                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                    provider: provider,
                                    decode: nil,
                                    shouldInterpolate: false,
                                    intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}

// MARK: - UIImage 扩展（仅定义一次）
extension UIImage {
    
    /// 缩放图片到指定尺寸
    func resized(to targetSize: CGSize) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
    
    /// 转为灰度图并返回指定尺寸的 UIImage
    func convertToGrayscale(targetSize: CGSize) -> UIImage? {
        guard let resized = self.resized(to: targetSize),
              let cgImage = resized.cgImage else { return nil }
        
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let grayCGImage = context.makeImage() else { return nil }
        return UIImage(cgImage: grayCGImage)
    }
    
    /// 提取缩放后灰度图的字节数组（0~255），用于模型输入
    func grayscaleBytes(width: Int, height: Int) -> [UInt8]? {
        guard let resized = self.resized(to: CGSize(width: width, height: height)),
              let cgImage = resized.cgImage else { return nil }
        
        let bytesPerRow = width
        var pixelData = [UInt8](repeating: 0, count: width * height)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        guard let context = CGContext(data: &pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelData
    }
}
