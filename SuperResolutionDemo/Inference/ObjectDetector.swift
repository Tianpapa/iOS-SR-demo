// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import UIKit

class ObjectDetector {
    
    lazy var classes: [String] = {
        if let filePath = Bundle.main.path(forResource: "classes", ofType: "txt"),
            let classes = try? String(contentsOfFile: filePath) {
            return classes.components(separatedBy: .newlines)
        } else {
            fatalError("classes file was not found.")
        }
    }()
}

class SuperResolutionModel {
    lazy var model: SRModel = {
        if let model = SRModel() {
            return model
        } else {
            fatalError("Failed to load model!")
        }
    }()
}
