//
//  VisionObjectRecognitionViewController.swift
//  ImVisible
//
//  Created by Samuel Yu on 12/19/19.
//  Copyright © 2019 Samuel Yu. All rights reserved.
//

import UIKit
import AVFoundation
import Vision
import simd
import AudioToolbox

class VisionObjectRecognitionViewController: ViewController {
    var translate_matrix = simd_double3x3([simd_double3(x: -1.17079727*pow(10,-1), y: -9.02276490*pow(10, -16), z: -2.75975270*pow(10, -19)),simd_double3(x: -1.56391162, y: -2.59783431, z: -7.75749810*pow(10,-4)),simd_double3(x: 2.25203273*pow(10, 3), y: 3.71606050*pow(10, 3), z: 1.0)])
    var prob_store: [[Double]] = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    var class_labels: [String] = ["red", "green", "yellow", "none"]
    var class_colors: [UIColor] = [UIColor.red, UIColor.green, UIColor.yellow, UIColor.lightGray]
    var direc_store: [[Double]] = [[],[],[],[],[]]
    var direc_store_avg: [Double] = [0,0,0,0]
    var prob_store_avg: [Double] = [0,0,0,0]
    var counter = 0
    var start = CFAbsoluteTimeGetCurrent()
    var temp = 0.0
    var last_beep = 0.0
    var last_start = 0.0
    var last_angle = 0.0
    var last_instruction = 0.0
    var previous_decision = ""
    var label_constant = 0
    var pos_good = false
    var angle_good = false
    var informed_pos = false
    var informed_angle = false
    let label: UILabel = {
        let label = UILabel()
        label.font = label.font.withSize(24)
        label.isHighlighted = true
        label.translatesAutoresizingMaskIntoConstraints = false
        label.attributedText = NSAttributedString(string: "Label", attributes: [ NSAttributedString.Key.foregroundColor: UIColor.white, NSAttributedString.Key.backgroundColor: UIColor.yellow])
        return label
    }()
    
    let time_label: UILabel = {
        let label = UILabel()
        label.font = label.font.withSize(24)
        label.isHighlighted = true
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "0.0"
        return label
    }()
    
    let fps_label: UILabel = {
        let label = UILabel()
        label.font = label.font.withSize(24)
        label.isHighlighted = true
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "0.0"
        return label
    }()
    
    let five_frame_average: UILabel = {
        let label = UILabel()
        label.font = label.font.withSize(24)
        label.isHighlighted = true
        label.translatesAutoresizingMaskIntoConstraints = false
        label.attributedText = NSAttributedString(string: "Label", attributes: [ NSAttributedString.Key.foregroundColor: UIColor.white, NSAttributedString.Key.backgroundColor: UIColor.yellow])
        return label
    }()
    
    let angle_decision: UILabel = {
        let label = UILabel()
        label.font = label.font.withSize(24)
        label.isHighlighted = true
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "good orientation"
        return label
    }()
    
    let start_decision: UILabel = {
        let label = UILabel()
        label.font = label.font.withSize(24)
        label.isHighlighted = true
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "Good Start"
        return label
    }()
    
    
    private var detectionOverlay: CALayer! = nil
    
    // Vision parts
    private var requests = [VNRequest]()
    
    @discardableResult
    func setupVision() -> NSError? {
        // Setup Vision parts
        let error: NSError! = nil
        
        do {
            let visionModel = try VNCoreMLModel(for: LytNetV1().model)
            let request = VNCoreMLRequest(model: visionModel, completionHandler: { (finishedRequest, error) in
                let output = finishedRequest.results as! [VNCoreMLFeatureValueObservation]
                let direction = output[1].featureValue.multiArrayValue
                let classes = output[0].featureValue.multiArrayValue
                let classes_simplified: [Double] = [classes![0].doubleValue, classes![1].doubleValue, classes![2].doubleValue + classes![3].doubleValue, classes![4].doubleValue]
                let coords = [direction![0].doubleValue, direction![1].doubleValue, direction![2].doubleValue, direction![3].doubleValue]
                
                let point1_ = self.translate_matrix*simd_double3(x:self.direc_store_avg[0]*4032,y:self.direc_store_avg[1]*3024,z:1)
                let point2_ = self.translate_matrix*simd_double3(x:self.direc_store_avg[2]*4032,y:self.direc_store_avg[3]*3024,z:1)
                var point1__: [Double] = [point1_[0]/point1_[2], point1_[1]/point1_[2]]
                let point2__: [Double] = [point2_[0]/point2_[2], point2_[1]/point2_[2]]
                let slope = (point2__[1]-point1__[1])/(point1__[0]-point2__[0])
                var angle = atan(slope)*180/Double.pi
                point1__[1] = 3024-point1__[1]
                let start = (0-point1__[1])/slope + point1__[0]
                let transposed_coords = [point1__[0]/4032,(3024-point1__[1])/3024,point2__[0]/4032, point2__[1]/3024]
     
                self.prob_store[self.counter%5] = classes_simplified
                self.direc_store[self.counter%5] = coords

                var angle_text = "good orientation"
                var start_text = "good startpoint"
                if(start > 2166){
                    start_text = "move right"
                }
                else if(start < 1866){
                    start_text = "move left"
                }
                if(angle < 0.0){
                    angle = 180 + angle
                }
                if(90.0-angle > 10.0){
                    angle_text = "turn right"
                }
                else if(90.0-angle < -10.0){
                    angle_text = "turn left"
                }
                
                var index = 0
                for i in 1...3{
                    if classes_simplified[i] > classes_simplified[index]{
                        index = i
                    }
                }
                
                let output_class = self.class_labels[index]
                let color = self.class_colors[index]
                
                if self.counter%5==4{
                    self.prob_store_avg = [0,0,0,0]
                    self.direc_store_avg = [0,0,0,0]
                    for i in 0...4{
                        for j in 0...3{
                            self.prob_store_avg[j] += self.prob_store[i][j]
                        }
                    }
                    for i in 0...3{
                        self.direc_store_avg[i] += (self.direc_store[0][i]+self.direc_store[1][i]+self.direc_store[2][i]+self.direc_store[3][i]+self.direc_store[4][i])/5
                    }
                    for k in 0...3{
                        self.prob_store_avg[k] = self.prob_store_avg[k]/5
                    }
                }
                
                var index_avg = 0
                for i in 1...3{
                    if self.prob_store_avg[i] > self.prob_store_avg[index_avg]{
                        index_avg = i
                    }
                }
                var output_class_avg = self.class_labels[index_avg]
                var color_avg = self.class_colors[index_avg]
                if self.prob_store_avg[index_avg] < 0.8{
                    output_class_avg = "no decision"
                    color_avg = UIColor.darkGray
                }

                let timeElapsed = CFAbsoluteTimeGetCurrent() - self.start
                var time = timeElapsed-self.temp
                time = round(1000*time)
                let fps = String(round(100/time*10000)/1000) + "fps"
                let time_string = String(time) + "ms"
                
                self.temp = timeElapsed
                self.counter += 1
                
                DispatchQueue.main.async(execute: {
                    self.drawVisionRequestResults(label: output_class, points: coords, color: color, time: time_string, fps: fps, color2: color_avg, label2: output_class_avg, points2: self.direc_store_avg, angle: angle_text, start: start_text, transposed: transposed_coords)
                    })
            })
            request.imageCropAndScaleOption = .centerCrop
            self.requests = [request]
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
        return error
    }
    func drawVisionRequestResults(label: String, points: [Double], color: UIColor, time: String, fps: String, color2: UIColor, label2: String, points2: [Double], angle: String, start: String, transposed: [Double]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil
        
        let shapeLayer = self.createLine(points: points, color: UIColor.blue)
        self.fps_label.text = fps
        self.time_label.text = time
        self.angle_decision.text = angle
        self.start_decision.text = start
        self.label.attributedText = NSAttributedString(string: label, attributes: [ NSAttributedString.Key.foregroundColor: UIColor.black, NSAttributedString.Key.backgroundColor: color])
        self.five_frame_average.attributedText = NSAttributedString(string: label2, attributes: [ NSAttributedString.Key.foregroundColor: UIColor.black, NSAttributedString.Key.backgroundColor: color2])
        detectionOverlay.addSublayer(shapeLayer)
        let shapeLayer2 = self.createLine(points: points2, color: UIColor.red)
        detectionOverlay.addSublayer(shapeLayer2)
        self.updateLayerGeometry()
        CATransaction.commit()
    }
    
    override func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let exifOrientation = exifOrientationFromDeviceOrientation()
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: exifOrientation, options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
    
    override func setupAVCapture() {
        
        super.setupAVCapture()

        setupLayers()
        updateLayerGeometry()
        setupVision()
        
        if #available(iOS 13.0, *) {
            let blurEffectView = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterialDark))
            blurEffectView.frame = CGRect(x: 0, y: 0, width: self.view.bounds.width, height: self.view.bounds.height/8)
            view.addSubview(blurEffectView)
            let blurEffectView2 = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterialDark))
            blurEffectView2.frame = CGRect(x: 0, y: self.view.bounds.height*7/8, width: self.view.bounds.width, height: self.view.bounds.height/8)
            view.addSubview(blurEffectView2)
        } else {
            // Fallback on earlier versions
            print("old ios")
            let blurEffectView = UIVisualEffectView(effect: UIBlurEffect(style: .dark))
            blurEffectView.frame = CGRect(x: 0, y: 0, width: self.view.bounds.width, height: self.view.bounds.height/8)
            view.addSubview(blurEffectView)
            let blurEffectView2 = UIVisualEffectView(effect: UIBlurEffect(style: .dark))
            blurEffectView2.frame = CGRect(x: 0, y: self.view.bounds.height*7/8, width: self.view.bounds.width, height: self.view.bounds.height/8)
            view.addSubview(blurEffectView2)
        }

        view.addSubview(label)
        view.addSubview(fps_label)
        view.addSubview(time_label)
        view.addSubview(five_frame_average)
        view.addSubview(angle_decision)
        view.addSubview(start_decision)
        setupLabel()

        startCaptureSession()
        
    }
    
    func setupLayers() {
        detectionOverlay = CALayer()
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: bufferSize.width,
                                         height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionOverlay)
    }
    
    func updateLayerGeometry() {
        let bounds = rootLayer.bounds
        var scale: CGFloat
        
        let xScale: CGFloat = bounds.size.width / bufferSize.height
        let yScale: CGFloat = bounds.size.height / bufferSize.width
        scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)

        detectionOverlay.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale))

        detectionOverlay.position = CGPoint (x: bounds.midX, y: bounds.midY)
        
        CATransaction.commit()
        
    }
    
    func createLine(points: [Double], color: UIColor) -> CALayer {
        let shapeLayer = CALayer()
        let line = CAShapeLayer()
        let linePath = UIBezierPath()
        shapeLayer.name = "Direction Vector"
        linePath.move(to: CGPoint(x:points[2]*960+160, y:720.0-points[3]*720))
        linePath.addLine(to: CGPoint(x:points[0]*960+160,y:720.0-points[1]*720))
        line.path = linePath.cgPath
        line.fillColor = nil
        line.opacity = 1.0
        line.lineWidth = 5.0
        line.strokeColor = color.cgColor
        shapeLayer.addSublayer(line)
        return shapeLayer
    }

    func setupLabel() {
        label.centerXAnchor.constraint(equalTo: view.centerXAnchor, constant: -150).isActive = true
        label.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -210).isActive = true
        label.transform = CGAffineTransform.init(rotationAngle: CGFloat(Double.pi / 2))
        time_label.centerXAnchor.constraint(equalTo: view.centerXAnchor, constant: -80).isActive = true
        time_label.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -210).isActive = true
        time_label.transform = CGAffineTransform.init(rotationAngle: CGFloat(Double.pi / 2))
        fps_label.centerXAnchor.constraint(equalTo: view.centerXAnchor, constant: -115).isActive = true
        fps_label.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -210).isActive = true
        fps_label.transform = CGAffineTransform.init(rotationAngle: CGFloat(Double.pi / 2))
        five_frame_average.centerXAnchor.constraint(equalTo: view.centerXAnchor, constant:-45).isActive = true
        five_frame_average.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -210).isActive = true
        five_frame_average.transform = CGAffineTransform.init(rotationAngle: CGFloat(Double.pi / 2))
        angle_decision.centerXAnchor.constraint(equalTo: view.centerXAnchor, constant: -15).isActive = true
        angle_decision.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -210).isActive = true
        angle_decision.transform = CGAffineTransform.init(rotationAngle: CGFloat(Double.pi / 2))
        start_decision.centerXAnchor.constraint(equalTo: view.centerXAnchor, constant: 20).isActive = true
        start_decision.centerYAnchor.constraint(equalTo: view.centerYAnchor, constant: -210).isActive = true
        start_decision.transform = CGAffineTransform.init(rotationAngle: CGFloat(Double.pi / 2))
    }
}

