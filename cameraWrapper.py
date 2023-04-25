import pyrealsense2 as rs
import numpy as np
import cv2

TARGET_WIDTH  = 640
TARGET_HEIGHT = 480
TARGET_FPS    = 6

class CameraWrapper:
    
    def __init__(self):
        self.camera = getCamera()
        
        
        if not self.camera:  
            raise Exception("Could not find any RealSense Device")
        
        self.initCamera()
        
    def initCamera(self):
            
        self.config      = rs.config()
        self.pipe        = rs.pipeline()  
        self.colorizer   = rs.colorizer(3)
        self.align       = rs.align(rs.stream.color)
        
        self.config.enable_stream(rs.stream.depth, TARGET_WIDTH, TARGET_HEIGHT, rs.format.z16,  TARGET_FPS)
        self.config.enable_stream(rs.stream.color, TARGET_WIDTH, TARGET_HEIGHT, rs.format.bgr8, TARGET_FPS)

        self.pipe.start(self.config)

        self.detector = cv2.FaceDetectorYN.create("Models/face_detection_yunet_2022mar.onnx", "", (320, 320))
        self.detector.setInputSize((TARGET_WIDTH, TARGET_HEIGHT))
            
        rgb_sensor   = None
        depth_sensor = None

        for s in self.camera.sensors:                              
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                rgb_sensor = s 
            if s.get_info(rs.camera_info.name) == 'Stereo Module':
                depth_sensor = s

        if not rgb_sensor:
            print("No rgb sensor")
            return
        
        if not depth_sensor:    
            print("No depth sensor")
            return
        
    def getNextFrames(self, enableAnonymization):
        frameset = self.pipe.wait_for_frames()                  
        
        aligned_frames = self.align.process(frameset)
        color_frame    = aligned_frames.first(rs.stream.color)
        depth_frame    = aligned_frames.get_depth_frame()

        # depth_frame = rs.decimation_filter(1).process(depth_frame)
        # depth_frame = rs.disparity_transform(True).process(depth_frame)
        # depth_frame = rs.spatial_filter().process(depth_frame)
        # depth_frame = rs.temporal_filter().process(depth_frame)
        # depth_frame = rs.disparity_transform(False).process(depth_frame)
        # depth_frame = rs.hole_filling_filter().process(depth_frame)
        
        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if enableAnonymization:
            _, faces = self.detector.detect(color_image) 
            
            faces = faces if faces is not None else []

            for face in faces:
                cv2.rectangle(color_image, list(map(int, face[:4])), (0, 0, 0), -1)
                cv2.rectangle(depth_image, list(map(int, face[:4])), (0, 0, 0), -1)
            
        return color_image, depth_image
        
def getCamera():
    cameras = list()
    device  = None
    
    try:
        cameras = rs.context().devices
    except Exception as e:
        print(e)
    
    if cameras:
        device = cameras[0]
        
    return device

def run():     
  
    device = getCamera()
    
    if not device:  
        print("No camera detected")
        return   
        
    config      = rs.config()
    pipe        = rs.pipeline()  
    colorizer   = rs.colorizer(3)
    align       = rs.align(rs.stream.color)

    config.enable_stream(rs.stream.depth, TARGET_WIDTH, TARGET_HEIGHT, rs.format.z16,  TARGET_FPS)
    config.enable_stream(rs.stream.color, TARGET_WIDTH, TARGET_HEIGHT, rs.format.bgr8, TARGET_FPS)

    profile = pipe.start(config)

    detector = cv2.FaceDetectorYN.create("face_detection_yunet_2022mar.onnx", "", (320, 320))
    detector.setInputSize((TARGET_WIDTH, TARGET_HEIGHT))
        
    rgb_sensor   = None
    depth_sensor = None

    for s in device.sensors:                              
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            rgb_sensor = s 
        if s.get_info(rs.camera_info.name) == 'Stereo Module':
            depth_sensor = s

    if not rgb_sensor:      
        print("No rgb sensor")
        return
    
    if not depth_sensor:    
        print("No depth sensor")
        return
    
    cv2.namedWindow('RGB',   cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    
    while True:
        frameset = pipe.wait_for_frames()                  
        
        aligned_frames = align.process(frameset)
        color_frame    = aligned_frames.first(rs.stream.color)
        depth_frame    = aligned_frames.get_depth_frame()

        # depth_frame = rs.decimation_filter(1).process(depth_frame)
        # depth_frame = rs.disparity_transform(True).process(depth_frame)
        # depth_frame = rs.spatial_filter().process(depth_frame)
        # depth_frame = rs.temporal_filter().process(depth_frame)
        # depth_frame = rs.disparity_transform(False).process(depth_frame)
        # depth_frame = rs.hole_filling_filter().process(depth_frame)
        
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        _, faces = detector.detect(color_image) 
        
        faces = faces if faces is not None else []

        for face in faces:
            cv2.rectangle(color_image, list(map(int, face[:4])), (0, 0, 0), -1)
            cv2.rectangle(depth_image, list(map(int, face[:4])), (0, 0, 0), -1)
        
        cv2.imshow('RGB', color_image)
        cv2.imshow('Depth', depth_image)
        
        if cv2.waitKey(33) != -1:
            return
