import pyrealsense2 as rs
import numpy as np
import cv2

TARGET_WIDTH  = 640  # Width of captured images
TARGET_HEIGHT = 480  # Height of captured images
TARGET_FPS    = 6    # Number of FPS

class CameraWrapper:
    
    def __init__(self):
        """Gets the camera object from realsense API and initialize the pipeline

        Raises:
            Exception: If no realsense device is not connected
        """
        self.camera = getCamera()
        
        if not self.camera:  
            raise Exception("Could not find any RealSense Device")
        
        self.initCamera()
        
    def initCamera(self):
        """Initalise the camera pipeline
        """
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
        
    def getNextFrames(self, enableAnonymization: bool) -> tuple:
        """Recovers the lastest aligned frames from the camera feed and returns them
           as numpy arrays

        Args:
            enableAnonymization (bool): if true, will run face detection model and 
                                        draw a black rectangle on faces

        Returns:
            color_image: np.array, depth_image: np.array
        """
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
    """Revocers the camera object

    Returns:
        rs.device: camera object
    """
    cameras = list()
    device  = None
    
    try:
        cameras = rs.context().devices
    except Exception as e:
        print(e)
    
    if cameras:
        device = cameras[0]
        
    return device

def getIntrinsics():
    return rs.intrinsics

def frameToPointCloud(depth_frame, color_frame):
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)
    
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

if __name__ == "__main__":
    # print(getIntrinsics().fx)
    imgD   = cv2.imread("D_20230426_144008_00000.tiff", cv2.IMREAD_ANYDEPTH)
    imgD = imgD / 255
    # imgRGB = cv2.imread("RGB_20230426_144008_00000.jpeg")

    for x, row in enumerate(imgD):
        for y, value in enumerate(row):
            print( rs.rs2_deproject_pixel_to_point(rs.intrinsics(), [float(x), float(y)], float(imgD[x][y])) )
    # rs.rs2_deproject_pixel_to_point()
    # frameToPointCloud(imgD, imgRGB)