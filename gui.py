from datetime import datetime
import cameraWrapper as cw
import PySimpleGUI as sg
import json
import cv2
import os

DATABASE_PATH = "Database"
LOCATIONS = [
    "Epaule",
    "Abdomen",
    "Bassin",
    "Jambe",
    "Thorax Face",  
    "Thorax profil droit",
    "Thorax profil gauche",
    "Genou",
    "Bras",
    "Avant-bras",
    "Doigt",
    "Coude",
    "Main",
    "Cheville",
    "Pied"
]

class GUI:
    
    def __init__(self):
        self.window                 = None
        self.camera                 = cw.CameraWrapper()
        self.isPlaying              = False
        self.isRecording            = False
        self.enableAnonymization    = True
        self.frameCount             = 0
        
        self.loadConfig()
        self.initUI()
    
    def loadConfig(self):
        try:
            with open("config.json", "r") as f:
                self.config = json.load(f)
        except:
            sg.popup_error("Could not find 'config.json'")
            exit()
            
    def initUI(self):
        sg.LOOK_AND_FEEL_TABLE["SystemDefaultForReal"]["BACKGROUND"] = "#ffffff"
        sg.theme("SystemDefaultForReal")
        
        layoutColumnRGB = [
            [sg.Image(k="imageRGB", s=(640, 480))]
        ]
        
        layoutColumnDepth = [
            [sg.Image(k="imageDepth", s=(640, 480))]
        ]
        
        layout = [
            [sg.Text("ID", s=(15,1)), 
             sg.Input(str(self.config["nextID"]), k="inputID", 
                      s=(23,1), readonly=True),
             sg.Button("NEXT", k="_buttonNextID")],
            [sg.Text("Current location", s=(15, 1)), 
             sg.Combo(LOCATIONS, s=(20, 1), readonly=True, 
                      default_value=LOCATIONS[0], k="comboLocations"),
            #  sg.Button("NEXT", k="_buttonNextLocation"),],
            [sg.Button("Start camera", k="_buttonToggleCamera"), 
             sg.Button("Toggle anonymization", k="_buttonToggleAnonymization")],
            [sg.Button("Start recording", k="_buttonToggleRecording")],
            [sg.P(), sg.HorizontalSeparator(pad=(10, 30)), sg.P()],
            [sg.Column(layout=layoutColumnRGB, k="columnImageRGB"), 
             sg.VerticalSeparator(), 
             sg.Column(layout=layoutColumnDepth, k="columnImageDepth")]
        ]
        
        self.window = sg.Window("Rec",
                                layout=layout, 
                                finalize=True, 
                                ttk_theme=sg.THEME_VISTA,
                                use_ttk_buttons=True)

    def buttonToggleCameraClicked(self):
        
        if self.isPlaying:
            self.isPlaying = False
            text = "Start camera"
        else:
            self.isPlaying = True
            text = "Stop camera"
        
        self.window["_buttonToggleCamera"].update(text)
    
    def buttonToggleRecordingClicked(self):
        if self.isRecording:
            self.isRecording = False
            text = "Start recording"
        else:
            self.isRecording = True
            text = "Stop recording"
        
        self.window["_buttonToggleRecording"].update(text)
        
    def buttonNextIDClicked(self):
        self.config["nextID"] += 1
        f = open("config.json", "w")
        json.dump(self.config, f)
        f.close()
        
        self.window["inputID"].update(self.config["nextID"])
         
    def handleFrames(self):
        frames = self.camera.getNextFrames(self.enableAnonymization)
                
        if not frames:
            return
        
        RGBFrame    = frames[0]
        DepthFrame  = frames[1]
        
        dateTime         = datetime.now().strftime("%Y%m%d_%H%M%S")
        frameCountString = '{:0>5}'.format(self.frameCount)
            
        if self.isRecording:
            
            writeDirectoryPath = os.path.join(DATABASE_PATH, 
                                              self.window["inputID"].get(),
                                              self.window["comboLocations"].get())
        
            if not os.path.exists(writeDirectoryPath):
                os.makedirs(writeDirectoryPath)
                
            rgbImagePath   = os.path.join(writeDirectoryPath, 
                                          f"RGB_{dateTime}_{frameCountString}.jpeg")
            depthImagePath = os.path.join(writeDirectoryPath, 
                                          f"D_{dateTime}_{frameCountString}.tiff")
            
            cv2.imwrite(rgbImagePath, RGBFrame)
            cv2.imwrite(depthImagePath,   DepthFrame)
        
        bufferRGB = cv2.imencode('.png', RGBFrame)[1].tobytes()
        bufferDPT = cv2.imencode('.png', DepthFrame)[1].tobytes()
        
        self.frameCount += 1
        
        self.window["imageRGB"].update(data=bufferRGB)
        self.window["imageDepth"].update(data=bufferDPT)
                
    def run(self):
        
        while True:
            event, _ = self.window.read(timeout=0)
            
            if event == sg.WINDOW_CLOSED:
                break
            
            if self.isPlaying:
                self.handleFrames()
            
            if event == "_buttonToggleCamera":
                self.buttonToggleCameraClicked()
            
            if event == "_buttonToggleRecording":
                self.buttonToggleRecordingClicked()
                
            if event == "_buttonToggleAnonymization":
                self.enableAnonymization = not self.enableAnonymization
            
            if event == "_buttonNextID":
                self.buttonNextIDClicked()
            
            
            
    

if __name__ == "__main__":
    GUI().run()