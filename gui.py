from datetime import datetime
import cameraWrapper as cw
import PySimpleGUI as sg
import json
import sys
import cv2
import os

DATABASE_PATH = "Database"
TARGET_IMAGES = 15
LOCATIONS = [
    "Epaule",
    "Abdomen",
    "Thorax Face",  
    "Thorax profil",
    "Coude",
    "Main",
    "Jambe",
    "Genou",
    # "Cheville"
]

class GUI:
    
    def __init__(self):
        self.isPlaying: bool                = False
        self.isRecording: bool              = False
        self.enableAnonymization: bool      = True
        self.autoIncrementLocation: bool    = True
        self.autoIncrementID: bool          = True
        self.frameCount: int                = 0
        self.camera: cw.CameraWrapper       = None
        
        self.loadConfig()
        self.initUI()
        self.bindTkinterEvents()
    
    def loadConfig(self):
        try:
            with open("config.json", "r") as f:
                self.config = json.load(f)
        except:
            sg.popup_error("Could not find 'config.json'")
            sys.exit()
            
    def initUI(self):
        sg.LOOK_AND_FEEL_TABLE["SystemDefaultForReal"]["BACKGROUND"] = "#ffffff"
        sg.theme("SystemDefaultForReal")
        
        layoutColumnRGB = [
            [sg.Image(k="imageRGB",   s=(640, 480))]
        ]
        
        layoutColumnDepth = [
            [sg.Image(k="imageDepth", s=(640, 480))]
        ]
        
        layout = [
            [sg.Text("ID", s=(15,1)), 
             sg.Input(str(self.config["nextID"]), k="inputID", 
                      s=(23,1), readonly=True),
             sg.Checkbox("Auto increment", k="_checkboxAutoIncrementID", 
                         default=self.autoIncrementID),
             sg.Button("NEXT", k="_buttonNextID")],
            [sg.Text("Current location", s=(15, 1)), 
             sg.Combo(LOCATIONS, s=(20, 1), readonly=True, 
                      default_value=LOCATIONS[0], k="comboLocations"),
             sg.Checkbox("Auto increment", k="_checkboxAutoIncrementLocations", 
                         default=self.autoIncrementLocation),
             sg.Button("Start recording", k="_buttonToggleRecording", disabled=True),
             sg.Text("...", k="textNumberFrames")],
            [sg.Button("Start camera", k="_buttonToggleCamera"),
             sg.Button("Disable anonymization", k="_buttonToggleAnonymization")],
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

    def bindTkinterEvents(self):
        self.window.bind("<Right>", "_right")
        self.window.bind("<Left>", "_left")
        self.window.bind("<space>", "_spacebar")
        self.window.bind("<Up>", "_up")
        self.window.bind("<Down>", "_down")
        
    def buttonToggleCameraClicked(self):
        
        if not self.camera:
            try:
                self.camera = cw.CameraWrapper()
                self.isPlaying = True
                text = "Stop playback"
                self.window["_buttonToggleRecording"].update(disabled=False)
            except Exception as e:
                sg.popup_error(e)
                return
            
        elif self.isPlaying:
            self.isPlaying   = False
            self.isRecording = False
            text = "Start playback"
            self.window["_buttonToggleRecording"].update(disabled=True)
            
        else:
            self.isPlaying = True
            text = "Stop playback"
            self.window["_buttonToggleRecording"].update(disabled=False)
        
        self.window["_buttonToggleCamera"].update(text)
    
    def buttonToggleRecordingClicked(self):
        if self.isRecording:
            self.frameCount = 0
            self.isRecording = False
            text = "Start recording"
            self.window["textNumberFrames"].update(str(self.frameCount) + " / " + str(TARGET_IMAGES))
        else:
            self.isRecording = True
            text = "Stop recording"
        
        self.window["_buttonToggleRecording"].update(text)
        
    def buttonNextIDClicked(self):
        self.config["nextID"] += 1
        with open("config.json", "w") as f:
            json.dump(self.config, f)
        
        self.window["inputID"].update(self.config["nextID"])
    
    def buttonPreviousIDClicked(self):
        self.config["nextID"] -= 1
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
                
            elif self.frameCount == 0:
                print("folder already exists")
                # sg.PopupYesNo("")
                pass
                
            rgbImagePath   = os.path.join(writeDirectoryPath, 
                                          f"RGB_{dateTime}_{frameCountString}.jpeg")
            depthImagePath = os.path.join(writeDirectoryPath, 
                                          f"D_{dateTime}_{frameCountString}.tiff")
            
            cv2.imwrite(rgbImagePath, RGBFrame)
            cv2.imwrite(depthImagePath,   DepthFrame)
            
            if self.frameCount >= TARGET_IMAGES:
                self.buttonToggleRecordingClicked()
                if self.autoIncrementLocation:
                    self.updateComboLocation("next")
            else:
                self.frameCount += 1
                self.window["textNumberFrames"].update(str(self.frameCount) + " / " + str(TARGET_IMAGES))
        
        bufferRGB = cv2.imencode('.png', RGBFrame)[1].tobytes()
        bufferDPT = cv2.imencode('.png', DepthFrame)[1].tobytes()
        
        self.window["imageRGB"].update(data=bufferRGB)
        self.window["imageDepth"].update(data=bufferDPT)
    
    def buttonToggleAnonymizationClicked(self):
        if self.enableAnonymization:
            self.window["_buttonToggleAnonymization"].update("Enable anonymization")
        else:
           self.window["_buttonToggleAnonymization"].update("Disable anonymization") 
        self.enableAnonymization = not self.enableAnonymization
    
    def updateComboLocation(self, direction):
        
        currentSelection      = self.window["comboLocations"].get()
        currentSelectionIndex = LOCATIONS.index(currentSelection)
        
        if direction == "next":
            if currentSelectionIndex < len(LOCATIONS)-1:
                newSelection = LOCATIONS[currentSelectionIndex + 1]
            else:
                newSelection = LOCATIONS[0]
                if self.autoIncrementID:
                    self.buttonNextIDClicked()
                
        elif direction == "previous":
            if currentSelectionIndex > 0:
                newSelection = LOCATIONS[currentSelectionIndex - 1]
            else:
                newSelection = LOCATIONS[-1]
                
        else:
            raise Exception("Not implemented")
        
        self.window["comboLocations"].update(newSelection)
        
    def run(self):
        
        while True:
            event, _ = self.window.read(timeout=0)

            if event == sg.WINDOW_CLOSED:
                break
            
            if self.isPlaying:
                self.handleFrames()
                
            if event == "_buttonToggleCamera":
                self.buttonToggleCameraClicked()
            
            elif event in ("_buttonToggleRecording", "_spacebar"):
                self.buttonToggleRecordingClicked()
                
            elif event == "_buttonToggleAnonymization":
                self.buttonToggleAnonymizationClicked()
            
            elif event in ("_buttonNextID", "_right"):
                self.buttonNextIDClicked()
                
            elif event == "_left":
                self.buttonPreviousIDClicked()
                
            elif event == "_up":
                self.updateComboLocation("previous")
            
            elif event == "_down":
                self.updateComboLocation("next")
            
            elif event == "_checkboxAutoIncrementID":
                self.autoIncrementID = self.window["_checkboxAutoIncrementID"].get()
                
            elif event == "_checkboxAutoIncrementLocations":
                self.autoIncrementLocation = self.window["_checkboxAutoIncrementLocations"].get()
            
if __name__ == "__main__":
    GUI().run()