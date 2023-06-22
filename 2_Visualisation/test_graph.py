from io import BytesIO
from PIL import Image
import numpy as np
import PySimpleGUI as sg
import cv2

imagePath = "D_20230426_144008_00000.tiff"

def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="png")
        data = output.getvalue()
    return data

font = ("Courier New", 11)
sg.theme("DarkBlue3")
sg.set_options(font=font)

image = cv2.imread(imagePath, cv2.IMREAD_ANYDEPTH)

data = array_to_data(image)

width, height = 640, 480

layout = [[sg.Graph(
    canvas_size=(width, height),
    graph_bottom_left=(0, 0),
    graph_top_right=(width, height),
    key="-GRAPH-",
    enable_events=True, 
    background_color='lightblue',
    drag_submits=True), ],]
window = sg.Window("Test", layout, finalize=True)
window.bind("<Space>", "space")
graph = window["-GRAPH-"]
graph.draw_image(data=data, location=(0, height))

while True:

    event, values = window.read()
    if event in (sg.WINDOW_CLOSED, 'Exit'):
        break
    
    if event == "-GRAPH-":
        x = values["-GRAPH-"][0]
        y = height - values["-GRAPH-"][1]
        
        print(image.shape)
        if x < image.shape[1] and y < image.shape[0]:
            print(f"{x}-{y}: {image[y][x]}")
    
    if event == "space":
        pass

window.close()