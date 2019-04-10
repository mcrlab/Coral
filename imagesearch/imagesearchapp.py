# import the necessary packages
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import imutils
import cv2
from edgetpu.detection.engine import DetectionEngine

WIDTH = 640
HEIGHT = 480

# Function to read labels from text files.
def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

class App:
    def __init__(self):

        self.frame = None
        self.thread = None
        self.stopEvent = None
        
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, 640)
        self.camera.set(4, 480)
        

        self.engine = DetectionEngine('./mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite')
        self.labels = ReadLabelFile('./coco_labels.txt')
        
        
        self.root = tki.Tk()
#        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.onClose())
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        self.panel = None

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

    def videoLoop(self):
        try:
            _, width, height, channels = self.engine.get_input_tensor_shape()
            while not self.stopEvent.is_set():
                if not self.camera.isOpened():
                    continue
                ret, self.frame = self.camera.read()
                if not ret:
                    continue

                font = cv2.FONT_HERSHEY_SIMPLEX
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                input = cv2.resize(self.frame, (width, height))
                input = input.reshape((width * height * channels))
                ans = self.engine.DetectWithInputTensor(input, threshold=0.05)
                if ans:
                    for obj in ans:
                        if(obj.score > 0.1):
                            box = obj.bounding_box.flatten().tolist()
            
                            cv2.rectangle(self.frame, (int(WIDTH * box[0]), int(HEIGHT * box[1])), (int(WIDTH * box[2]), int(HEIGHT * box[3])), (0, 255, 0), 2)
                
                            label = self.labels[obj.label_id]
                            label_size = cv2.getTextSize(label, font, 0.5,cv2.LINE_AA)
                            label_width = label_size[0][0]
                            label_height = label_size[0][1]
                            cv2.rectangle(self.frame, (int(WIDTH * box[0])-1, int(HEIGHT * box[1])-label_height), (int(WIDTH * box[0])+label_width, int(HEIGHT * box[1])), (0,255,0),-1)
                            
                            
                            cv2.putText(self.frame, label, (int(WIDTH * box[0])+5, int(HEIGHT * box[1])-5), font, 0.5, (255,255,255),1,cv2.LINE_AA)
                
                image = Image.fromarray(self.frame)
                
                image = ImageTk.PhotoImage(image)
        
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=0, pady=0)

                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
                        
        except Exception as e:
            
            print("[INFO] caught a RuntimeError")
            print(e)
        finally:
            self.camera.release()
            self.root.destroy()


    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()

