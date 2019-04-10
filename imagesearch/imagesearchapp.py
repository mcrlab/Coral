# import the necessary packages
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import imutils
import cv2
from edgetpu.detection.engine import DetectionEngine

WIDTH = 960
HEIGHT = 480

class App:
    def __init__(self, vs):
        self.vs = vs
        self.frame = None
        self.thread = None
        self.stopEvent = None

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
            
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=WIDTH, height=HEIGHT)
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                label = "James"
                label_size = cv2.getTextSize(label, font, 0.5,cv2.LINE_AA)
                label_width = label_size[0][0]
                label_height = label_size[0][1]
                cv2.rectangle(image, (19, 20-label_height), (20+label_width, 20), (0,255,0),-1)
                
                cv2.rectangle(image, (20, 20), (40, 40), (0, 255, 0), 2)
                cv2.putText(image, "James", (25,15), font, 0.5, (255,255,255),1,cv2.LINE_AA)
                
                image = Image.fromarray(image)
                
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
            self.vs.stop()
            self.root.destroy()


    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()

