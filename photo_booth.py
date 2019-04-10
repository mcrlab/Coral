import argparse
from imagesearch.imagesearchapp import App
from imutils.video import VideoStream
import time

def main():
    print("[INFO] warming up camera...")
    vs = VideoStream().start()
    time.sleep(2)

    app = App(vs)
    app.root.mainloop()

if __name__ == '__main__':
    main()