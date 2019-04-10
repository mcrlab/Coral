import argparse
from imagesearch.imagesearchapp import App
from imutils.video import VideoStream
import time

def main():
    app = App()
    app.root.mainloop()

if __name__ == '__main__':
    main()