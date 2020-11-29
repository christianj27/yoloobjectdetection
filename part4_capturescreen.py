import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from mss import mss
from PIL import Image
from darkflow.net.build import TFNet
import time

options = {
    'model' : 'cfg/v1/yolo-full-1c.cfg' ,
    'load' : -1,
    'threshold' : 0.2,
    'gpu' : 1.0 }
tfnet = TFNet( options )
color = (0, 255, 0) # bounding box color.

# This defines the area on the screen.
mon = {'top' : 10, 'left' : 10, 'width' : 1000, 'height' : 800}
sct = mss()
previous_time = 0
while True :
    sct.get_pixels(mon)
    frame = Image.frombytes( 'RGB', (sct.width, sct.height), sct.image )
    frame = np.array(frame)
    # image = image[ ::2, ::2, : ] # can be used to downgrade the input
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict( frame )
    for result in results :
        tl = ( result['topleft']['x'], result['topleft']['y'] )
        br = ( result['bottomright']['x'], result['bottomright']['y'] )
        label = result['label']
        confidence = result['confidence']
        text = '{} : {:.0f}%'.format( label, confidence * 100 )
        frame = cv2.rectangle( frame, tl, br, color, 5 )
        frame = cv2.putText( frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2 )
    cv2.imshow ( 'frame', frame )
    if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
        cv2.destroyAllWindows()
    txt1 = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
    previous_time = time.time()
    print(txt1)