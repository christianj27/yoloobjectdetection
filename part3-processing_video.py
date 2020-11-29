import cv2
from darkflow.net.build import TFNet
import numpy as np

option = {
    'model': 'cfg/v1/yolo-full-1c.cfg',
    'load': -1,
    'threshold': 0.4,
    'gpu': 1.0
}

tfnet = TFNet(option)

capture = cv2.VideoCapture('ambildata/jalan.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            print('FPS {:.1f} -> {}'.format(1 / (time.time() - stime), text))
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        #print('FPS {:.1f} - {}: {:.0f}%'.format(1 / (time.time() - stime), result['label'], result['confidence'] * 100))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
