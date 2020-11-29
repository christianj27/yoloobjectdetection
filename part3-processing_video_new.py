import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import xlsxwriter

option = {
    'model': 'cfg/v1/yolo-full-1c.cfg',
    'load': 10100,
    'threshold': 0.4,
    'gpu': 1.0
}

tfnet = TFNet(option)

#export to excel
workbook = xlsxwriter.Workbook('D:/KULIAH/Hasil Deteksi/Performa/laptop_malam.xlsx') 
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "FPS")
worksheet.write(0, 1, "Objek")
worksheet.write(0, 2, "Persentasi Deteksi")

capture = cv2.VideoCapture('test_video_night2.mp4')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
out = cv2.VideoWriter('D:/KULIAH/Hasil Deteksi/Performa/laptop_malam.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
row = 1
col = 0

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            if confidence >= 1.0:
              confidence = 1
            text = '{}: {:.2f}%'.format(label, confidence * 100)
            fps = '{:.1f}'.format(1 / (time.time() - stime))
            #text_result = '{}'
            worksheet.write(row, col, fps)
            worksheet.write(row, col + 1, label)
            worksheet.write(row, col + 2, confidence * 100)
            row += 1
            print('FPS {:.1f} -> {}'.format(1 / (time.time() - stime), text))
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        #cv2.imshow('frame', frame)
        #print('FPS {:.1f} - {}: {:.0f}%'.format(1 / (time.time() - stime), result['label'], result['confidence'] * 100))
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        out.release()
        workbook.close() 
        cv2.destroyAllWindows()
        break