#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import cv2
from darkflow.net.build import TFNet


options = {"model": "cfg/v1/yolo-full.cfg",
           "load": "bin/yolo-full.weights",
           "threshold": 0.2,
           "gpu": 1.0}

tfnet2 = TFNet(options)
#tfnet2.load_from_ckpt()

original_img = cv2.imread("/content/darkflow/images/969.png")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
print(results)

fig, ax = plt.subplots(figsize=(15, 15))
ax.imshow(original_img)

def boxing(original_img , predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.2f}%'.format(label, confidence * 100)
        
        if confidence > 0.2:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, text, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            print(text)        

    return newImage


fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(boxing(original_img, results))