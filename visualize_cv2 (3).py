import cv2
import numpy as np
import os
import sys
import random
import itertools
import colorsys
import time
import math
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from samples.pedestrian import pedestrian
from matplotlib import pyplot as plt


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#PEDESTRIAN_WEIGHTS_PATH = "/home/bilun/Documents/mask-rcnn2/mask_rcnn_pedestrian_8020_night_new.h5"
PEDESTRIAN_WEIGHTS_PATH = "mask_rcnn_pedestrian_8020_day_new.h5"
config = pedestrian.PedestrianConfig()
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
#weights_path = "/home/bilun/Documents/mask-rcnn2/mask_rcnn_pedestrian_8020_night_new.h5"
weights_path = "mask_rcnn_pedestrian_8020_day_new.h5"
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

class_names = ['BG', 'pedestrian']
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, ids, names, scores, list_all, list1, list2, list3, list4, list5, list6, colors=None, speed=None, xc=None, yc=None):
    """
        take the image and results and apply the mask, box, and Label
    """
    #initialisasi beberapa parameter
    n_instances = boxes.shape[0]
    #hitf = open("hasil.txt",'w')
    # batas = 0.5
    # #hitf.write('image, class, scores, y1, x1, y2, x2\n')
    if not n_instances:
         print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    
    colors = colors or random_colors(n_instances)
    probabilitas=0
    y1=0 
    x1=0 
    y2=0 
    x2=0
    xc=0
    yc=0
    dist=0
    #list0 = []
    #list1 = []
    #list2 = []
    #list3 = []
    #list4 = []
    #list5 = []
    #list6 = []
    print("Jumlah objek per frame: "+str(n_instances))

    #deteksi
    for i in range(n_instances):
        print('Objek ke:')
        print(i+1)
        if not np.any(boxes[i]):
            continue
        #j=i+1
        y1, x1, y2, x2 = boxes[i]
        #y11, x11, y22, x22 = boxes[j]
        xc1 = (x1+x2)/2
        yc1 = (y1+y2)/2

        #kondisi menghitung jarak
        if i == 0:
            list_all.append(xc1)
            list_all.append(yc1)
            print("hasil {}".format(list_all))
            if len(list_all) == 4:
                dist = math.sqrt((abs(list_all[0]-list_all[2]))**(2)+(abs(list_all[1]-list_all[3]))**(2))
                del list_all[0]
                del list_all[0]
                print(dist)
            #print(list0)

        elif i == 1:
            list1.append(xc1)
            list1.append(yc1)
            print("hasil {}".format(list1))
            if len(list1) == 4:
                dist = math.sqrt((abs(list1[0]-list1[2]))**(2)+(abs(list1[1]-list1[3]))**(2))
                del list1[0]
                del list1[0]
                print(dist)
  
        elif i == 2:
            list2.append(xc1)
            list2.append(yc1)
            print("hasil {}".format(list2))
            if len(list2) == 4:
                dist = math.sqrt((abs(list2[0]-list2[2]))**(2)+(abs(list2[1]-list2[3]))**(2))
                del list2[0]
                del list2[0]
                print(dist)
   
        elif i == 3:
            list3.append(xc1)
            list3.append(yc1)
            print("hasil {}".format(list3))
            if len(list3) == 4:
                dist = math.sqrt((abs(list3[0]-list3[2]))**(2)+(abs(list3[1]-list3[3]))**(2))
                del list3[0]
                del list3[0]
                print(dist)
           
        elif i == 4:
            list4.append(xc1)
            list4.append(yc1)
            print("hasil {}".format(list4))
            if len(list4) == 4:
                dist = math.sqrt((abs(list4[0]-list4[2]))**(2)+(abs(list4[1]-list4[3]))**(2))
                del list4[0]
                del list4[0]
                print(dist)
   
        elif i == 5:
            list5.append(xc1)
            list5.append(yc1)
            print("hasil {}".format(list5))
            if len(list5) == 4:
                dist = math.sqrt((abs(list5[0]-list5[2]))**(2)+(abs(list5[1]-list5[3]))**(2))
                del list5[0]
                del list5[0]
                print(dist)
                
        elif i == 6:
            list6.append(xc1)
            list6.append(yc1)
            print("hasil {}".format(list6))
            if len(list6) == 4:
                dist = math.sqrt((abs(list6[0]-list6[2]))**(2)+(abs(list6[1]-list6[3]))**(2))
                del list6[0]
                del list6[0]
                print(dist)
 
        else:
            continue

        fname = "image"+str(i)
        label = names[ids[i]]
        color = colors[i]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f} {}'.format(label, score, xc1) if score else label
        mask = masks[:, :, i]
        probabilitas = str(score)
        #print('nilai')
        print("Probabilitas "+probabilitas)
        bbox1 = str(boxes[i])
        #bbox2 = str(boxes[i+1])
        #print("x1: "+x1)
        #print("y1: " +y1)
        #print("x2: " +x2)
        #print("y2: " +y2)
        #nilaisemua = print(probabilitas)
        #hitf.write('image, class, scores, y1, x1, y2, x2\n')
        #hitf.write(probabilitas)
        #hitf.write("\n")
        #if (score>=batas):
            #probabilitas = str(score)
            #  bbox = str(boxes[i])
            # label1 = str(label)
            #line = ",".join([fname,label1,probabilitas,bbox])
            #hitf.write(line+"\n")
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        print("Koordinat bounding box ({},{}) ({},{})".format(x1, y1, x2, y2))
        print("x center adalah "+str(xc1))
        print("y center adalah "+str(yc1))
    return list_all, list1, list2, list3, list4, list5, list6, image
    #hitf.flush()
    #hitf.close()


if __name__ == '__main__':
    """
        test everything
    """

    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        print(frame)
        #plt.imshow('frame', frame)
        #plt.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    #cv2.destroyAllWindows()
