import cv2
from darkflow.net.build import TFNet
from datetime import datetime
import numpy as np
import time
import xlsxwriter

options = {
    'model': 'cfg/v1/yolo-full-1c.cfg',
    'load': 10100,
    'threshold': 0.4,
    'gpu': 1.0
}

def reset_variable():
    #print("Masuk function")
    global timestamp, position, direction, estimated, logged, lastPoint, speedMPS, speed
    timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}
    position = {"A": None, "B": None, "C": None, "D": None}
    direction = None
    estimated = False
    logged = False
    lastPoint = False
    speedMPS = 0
    speed = ""

width = 1920 #Desired width output video
height = 1080 #Desired height output video
tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
out = cv2.VideoWriter('Hasilspeed_malem.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 4, (width,height)) #Untuk menyimpan hasil deteksi ke dalam video

#export to excel (deklarasi header tabel excel)
workbook = xlsxwriter.Workbook('Hasilspeed_malem.xlsx') 
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "FPS")
worksheet.write(0, 1, "Objek")
worksheet.write(0, 2, "Persentasi Deteksi")
worksheet.write(0, 5, "No")
worksheet.write(0, 6, "Pergerakan Objek")
worksheet.write(0, 7, "Timestamp A")
worksheet.write(0, 8, "Position A (px)")
worksheet.write(0, 9, "Timestamp B")
worksheet.write(0, 10, "Position BA (px)")
worksheet.write(0, 11, "Timestamp C")
worksheet.write(0, 12, "Position C (px)")
worksheet.write(0, 13, "Timestamp D")
worksheet.write(0, 14, "Position D (px)")
worksheet.write(0, 15, "Kecepatan (m/s)")
row = 1
row2 = 1
col = 0

distance = 9 #Approx real distance in a video frame from every edge
meterPerPixel = distance/width #Calculate meter per pixel
condition_dir = width/2
speed_estimation_zone = {"A": 160, "B": 680, "C": 1200, "D": 1700} #Zona estimasi untuk centroid dalam column
timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}
points = [("A", "B"), ("B", "C"), ("C", "D")]
position = {"A": None, "B": None, "C": None, "D": None}
direction = None
estimated = False
logged = False
lastPoint = False
speed = ""
pergerakanobjek = ""
speedMPS_sebelumnya = 0

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            ts = datetime.now()
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            centerCoord = (int((result['topleft']['x']+result['bottomright']['x'])/2), int((result['topleft']['y']+result['bottomright']['y'])/2)) #Mengambil nilai centroid dari setiap bounding box deteksi
            
            #print(estimated)
            #Perhitungan Speed Estimation Object
            if not estimated:
                if direction is None:
                    direction = centerCoord[0] #Untuk menentukan arah munculnya objek
                    print('direction : {}'.format(direction))
                    print('Timestamp A start : {}'.format(timestamp["A"]))
                    print('Timestamp B start : {}'.format(timestamp["B"]))
                    print('Timestamp C start : {}'.format(timestamp["C"]))
                    print('Timestamp D start : {}'.format(timestamp["D"]))

                if direction < condition_dir: #Jika objek muncul dari kiri ke kanan framse
                    #print("Kiri ke kanan")
                    pergerakanobjek = "Kiri ke kanan"
                    if timestamp["A"] == 0 :
                        if centerCoord[0] > speed_estimation_zone["A"]: #Jika objek sudah sampai di point A maka masuk kondisi ini
                            timestamp["A"] = ts #Set waktu objek sampai di point A berdasarkan jam WIndows
                            position["A"] = centerCoord[0] #Set posisi objek dalam kolom di gambar
                            print('Timestamp A : {}'.format(timestamp["A"]))
                            print('Position A : {}'.format(position["A"]))
                    elif timestamp["B"] == 0 :
                        if centerCoord[0] > speed_estimation_zone["B"]:
                            timestamp["B"] = ts
                            position["B"] = centerCoord[0]
                            print('Timestamp B : {}'.format(timestamp["B"]))
                            print('Position B : {}'.format(position["B"]))
                    elif timestamp["C"] == 0 :
                        if centerCoord[0] > speed_estimation_zone["C"]:
                            timestamp["C"] = ts
                            position["C"] = centerCoord[0]
                            print('Timestamp C : {}'.format(timestamp["C"]))
                            print('Position C : {}'.format(position["C"]))
                    elif timestamp["D"] == 0 :
                        if centerCoord[0] > speed_estimation_zone["D"]:
                            timestamp["D"] = ts
                            position["D"] = centerCoord[0]
                            print('Timestamp D : {}'.format(timestamp["D"]))
                            print('Position D : {}'.format(position["D"]))
                            lastPoint = True #Jika objek sudah di point D maka set lastPoint jadi true untuk melakukan perhitungan

                elif direction > condition_dir: #Jika objek muncul dari kanan ke kiri framse
                    #print("Kanan ke kiri")
                    pergerakanobjek = "Kanan ke kiri"
                    if timestamp["D"] == 0 :
                        if centerCoord[0] < speed_estimation_zone["D"]:
                            timestamp["D"] = ts
                            position["D"] = centerCoord[0]
                            print('Timestamp D : {}'.format(timestamp["D"]))
                            print('Position D : {}'.format(position["D"]))
                    elif timestamp["C"] == 0 :
                        if centerCoord[0] < speed_estimation_zone["C"]:
                            timestamp["C"] = ts
                            position["C"] = centerCoord[0]
                            print('Timestamp C : {}'.format(timestamp["C"]))
                            print('Position C : {}'.format(position["C"]))
                    elif timestamp["B"] == 0 :
                        if centerCoord[0] < speed_estimation_zone["B"]:
                            timestamp["B"] = ts
                            position["B"] = centerCoord[0]
                            print('Timestamp B : {}'.format(timestamp["B"]))
                            print('Position B : {}'.format(position["B"]))
                    elif timestamp["A"] == 0 :
                        if centerCoord[0] < speed_estimation_zone["A"]:
                            timestamp["A"] = ts #Set waktu objek sampai di point A berdasarkan jam WIndows
                            position["A"] = centerCoord[0] #Set posisi objek dalam kolom di gambar
                            print('Timestamp A : {}'.format(timestamp["A"]))
                            print('Position A : {}'.format(position["A"]))
                            lastPoint = True #Jika objek sudah di point D maka set lastPoint jadi true untuk melakukan perhitungan

                if lastPoint and not estimated: #jika nilai lastPoint true maka masuk ke kondisi ini
                    estimatedSpeeds = []

                    for (i, j) in points: #for untuk menghitung estimasi
                        # calculate the distance in pixels
                        d = position[j] - position[i]
                        distanceInPixels = abs(d)

                        # check if the distance in pixels is zero, if so,
                        # skip this iteration
                        if distanceInPixels == 0:
                            continue

                        # calculate the time in hours
                        t = timestamp[j] - timestamp[i]
                        timeInSeconds = abs(t.total_seconds())
                        timeInHours = timeInSeconds / (60 * 60)

                        # calculate distance in kilometers and append the
                        # calculated speed to the list
                        distanceInMeters = distanceInPixels * meterPerPixel
                        #distanceInKM = distanceInMeters / 1000
                        #estimatedSpeeds.append(distanceInKM / timeInHours)
                        estimatedSpeeds.append(distanceInMeters / timeInSeconds)

                    speedMPS = np.average(estimatedSpeeds) #Nilai rata2 speed dalam satuan meter/detik
                    #speedKMPH = np.average(estimatedSpeeds)
                    #MILES_PER_ONE_KILOMETER = 0.621371
                    #speedMPH = speedKMPH * MILES_PER_ONE_KILOMETER

                    estimated = True
                    speed = "[INFO] Kecepatan manusia : {:.2f} m/s".format(speedMPS)
                    #Input Kecepatan ke Excel
                    if speedMPS != speedMPS_sebelumnya:
                        worksheet.write(row2, 5, row2)
                        worksheet.write(row2, 6, pergerakanobjek)
                        worksheet.write(row2, 7, timestamp["A"])
                        worksheet.write(row2, 8, position["A"])
                        worksheet.write(row2, 9, timestamp["B"])
                        worksheet.write(row2, 10, position["B"])
                        worksheet.write(row2, 11, timestamp["C"])
                        worksheet.write(row2, 12, position["C"])
                        worksheet.write(row2, 13, timestamp["D"])
                        worksheet.write(row2, 14, position["D"])
                        worksheet.write(row2, 15, speedMPS)
                        speedMPS_sebelumnya = speedMPS
                        row2 += 1

                    print('{} -> Objek ke {}'.format(speed, row2)) #melakukan print di command prompt

            else:
                #print("Masuk else")
                reset_variable()

            text = '{}: {:.2f}%'.format(label, confidence * 100)
            fps = '{:.1f}'.format(1 / (time.time() - stime))
            #input hasil deteksi dan fps ke Excel
            worksheet.write(row, col, fps)
            worksheet.write(row, col + 1, label)
            worksheet.write(row, col + 2, confidence * 100)
            row += 1
            print('FPS {:.1f} -> {}'.format(1 / (time.time() - stime), text))
            if speed != "": #Jika sudah ada nilai dari Speed maka masuk kondisi ini
                frame = cv2.rectangle(frame, (1, 1), (680,40), (215, 215, 215), -1) #Set kotak abu2 dibelakang tulisan
                frame = cv2.putText(frame, speed, (1,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1) #Set tulisan di video untuk info perkiraan kecepatan
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #frame = cv2.putText(
                #frame, '.', centerCoord, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 20)
        cv2.imshow('frame', frame)
        out.write(frame) #Untuk melakukan write video dari hasil deteksi
        #print('FPS {:.1f} - {}: {:.0f}%'.format(1 / (time.time() - stime), result['label'], result['confidence'] * 100))
        #print('FPS {:.1f} -> '.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
out.release()
workbook.close()
cv2.destroyAllWindows()
