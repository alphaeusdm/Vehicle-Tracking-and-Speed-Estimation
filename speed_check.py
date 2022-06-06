import cv2
import dlib
import time
import threading
import math
import requests
from pprint import pprint
import os
from db import conn,psycopg2
import pytesseract
import pandas as pd
try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
from tabulate import tabulate

regions = ['in']
carCascade = cv2.CascadeClassifier('/Haar Data/drive/cascade.xml')
video = cv2.VideoCapture('video.mp4')

WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	ppm = 2
	d_meters = d_pixels / ppm
	fps = 18
	speed = d_meters * fps * 3.6
	return speed
	

def trackMultipleObjects():
	SPEED_LIMIT = 60
	print("SPEED_LIMIT SET AT "+str(SPEED_LIMIT)+" Km / Hr")
	print()
	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	carLocation2 = {}
	speed = [None] * 1000
	LP = [None] * 1000
	
	# Write output to video file
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))


	while True:
		start_time = time.time()
		rc, image = video.read()
		if type(image) == type(None):
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []

		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			if trackingQuality < 7:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			# print ('Removing carID ' + str(carID) + ' from list of trackers.')
			# print ('Removing carID ' + str(carID) + ' previous location.')
			# print ('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)
		
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				
				if matchCarID is None:
					print ('Creating new tracker ' + str(currentCarID))
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[currentCarID] = tracker
					carLocation1[currentCarID] = [x, y, w, h]
					cropped = image[y:y+h,x:x+w]
					if(len(cropped) == 0):
						break
					cv2.imwrite("./data/thumbnail.jpg",cropped)

					try:
						x = []
						y = []

						image = cv2.imread("./data/thumbnail.jpg")
						gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
						gray = cv2.bilateralFilter(gray, 11, 17, 17)
						edged = cv2.Canny(gray, 170, 200)
						_,contours,hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
						contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
						NumberPlateCnt = 0
						count = 0
						for c in contours:
								peri = cv2.arcLength(c, True)
								approx = cv2.approxPolyDP(c, 0.02 * peri, True)
								if len(approx) == 4:  
									NumberPlateCnt = approx
									break
						mask = np.zeros(gray.shape,np.uint8)
						new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
						new_image = cv2.bitwise_and(image,image,mask=mask)
						new_image = cv2.resize(new_image,None,fx=10, fy=10, interpolation = cv2.INTER_CUBIC)

						rc = cv2.minAreaRect(contours[0])
						box = cv2.boxPoints(rc)
						for p in box:
							pt = (p[0],p[1])
							x.append(p[0])
							y.append(p[1])

						if (x[0]-x[1])<=10:
							if (x[0]-x[1])<2:
								exit
							else:
								angle = np.rad2deg(np.arctan2(np.abs(y[3] - y[0]), np.abs(x[3] - x[0])))
								angle = angle*(-1)
								img = Image.fromarray(new_image)
								rotated = img.rotate(angle)
								rotated.save("./data/rotated"+str(currentCarID)+".jpg")
								new_image = cv2.imread("./data/rotated"+str(currentCarID)+".jpg")
						elif (x[0]-x[3])<=10:
							if (x[0]-x[3])<2:
								exit
							else:
								angle = np.rad2deg(np.arctan2(np.abs(y[1] - y[0]), np.abs(x[1] - x[0])))
								angle = angle*(-1)
								img = Image.fromarray(new_image)
								rotated = img.rotate(angle)
								rotated.save('./rotate-output.jpg')
								new_image = cv2.imread('./rotate-output.jpg')
						config = ('-l eng --oem 1 --psm 3')
						text = pytesseract.image_to_string(new_image, config=config)
						raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
								'v_number': [text]}
						df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
						df.to_csv('data.csv')
						print("OCR Result : "+text)
						LP[currentCarID] = text
						try:
							cursor = conn.cursor()
							postgreSQL_select_Query = "select * from black_list where Number_Plate=%s"

							cursor.execute(postgreSQL_select_Query, (LP[currentCarID],))
							print("Looking for match in Black Listed Vehicles...")

							records = cursor.fetchall() 
							if len(records) != 0:
								print()
								print("Match Found")
								for row in records:
									table = [["Vehicle Id",row[0]],["Number Plate",row[1]], ["Type",row[2]],["Description",row[3]]]   
									print(tabulate(table, tablefmt="fancy_grid"))
									print()
							else:
								print("Result : Vehicle was never Black Listed")

						except (Exception, psycopg2.Error) as error :
							print ("Error while fetching data from PostgreSQL", error)
						# LP var is globally declared
					except:
						print("No plate found for boundary id"+str(currentCarID))
					
					
					os.remove("./data/thumbnail.jpg")
					currentCarID = currentCarID + 1

		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			if(LP[carID] != None and len(LP[carID]) != 0):
				cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

			if(LP[carID]!= None and len(LP[carID]) != 0):
				s = str(LP[carID])
				cv2.putText(resultImage, str(s).upper(), (int(t_x + t_w/2), int(t_y-30)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]
		
		end_time = time.time()
		
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)

		for i in carLocation1.keys():	
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = carLocation1[i]
				[x2, y2, w2, h2] = carLocation2[i]
		
				carLocation1[i] = [x2, y2, w2, h2]
		
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
						speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

					#if y1 > 275 and y1 < 285:
					if speed[i] != None and y1 >= 180:
						# print(i)
						if(int(speed[i]) > SPEED_LIMIT and LP[i] != None and len(LP[i]) != 0):
							recColor = (0,0,255)
							cv2.rectangle(resultImage, (x2, y2), (x2 + w2, y2 + h2), recColor, 4)
							# call api here and send speed[i] with LP[i] then flask will flag the vehicle in database
							if(LP[i]!=None and len(LP[i]) != 0):
								try:
									
									cursor = conn.cursor()
									postgreSQL_select_Query = "select from speed_list where Number_Plate = %s"
									cursor.execute(postgreSQL_select_Query, (LP[i],))
									records = cursor.fetchall()
										

									if len(records)== 0:
										print("The Vehicle details have been filed for violating the Speed Limit")
										postgreSQL_insert_Query = "insert into speed_list(Number_Plate,Type,Speed) values(%s,%s,%s)"
										TYPE = "Car"
										cursor.execute(postgreSQL_insert_Query, (LP[i],TYPE,speed[i]))
										conn.commit()
										print("Speed inserted in Speed List DB")
									#print("Speed queries completed")
							

								except (Exception, psycopg2.Error) as error :
									print ("Error while fetching data from PostgreSQL", error)

						
						if(LP[i] != None and len(LP[i]) != 0):
							cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


		cv2.imshow('result', resultImage)
		# Write the frame into the file 'output.avi'
		#out.write(resultImage)


		if cv2.waitKey(33) == 27:
			break
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()
	conn.close()
