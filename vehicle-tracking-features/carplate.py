import cv2
import boto3


if __name__ == "__main__":
	client=boto3.client('rekognition','us-east-2')
	outputVideoFPS = 60
	video_src = 'video/A0026.mpg'
	video_capture = cv2.VideoCapture(video_src)
	frame_count = 0
	frame_interval = 30

	saveVideo = 'output/output.mp4'
	fourcc  = cv2.VideoWriter_fourcc(*'MP4V')
	# out = cv2.VideoWriter(saveVideo,fourcc,20.0,(640,480))
	first = True
	num = 0
	carplate = None

	while True:  # fps._numFrames < 120
		num += 1
		# print(num)
		flags,frame = video_capture.read()


		if flags == False:
			break

		im_height, im_width, _ = frame.shape

		if first:
			out = cv2.VideoWriter(saveVideo,fourcc,outputVideoFPS,(im_width,im_height))
			first = False

		if (frame_count%frame_interval)==0:
			cropRegion = frame[319:492,420:666]

			enc = cv2.imencode('.png',cropRegion)[1].tostring()

			response= client.detect_text(
				Image={
					'Bytes': enc,
				}
			)
								
			textDetections=response['TextDetections']
			# print(response)
			alpha = None
			letter = None
			for text in textDetections:
				word = text['DetectedText']
				print('Detected text:' + word)
				if len(word)<3:
					continue
				if len(word)>6 and len(word)<9:
					carplate = word
					break
				if word.isalpha():
					alpha = word
					continue
				if word.isdigit():
					letter = word
					continue
				# print('Confidence: ' + "{:.2f}".format(text['Confidence']) + "%")
				# print('Id: {}'.format(text['Id']))
				# if 'ParentId' in text:
				# 	print('Parent Id: {}'.format(text['ParentId']))
				# print('Type:' + text['Type'])
				# print()
				# bbox = text['Geometry']['BoundingBox']
				# x1 = bbox['Left']
				# y1 = bbox['Top']
				# x2 = x1+bbox['Width']
				# y2 = y1+bbox['Height']

				# cv2.rectangle(frame, (int(x1),int(y1)),
				# 			  (int(x2),int(y2)),
				# 			  (255,0,0),2)
			if alpha is not None and letter is not None:
				carplate = alpha+" "+letter
		textT = 'License plate: '+str(carplate)

		cv2.putText(frame, textT, (10,50),
					cv2.FONT_HERSHEY_SIMPLEX,
					1, (255, 0, 255), 2)

		cv2.rectangle(frame, (420,319),
					  (666,492),
					  (255,0,0),2)            


		out.write(frame)

		cv2.imshow('Video',frame)

		frame_count += 1
		# cv2.imshow('Video', output_bgr)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	out.release()
	cv2.destroyAllWindows()
