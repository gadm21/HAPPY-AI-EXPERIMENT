import cv2
import boto3
import argparse
import sys

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # add your video link in 'default' parameter
    parser.add_argument("--video", default="video/vlc-record-2019-01-07-22h41m24s-Ainsdale-K09.mp4-.mp4", type=str, help="video path/ip address")
    parser.add_argument(
        "--width", default=0, type=int, help="the width of video output"
    )
    parser.add_argument(
        "--height", default=0, type=int, help="the height of video output"
    )
    parser.add_argument(
        "--output", default="output/test.mov", type=str, help="output video path"
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    client = boto3.client("rekognition", "us-east-2")    
    args = parse_arguments(sys.argv[1:])

    if args.video is None:
        video_src = 0
    else:
        video_src = args.video

    cap = cv2.VideoCapture(video_src)

    # resize video if required
    resize = False # default is False
    # args.width = 1024 # uncomment if resize is True
    # args.height = 790 # uncomment if resize is True

    if args.width > 0:
        width = args.width
        resize = True
    else:    
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if args.height > 0:
        height = args.height
        resize = True
    else:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = args.output

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output, fourcc, fps, (width, height)
    )

    carplate = None

    # define what frame interval and crop region based on video width, height and fps
    print(width, height, fps)

    frame_count = 0
    frame_interval = 10

    # define your crop region here
    x1 = 300
    y1 = 350
    x2 = 600
    y2 = 500

    while True:
        flags, frame = cap.read()

        if flags == False:
            break

        if resize:
            frame = cv2.resize(frame, (width, height))

        if (frame_count % frame_interval) == 0:
            cropRegion = frame[y1:y2, x1:x2]

            enc = cv2.imencode(".png", cropRegion)[1].tostring()

            response = client.detect_text(Image={"Bytes": enc})

            textDetections = response["TextDetections"]

            alpha = None
            letter = None

            for text in textDetections:
                word = text["DetectedText"]
                # print("Detected text:" + word)
                if len(word) < 3:
                    continue
                if len(word) > 6 and len(word) < 9: #and response["Confidence"] > 60:
                    carplate = word
                    print(carplate)
                    break
                if word.isalpha():
                    alpha = word
                    continue
                if word.isdigit():
                    letter = word
                    continue

            if alpha is not None and letter is not None:
                carplate = alpha + " " + letter
                print(carplate)

        textT = "License plate: " + str(carplate)

        cv2.putText(
            frame, textT, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Video", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()    
