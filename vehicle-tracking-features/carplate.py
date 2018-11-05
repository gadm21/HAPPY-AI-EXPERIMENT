import cv2
import boto3
import argparse
import sys

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="video path/ip address")
    parser.add_argument(
        "--width", default=0, type=int, help="the width of video output"
    )
    parser.add_argument(
        "--height", default=0, type=int, help="the height of video output"
    )
    parser.add_argument(
        "--output", default="output/output.mov", type=str, help="output video path"
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

    resize = False

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

    frame_count = 0
    frame_interval = 40

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output, fourcc, fps, (width, height)
    )

    carplate = None

    while True:
        flags, frame = cap.read()

        if flags == False:
            break

        if resize:
            frame = cv2.resize(frame, (width, height))

        if (frame_count % frame_interval) == 0:
            cropRegion = frame[319:492, 420:666]

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
                if len(word) > 6 and len(word) < 9:
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
        cv2.rectangle(frame, (420, 319), (666, 492), (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Video", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()    
