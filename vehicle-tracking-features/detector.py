import numpy as np
import color.image_processor as colorDetector

class Detector(object):
    def __init__(self):
        self.model = None
    
    def setModel(self,model):
        self.model = model

    def detectObject(self,image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.model.detection_graph.get_tensor_by_name("image_tensor:0")

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.model.detection_graph.get_tensor_by_name("detection_boxes:0")

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.model.detection_graph.get_tensor_by_name("detection_scores:0")
        classes = self.model.detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = self.model.detection_graph.get_tensor_by_name("num_detections:0")

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.model.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded},
        )

        return (np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))

    def detectColor(self,img):
        return colorDetector.process_image(img)  