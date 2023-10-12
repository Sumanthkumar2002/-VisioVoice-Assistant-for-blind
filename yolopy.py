import cv2
import numpy as np

class YoloDetector:
    def __init__(self, labelsPath, weightsPath, configPath):
        # Load the COCO class labels our YOLO model was trained on
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # Initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                   dtype="uint8")
        # Load the YOLO object detector
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # Get the output layer names
        self.ln = self.net.getUnconnectedOutLayersNames()

    def detect_objects(self, frame, confidence_threshold=0.5, nms_threshold=0.3):
        (H, W) = frame.shape[:2]

        # Create a blob from the input image
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Perform forward pass
        layer_outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        class_ids = []

        # Loop over each of the layer outputs
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    center_x, center_y, width, height = (detection[:4] * np.array([W, H, W, H])).astype(int)
                    x, y = int(center_x - width / 2), int(center_y - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                label = self.LABELS[class_ids[i]]
                confidence = confidences[i]
                box = boxes[i]
                results.append((label, confidence, box))

        return results

    def draw_results(self, frame, results):
        for label, confidence, (x, y, w, h) in results:
            color = [int(c) for c in self.COLORS[self.LABELS.index(label)]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def detect_and_show(self, frame):
        results = self.detect_objects(frame)
        self.draw_results(frame, results)
        cv2.imshow("YOLO Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    labels_path = "coco.names"
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    image_path = "your_image.jpg"

    detector = YoloDetector(labels_path, weights_path, config_path)
    frame = cv2.imread(image_path)
    detector.detect_and_show(frame)
import cv2
import numpy as np

class YoloDetector:
    def __init__(self, labelsPath, weightsPath, configPath):
        # Load the COCO class labels our YOLO model was trained on
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # Initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                   dtype="uint8")
        # Load the YOLO object detector
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # Get the output layer names
        self.ln = self.net.getUnconnectedOutLayersNames()

    def detect_objects(self, frame, confidence_threshold=0.5, nms_threshold=0.3):
        (H, W) = frame.shape[:2]

        # Create a blob from the input image
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # Perform forward pass
        layer_outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        class_ids = []

        # Loop over each of the layer outputs
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    center_x, center_y, width, height = (detection[:4] * np.array([W, H, W, H])).astype(int)
                    x, y = int(center_x - width / 2), int(center_y - height / 2)

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                label = self.LABELS[class_ids[i]]
                confidence = confidences[i]
                box = boxes[i]
                results.append((label, confidence, box))

        return results

    def draw_results(self, frame, results):
        for label, confidence, (x, y, w, h) in results:
            color = [int(c) for c in self.COLORS[self.LABELS.index(label)]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
