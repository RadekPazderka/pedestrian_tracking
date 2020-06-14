import cv2
import numpy as np

# YOLO weights:
#   wget https://pjreddie.com/media/files/yolov3.weights

class Detector():
    SCALE = 1 / 255.0

    CONF_THRESHOLD = 0.81
    NMS_THRESHOLD = 0.4

    def __init__(self, yolo_weights_path, yolo_config_path):

        self._net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def detect_image(self, frame):
        bboxes = []

        self._frame_width = frame.shape[1]
        self._frame_height = frame.shape[0]

        blob = cv2.dnn.blobFromImage(frame, self.SCALE, (416, 416), (0, 0, 0), True, crop=False)
        self._net.setInput(blob)
        outs = self._net.forward(self.get_output_layers(self._net))

        class_ids = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                # print("class id: {0}".format(class_id))
                confidence = scores[class_id]
                if class_id == 0 and confidence > self.CONF_THRESHOLD:
                    center_x = int(detection[0] * self._frame_width)
                    center_y = int(detection[1] * self._frame_height)
                    w = int(detection[2] * self._frame_width)
                    h = int(detection[3] * self._frame_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    bboxes.append((int(x), int(y), int(w), int(h)))

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.CONF_THRESHOLD, self.NMS_THRESHOLD)

        for i in indices:
            i = i[0]
            box = bboxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self._draw_prediction(frame, confidences[i], round(x), round(y), round(x + w), round(y + h))

        return True, bboxes, frame


    def _draw_prediction(self, img, confidence, x, y, x_plus_w, y_plus_h):
        label = "Person"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color=color, thickness=2)
        cv2.putText(img, "{0}:{1:.2f}".format(label, confidence), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=color, thickness=1)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers


