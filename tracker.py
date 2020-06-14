import cv2

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

class Track():

    def __init__(self, tracker_name, first_frame, bbox, id):
        self._tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        self._bbox = bbox
        self._tracker.init(first_frame, bbox)
        self._frame_height, self._frame_width, _ = first_frame.shape
        self._id = id

    def update(self, frame):
        success, self._bbox = self._tracker.update(frame)

    def draw_bbox(self, frame):
        (x, y, w, h) = [int(v) for v in self._bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, "{0}".format(self._id), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), thickness=2)

    def is_finish_track(self):
        bb_area = self._bbox[2] * self._bbox[3]

        xmin = max(0, self._bbox[0])
        ymin = max(0, self._bbox[1])
        xmax = min(self._frame_width, self._bbox[0] + self._bbox[2])
        ymax = min(self._frame_height, self._bbox[1] + self._bbox[3])

        bb_inner_area = (xmax - xmin) * (ymax - ymin)

        percent_in_area = bb_inner_area / bb_area
        if percent_in_area < 0.8:
            return True
        return False

    def get_bbox(self):
        return self._bbox

    def update_bbox(self, bbox):
        self._bbox = bbox

    def get_id(self):
        return self._id

    def check_bb_size(self):
        if (self._bbox[2] > self._frame_width / 3) or (self._bbox[3] > self._frame_height / 3):
            return False
        return True

class Tracker():
    TRACKER_TYPE = "csrt"
    CONF_THRESHOLD = 0.8
    NMS_THRESHOLD = 0.1

    def __init__(self):
        self._trackers = []
        self._last_bboxes = None
        self._track_id = 0

    def refresh_bbox(self, bboxes, better_bb_index):
        import operator
        bb1 = tuple(map(operator.mul, bboxes[better_bb_index], (.6, .6, .6, .6)))
        bb2 = tuple(map(operator.mul, bboxes[int(not better_bb_index)], (.4, .4, .4, .4)))
        return tuple(map(operator.add, bb1, bb2))


    def update_trackers_by_dets(self, frame, bboxes):
        for bbox in bboxes:
            add_new = True
            for tr in self._trackers:
                bb = [bbox, tr.get_bbox()]
                indicates = cv2.dnn.NMSBoxes(bb, [1.,.9], self.CONF_THRESHOLD, self.NMS_THRESHOLD)
                if indicates.size == 1:
                    add_new = False

                    new_bbox = self.refresh_bbox(bb, indicates[0][0])
                    tr.update_bbox(new_bbox)

            if add_new:
                new_track = Track("csrt", frame, bbox, self._track_id)
                if not new_track.is_finish_track() and new_track.check_bb_size():
                    self._trackers.append(new_track)
                    self._track_id += 1


    def track(self, frame):

        for track in self._trackers:
            track.update(frame)

        for track in self._trackers:
            track.draw_bbox(frame)

        def f(tr):
            return not tr.is_finish_track()

        self._trackers = list(filter(f, self._trackers))



        return frame