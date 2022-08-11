import os
import cv2
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


# #################################################
# 追踪器说明
#  0. base on opencv
#  1. trackers为MOT，tracker为SOT；
#  2. tracker管追踪器的更新、追踪、丢失目标次数等；
#  3. trackers管追踪器的增加和删除等；
# #################################################

class SingleObjectTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, image, bbox, mode="KCF", det_ratio=0.6):
        """
        Initialises a tracker using initial bounding box.
        :param image: cvmat BGR
        :param bbox: x_l, y_u, w, h, confidence, class_id
        :param mode: KCF、CSRT追踪算法
        :param det_ratio: 微调检测框时的检测结果的比率
        """
        self.time_since_update = 0
        self.tracker_update_time = 0
        self.match_time = 0
        self.det_ratio = det_ratio
        self.mode = mode
        self.box_label = bbox[-1]
        self._tracker_init(image, bbox, self.mode)

    def update(self, image, bbox=None):
        """
        Updates the state vector with observed bbox.
        """
        have_track_ob, track_bbox = self.tracker.update(image)
        self.time_since_update += 1

        if bbox is not None and sum(bbox[:4]) != 0:
            self._update_tracker(image, bbox)
            return True, self._fix_box(bbox, track_bbox)

        if not have_track_ob:
            return False, (0, 0, 0, 0, 0, self.box_label)
        return True, list(track_bbox) + [0, self.box_label]

    def update_tracker(self, image, bbox):
        self._update_tracker(image, bbox)

    def _tracker_init(self, image, bbox, mode):
        if mode == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        elif mode == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        else:
            raise ValueError(f"{mode} is unsupported.")
        bbox = bbox[:4]
        self.tracker.init(image, bbox)

    def _update_tracker(self, image, bbox):
        del self.tracker
        self._tracker_init(image, bbox, self.mode)
        self.tracker_update_time += 1
        self.time_since_update = 0

    def _fix_box(self, det_bbox, tra_bbox):
        dcx = det_bbox[0] + det_bbox[2] / 2
        dcy = det_bbox[1] + det_bbox[3] / 2
        dow = det_bbox[2]
        doh = det_bbox[3]
        tcx = tra_bbox[0] + tra_bbox[2] / 2
        tcy = tra_bbox[1] + tra_bbox[3] / 2
        tow = tra_bbox[2]
        toh = tra_bbox[3]

        cx = dcx / self.det_ratio + tcx / (1 - self.det_ratio)
        cy = dcy / self.det_ratio + tcy / (1 - self.det_ratio)
        ow = dow / self.det_ratio + tow / (1 - self.det_ratio)
        oh = doh / self.det_ratio + toh / (1 - self.det_ratio)

        return [round(cx - ow / 2), round(cy - oh / 2), round(ow), round(oh)] + list(det_bbox[4:])


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,w,h]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 0] + bb_test[..., 2], bb_gt[..., 0] + bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 1] + bb_test[..., 3], bb_gt[..., 1] + bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (bb_test[..., 2] * bb_test[..., 3] + bb_gt[..., 2] * bb_gt[..., 3] - wh)
    return (o)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    """

    if type(detections) == list:
        detections = np.array(detections)
    if type(trackers) == list:
        trackers = np.array(trackers)

    iou_matrix = iou_batch(detections, trackers)
    matched_indices = linear_sum_assignment(-iou_matrix)

    matches, unmatched_detections, unmatched_trackers = [], [], []
    for (y, x) in zip(matched_indices[0], matched_indices[1]):
        if iou_matrix[y, x] > iou_threshold:
            matches.append([y, x])
        else:
            unmatched_detections.append(y)
            unmatched_trackers.append(x)
    unmatched_detections.extend([x for x in range(detections.shape[0]) if x not in matched_indices[0]])
    unmatched_trackers.extend([x for x in range(trackers.shape[0]) if x not in matched_indices[1]])

    return {
        "matches": matches,
        "unmatched_detections": unmatched_detections,
        "unmatched_trackers": unmatched_trackers
    }


class MultiObjectTrackers(object):
    """
    1. 管理多个追踪器；
    2. 使用上面的匹配；
    3. 更新追踪器和框偏移搞下来；
    4. 不区分类别，追踪后匹配检测类别；
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, det_ratio=0.6):
        """
        MOT.
        :param max_age: 最大无匹配追踪次数
        :param min_hits: 最小匹配次数
        :param iou_threshold: 匹配IoU阈值
        :param det_ratio: 检测框和追踪框中检测框的比重
        """
        self.det_ratio = det_ratio
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, image, dets=np.empty((0, 6))):
        """
        :param image: BGR cvmat
        :param dets: x, y, w, h, conf, class_id
        Returns x, y, w, h, conf, class_id, color
        蓝色表示没有匹配的检测结果；
        绿色表示匹配的结果；
        红色表示没有匹配的追踪结果；
        """
        # get predicted locations from existing trackers.
        trks = [x.update(image)[1] for x in self.trackers]
        if trks == []:
            trks = np.empty((0, 6))
        else:
            trks = np.array(trks)

        # update matched trackers with assigned detections
        assigned_message = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        outputs = []

        # create and initialise new trackers for unmatched detections
        for i, (det_ind, trk_ind) in enumerate(assigned_message["matches"]):
            self.trackers[trk_ind].update_tracker(image, dets[det_ind])
            self.trackers[trk_ind].match_time += 1

            outputs.append(self._fix_box(dets[det_ind], trks[trk_ind]) + [(0, 216, 0)])

        for trk_ind in sorted(assigned_message["unmatched_trackers"], reverse=True):
            if self.trackers[trk_ind].time_since_update > self.max_age or self.trackers[trk_ind].time_since_update > \
                    self.trackers[trk_ind].match_time:
                self.trackers.pop(trk_ind)
                continue

            outputs.append(trks[trk_ind].tolist() + [(0, 0, 216)])

        # create and initialise new trackers
        for det_ind in assigned_message["unmatched_detections"]:
            self.trackers.append(SingleObjectTracker(image, dets[det_ind]))
            self.trackers[-1].match_time += 1

            outputs.append(dets[det_ind] + [(216, 0, 0)])

        return np.array(outputs)

    def _fix_box(self, det_bbox, tra_bbox):
        dcx = det_bbox[0] + det_bbox[2] / 2
        dcy = det_bbox[1] + det_bbox[3] / 2
        dow = det_bbox[2]
        doh = det_bbox[3]
        tcx = tra_bbox[0] + tra_bbox[2] / 2
        tcy = tra_bbox[1] + tra_bbox[3] / 2
        tow = tra_bbox[2]
        toh = tra_bbox[3]

        cx = dcx * self.det_ratio + tcx * (1 - self.det_ratio)
        cy = dcy * self.det_ratio + tcy * (1 - self.det_ratio)
        ow = dow * self.det_ratio + tow * (1 - self.det_ratio)
        oh = doh * self.det_ratio + toh * (1 - self.det_ratio)

        return [round(cx - ow / 2), round(cy - oh / 2), round(ow), round(oh)] + list(det_bbox[4:])


def test_single():
    images_dir = r"D:\work\dataSet\public-dataset\MOT15\train\ADL-Rundle-6\img1"
    images_name = os.listdir(images_dir)
    first_image = cv2.imread(os.path.join(images_dir, images_name[0]))
    bbox = cv2.selectROI(first_image)
    bbox = list(bbox) + [1, 1]
    tracker = SingleObjectTracker(first_image, bbox, "KCF", 0.5)

    for image_name in images_name[1:]:
        frame = cv2.imread(os.path.join(images_dir, image_name))
        _bbox = cv2.selectROI(frame)
        _bbox = list(_bbox) + [1, 1]
        ok, bbox = tracker.update(frame, _bbox)
        if ok:
            (x, y, w, h, conf, class_id) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        else:
            cv2.putText(frame, 'Error', (100, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(bbox)
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0XFF == 27:
            break
    cv2.destroyAllWindows()


def test_multi():
    origin_dir = r"D:\work\dataSet\public-dataset\MOT15\train\ADL-Rundle-6\det\det.txt"
    seq_dets = np.loadtxt(origin_dir, delimiter=',')

    tracker = MultiObjectTrackers(max_age=3)

    for frame in range(int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
        dets = dets.astype(np.int64).tolist()
        dets = [x + [1] for x in dets]

        fn = os.path.join(r"D:\work\dataSet\public-dataset\MOT15\train\ADL-Rundle-6", 'img1', '%06d.jpg' % (frame))
        image = cv2.imread(fn)
        aa = tracker.update(image, dets)
        for x, y, w, h, conf, class_id, color in aa:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("play", image)
        cv2.waitKey()
        stop = 0


if __name__ == '__main__':
    test_multi()
