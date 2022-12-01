import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

video_path = "line1.mp4"


def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                    score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only=[]):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path)  # detect on video
    else:
        vid = cv2.VideoCapture(0)  # detect from webcam

    # by default VideoCapture returns float instead of int
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (frame_width, height))  # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    # cam: adding code to detect ppl crossing certain point
    # this variable stores the index of the vertical line used as the threshold for counting someone as "moved"
    thresh_right = int(frame_width / 3)
    thresh_left = int(frame_width * 2 / 3)
    # note: we only want to count people who move in the NEGATIVE x direction
    total_crossers = 0
    # create a dictionary of people identified by #, with both their previous x-left and current x-left
    # we also want to store the avg delta as part of our "who is in line?" decision logic
    #             id: [prev, curr, avg delta]
    # persons = { 10: [352, 346, -6],
    #             45: [901, 852, -49],
    #             etc. }
    # idea is that if prev_x > x_thresh and curr_x <= x_thresh, we add them to tally of ppl who crossed
    persons = {}

    # if we have a video of people in line, most people are going to be moving at the same speed
    # let's find that most common speed, and if people are within a range of that, then we can with confidence say they're in line
    # build on the previous dictionary
    curr_mid_ppl = []
    curr_left_ppl = []
    prev_mid_ppl = []
    prev_left_ppl = []
    tput_history = []

    top_slope = 0
    top_int = 0
    bot_slope = 0
    bot_int = 0

    top_line_start = 0
    top_line_end = 0
    bot_line_start = 0
    bot_line_end = 0

    if "line1" in video_path:
        speed_up = 10
        range_val = 300
    elif "line2" in video_path:
        speed_up = 1
        range_val = 300
    else:
        range_val = int(fps * vid.get(cv2.CAP_PROP_FRAME_COUNT))
        speed_up = 1

    for i in range(range_val):
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        for id in persons.keys():
            # update average horizontal motion
            persons[id][2] = (persons[id][2] + (persons[id][1] - persons[id][0])) / 2
            # update previous x to be current x
            persons[id][1] = persons[id][0]

        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        # image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        # t1 = time.time()
        # pred_bbox = Yolo.predict(image_data)
        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) != 0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
                              bbox[3].astype(int) - bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        not_in_line_bboxes = []
        if i != 0 and i % 10 == 0:
            prev_left_ppl = curr_left_ppl
            prev_mid_ppl = curr_mid_ppl
        curr_left_ppl = []
        curr_mid_ppl = []
        # if i >= 10:
        if i == 10:
            top_X = np.array([])
            top_y = np.array([])
            bot_X = np.array([])
            bot_y = np.array([])
            top_line_start = 0
            top_line_end = 0
            heights =[]
            for track in tracker.tracks:
                tl_x, tl_y, width, height = track.to_tlwh()
                top_X = np.append(top_X, tl_x + int(width/2))
                top_y = np.append(top_y, tl_y)
                bot_X = np.append(bot_X, tl_x + int(width/2))
                bot_y = np.append(bot_y, tl_y + height)
                heights.append(height)

            heights = np.array(heights)
            max_height = np.median(heights)

            top_X = top_X.reshape(-1, 1)
            top_y = top_y.reshape(-1, 1)
            bot_X = bot_X.reshape(-1, 1)
            bot_y = bot_y.reshape(-1, 1)
            top_reg = LinearRegression().fit(top_X, top_y)
            bot_reg = LinearRegression().fit(bot_X, bot_y)
            top_slope = top_reg.coef_[0, 0]
            top_int = top_reg.intercept_[0] - max_height * 0.125
            bot_slope = bot_reg.coef_[0, 0]
            bot_int = bot_reg.intercept_[0] + max_height * 0.125

            top_line_start = (0, int(top_int))
            top_line_end = (int(frame_width), int(top_slope * frame_width + top_int))
            bot_line_start = (0, int(bot_int))
            bot_line_end = (int(frame_width), int(bot_slope * frame_width + bot_int))

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 50:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            bbox_height = bbox[3] - bbox[1]
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
            in_line = bbox[3] < bot_slope * bbox[2] + bot_int and bbox[1] > top_slope * bbox[
                0] + top_int  # bbox1 = y value of top left and bbox3 is y value of the bottom

            # people NOT in line
            if not in_line:
                not_in_line_bboxes.append(bbox.tolist() + [tracking_id, index])

            # people in line
            else:
                tracked_bboxes.append(bbox.tolist() + [tracking_id,
                                                       index])  # Structure data, that we could use it with our draw_bbox function
                if bbox[0] < thresh_left:
                    curr_left_ppl.append(tracking_id)
                if bbox[0] < thresh_right:
                    curr_mid_ppl.append(tracking_id)

        diff_left = list(set(curr_left_ppl) - set(prev_left_ppl))
        diff_mid = list(set(curr_mid_ppl) - set(prev_mid_ppl))

        throughput = (len(diff_left) + len(diff_mid)) / (2/(fps*speed_up))
        tput_history.append(throughput)
        if len(tput_history) > 30:
            tput_history = tput_history[1:]
            # # add everyone who should be tracked as being in line
            # if i == 0 and bbox[0] > x_thresh:
            #     persons[tracking_id] = [bbox[0]]
            # # now we will have everyone to the right of the threshold's locations and start to track average delta x's
            # if i == 1:
            #     for id in persons:
            #         persons[id].append(bbox[0])
            #         persons[id].append(persons[id][1] - persons[id][0])
            # # on the 50th iteration, stop tracking people who aren't moving left
            # if i == 50:
            #     for id in persons:
            #         if persons[id][2] > 0:
            #             persons.pop(id)

        # draw people in line on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True, in_line=True)
        # draw people not in line
        image = draw_bbox(image, not_in_line_bboxes, CLASSES=CLASSES, tracking=True, in_line=False)

        if i >= 10:
            image = cv2.line(image, top_line_start, top_line_end, (255, 0, 0), 2)
            image = cv2.line(image, bot_line_start, bot_line_end, (255, 0, 0), 2)

        t3 = time.time()
        times.append(t2 - t1)
        times_2.append(t3 - t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times) / len(times) * 1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)
        if len(tput_history) != 0:
            print(tput_history)
            tput_mean = sum(tput_history) / len(tput_history)
            image = cv2.putText(image, "Throughput: {:.3f}".format(tput_mean), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1,
                                (0, 0, 255), 2)
            if tput_mean != 0:
                image = cv2.putText(image, "Wait time: {:.2f} seconds".format(len(tracked_bboxes) / tput_mean), (0, 60),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

        # draw original yolo detection
        # image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


# cam-br0wn: added entry point for cleanliness
if __name__ == '__main__':
    yolo = Load_Yolo_model()
    Object_tracking(yolo, video_path, "track.mp4", input_size=YOLO_INPUT_SIZE, show=False, iou_threshold=0.1,
                    rectangle_colors=(255, 0, 0), Track_only=["person"])
    exit(0)
