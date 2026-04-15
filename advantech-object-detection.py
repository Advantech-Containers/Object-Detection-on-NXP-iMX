#!/usr/bin/env python3

"""
Object Detection with NNStreamer + TFLite
---------------------------------------------

This script runs Object Detection using GStreamer pipelines
with NNStreamer tensor_filter (TFLite backend). It supports CPU and NPU backends
on i.MX platforms and can render results with Wayland/X11/cairooverlay.

Author: <Your Name/Org>
License: MIT
"""


import os
import sys
import gi
import logging
import math
import numpy as np
import cairo
import argparse
import signal
import pathlib
import subprocess
import re
import glob

gi.require_version('Gst', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GLib

DEBUG = False

# ============================================================
#                    IOU TRACKER
# ============================================================
class IOUTracker:
    """IOU tracker to stabilize bounding boxes across frames."""
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.prev = {}       # id → [x,y,w,h]
        self.next_id = 1

    @staticmethod
    def iou(a, b):
        """Compute IOU of two boxes (x,y,w,h)."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b

        ax2 = ax1 + aw
        ay2 = ay1 + ah
        bx2 = bx1 + bw
        by2 = by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter_area = iw * ih

        if inter_area <= 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        return inter_area / (area_a + area_b - inter_area)

    def match(self, new_boxes):
        """
        Match new_boxes (list of dicts) to previous stored boxes
        and assign persistent IDs.
        """
        assignments = {}
        used_prev = set()

        # For each new box, find best IOU match
        for d in new_boxes:
            best_iou = 0.0
            best_id = None

            new_box = [d['x'], d['y'], d['width'], d['height']]

            for pid, pbox in self.prev.items():
                if pid in used_prev:
                    continue

                iou_val = self.iou(new_box, pbox)

                if iou_val > best_iou and iou_val >= self.iou_threshold:
                    best_iou = iou_val
                    best_id = pid

            # If matched → reassign previous ID
            if best_id is not None:
                assignments[d['temp_id']] = best_id
                used_prev.add(best_id)

            else:
                # new ID
                new_id = self.next_id
                self.next_id += 1
                assignments[d['temp_id']] = new_id
                used_prev.add(new_id)

        # Update self.prev for next frame
        self.prev = {}
        for d in new_boxes:
            oid = assignments[d['temp_id']]
            self.prev[oid] = [
                d['x'], d['y'], d['width'], d['height']
            ]

        return assignments


# ============================================================
#                 OBJECT DETECTION CLASS
# ============================================================
class ObjectDetection:
    """The class that manages the demo"""
    def __init__(
        self, platform, device, backend, model, labels, save_video,
        display, width,height,c_format, 
        callback=None, r=1, g=0, b=0):

        self.loop = None
        self.pipeline = None
        self.running = False
        self.video_caps = None
        self.first_frame = True

        # SSD parameters
        self.BOX_SIZE = 4
        self.LABEL_SIZE = 91
        self.DETECTION_MAX = 20
        self.MAX_OBJECT_DETECTION = 20

        self.VIDEO_WIDTH = width
        self.VIDEO_HEIGHT = height
        self.MODEL_WIDTH = 300
        self.MODEL_HEIGHT = 300

        self.tflite_model = model
        self.label_path = labels
        self.device = device
        self.backend = backend
        self.display = display
        self.video = save_video
        self.c_format = c_format
        self.tflite_labels = []
        self.detected_objects = []
        self.callback = callback
        self.r = r
        self.b = b
        self.g = g
        self.platform = platform

        # ---------------------------
        # Hybrid Flicker Reduction
        # ---------------------------
        self.iou_tracker = IOUTracker(iou_threshold=0.35)

        self.prev_smoothed = {}       # id → smoothed box array
        self.smooth_alpha = 0.35      # 0.25–0.35 recommended
        self.score_threshold = 0.50   # remove low-score jitters

        if not self.tflite_init():
            raise Exception("Model init failed")

        GObject.threads_init()
        Gst.init(None)

    def _select_bitrate(self):
        px = self.VIDEO_WIDTH * self.VIDEO_HEIGHT
        if px >= 1920 * 1080:
            return 8000   # 8 Mbps
        elif px >= 1280 * 720:
            return 5000   # 5 Mbps
        else:
            return 2500   # SD



    # Decodebin pad handler → link to hardware vpudec
    def on_decodebin_pad(self, dbin, pad):
        caps = pad.get_current_caps().to_string()

        # H264 input from MP4
        if "video/x-h264" in caps:
            h264parse = Gst.ElementFactory.make("h264parse", "h264_parse")
            decoder = Gst.ElementFactory.make("vpudec", "vpu_decoder")
            conv = self.pipeline.get_by_name("pre_tee_convert")

            self.pipeline.add(h264parse)
            self.pipeline.add(decoder)

            pad.link(h264parse.get_static_pad("sink"))
            h264parse.link(decoder)
            decoder.link(conv)

            h264parse.sync_state_with_parent()
            decoder.sync_state_with_parent()

        # Raw video → directly to conv
        elif "video/x-raw" in caps:
            conv = self.pipeline.get_by_name("pre_tee_convert")
            pad.link(conv.get_static_pad("sink"))

    # Dynamic pad handler for MP4/H264 (qtdemux)
    def on_pad_added(self, demux, pad):
        caps = pad.get_current_caps().to_string()
        if "video" in caps:
            decoder = self.pipeline.get_by_name("vpu_decoder")
            sinkpad = decoder.get_static_pad("sink")
            if not sinkpad.is_linked():
                pad.link(sinkpad)


    def tflite_init(self):
        """
        Initialize TFLite labels.
        """
        if not os.path.exists(self.tflite_model):
            logging.error('cannot find tflite model [%s]', self.tflite_model)
            return False

        try:
            with open(self.label_path, 'r') as f:
                for line in f.readlines():
                    if line[0].isdigit():
                        # numbered labels
                        while str(len(self.tflite_labels)) not in line:
                            self.tflite_labels.append("Invalid")
                        self.tflite_labels.append(line[line.find(' ')+1:])
                    else:
                        # plain labels
                        self.tflite_labels.append(line.strip())
        except FileNotFoundError:
            logging.error('cannot find tflite label [%s]', self.label_path)
            return False

        logging.info('finished to load labels, total [%d]', len(self.tflite_labels))
        return True


    def run_code(self):
        """Starts pipeline and runs detection loop"""

        # --------------------------
        # Backend configuration
        # --------------------------
        if self.backend == "CPU":
            backend = "true:CPU custom=NumThreads:4"
        elif self.backend == "GPU":
            os.environ["USE_GPU_INFERENCE"] = "1"
            backend = ("true:gpu custom=Delegate:External,"
                       "ExtDelegateLib:libvx_delegate.so")
        else:
            os.environ["USE_GPU_INFERENCE"] = "0"
            backend = ("true:npu custom=Delegate:External,"
                       "ExtDelegateLib:libvx_delegate.so")

        # --------------------------
        # Save video block
        # --------------------------
        save_video = ""
        if self.video == "yes":
            save_video = (
                " tt. ! queue max-size-buffers=16 leaky=no "
                " ! videoconvert "
                " ! videorate "
                " ! video/x-raw,format=NV12,framerate=30/1 "
                " ! queue max-size-buffers=16 leaky=no "
                f" ! vpuenc_h264 bitrate={self._select_bitrate()} gop-size=30 "
                " ! h264parse "
                " ! mp4mux faststart=true "
                " ! filesink location=output_video.mp4 sync=false "
            )

        # --------------------------
        # Display sink
        # --------------------------
        if self.display == "X11":
            display = "ximagesink name=img_tensor "
        elif self.display == "None":
            self.print_time = GLib.get_monotonic_time()
            display = "fakesink "
        else:
            display = "waylandsink sync=true name=img_tensor "

        # --------------------------
        # Setup main loop
        # --------------------------
        self.loop = GObject.MainLoop()
        self.old_time = GLib.get_monotonic_time()
        self.update_time = GLib.get_monotonic_time()
        self.reload_time = -1
        self.interval_time = 999999

        # --------------------------
        # Platform decoder
        # --------------------------
        if self.platform == "imx8qmmek":
            decoder = "h264parse ! v4l2h264dec ! imxvideoconvert_g2d "
        else:
            decoder = "vpudec "

        # --------------------------
        # Source selection
        # --------------------------
        if self.device.startswith("rtsp://"):
            gst_launch = (
                f"rtspsrc location={self.device} ! "
                "rtph264depay ! vpudec ! videoconvert ! tee name=t "
            )

        elif "/dev/video" in self.device and self.c_format == "YUY2":
            gst_launch = (
                f"v4l2src device={self.device} ! videoconvert ! video/x-raw,width={self.VIDEO_WIDTH},height={self.VIDEO_HEIGHT},format={self.c_format} ! tee name=t "
            )

        elif "/dev/video" in self.device:
            gst_launch = (
                f"v4l2src device={self.device} ! imxvideoconvert_g2d ! video/x-raw,width={self.VIDEO_WIDTH},height={self.VIDEO_HEIGHT},format={self.c_format} ! tee name=t "
            )

        else:
            gst_launch = (
                f"filesrc location={self.device} "
                f"! decodebin name=dec "
                f"! videoconvert name=pre_tee_convert "
                f"! tee name=t "
            )

        # --------------------------
        # Tensor pipeline
        # --------------------------
        gst_launch += (
            f" t. ! imxvideoconvert_g2d ! video/x-raw,width={self.MODEL_WIDTH},height={self.MODEL_HEIGHT},format=ARGB "
            " ! imxvideoconvert_g2d ! queue max-size-buffers=2 leaky=2 "
            " ! videoconvert ! video/x-raw,format=RGB ! tensor_converter "
            f" ! tensor_filter framework=tensorflow2-lite model={self.tflite_model} accelerator={backend} silent=FALSE name=tensor_filter latency=1 "
            " ! tensor_sink name=tensor_sink "
            " t. ! imxvideoconvert_g2d ! cairooverlay name=tensor_res "
            " ! tee name=tt "
            " tt. ! queue max-size-buffers=2 leaky=2 ! "
            + display +
            save_video
        )

        print(gst_launch)

        # --------------------------
        # Start GStreamer
        # --------------------------
        self.pipeline = Gst.parse_launch(gst_launch)

        decodebin = self.pipeline.get_by_name("dec")
        if decodebin is not None:
            decodebin.connect("pad-added", self.on_decodebin_pad)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        tensor_sink = self.pipeline.get_by_name("tensor_sink")
        tensor_sink.connect("new-data", self.new_data_cb)

        tensor_res = self.pipeline.get_by_name("tensor_res")
        tensor_res.connect("draw", self.draw_overlay_cb)
        tensor_res.connect("caps-changed", self.prepare_overlay_cb)

        if self.callback:
            GObject.timeout_add(500, self.callback, self)

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        self.set_window_title(
            'img_tensor', 'Object Detection Example')

        # --------------------------
        # Graceful Ctrl+C handler
        # --------------------------
        def handle_sigint(sig, frame):
            print("\nCtrl+C detected → Sending EOS to finalize MP4...")
            self.pipeline.send_event(Gst.Event.new_eos())

            msg = bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS)
            if msg:
                print("EOS OK")
            else:
                print("EOS timeout, force exit")

            self.loop.quit()

        signal.signal(signal.SIGINT, handle_sigint)

        # ---- Run main loop ----
        try:
            self.loop.run()
        finally:
            self.running = False
            self.pipeline.set_state(Gst.State.NULL)
            bus.remove_signal_watch()
            sys.exit(0)


    # ============================================================
    #                 NEW DATA CALLBACK
    # ============================================================
    def new_data_cb(self, sink, buffer):
        if not self.running:
            return

        if buffer.n_memory() != 4:
            return False

        # ---- BOXES ----
        mem_boxes = buffer.peek_memory(0)
        ret, info_boxes = mem_boxes.map(Gst.MapFlags.READ)
        if not ret:
            return
        boxes = np.frombuffer(info_boxes.data, dtype=np.float32)

        # ---- CLASSES ----
        mem_det = buffer.peek_memory(1)
        ret, info_det = mem_det.map(Gst.MapFlags.READ)
        detections = np.frombuffer(info_det.data, dtype=np.float32)

        # ---- SCORES ----
        mem_scores = buffer.peek_memory(2)
        ret, info_scores = mem_scores.map(Gst.MapFlags.READ)
        scores = np.frombuffer(info_scores.data, dtype=np.float32)

        # ---- NUM DET ----
        mem_num = buffer.peek_memory(3)
        ret, info_num = mem_num.map(Gst.MapFlags.READ)
        num_det = int(np.frombuffer(info_num.data, dtype=np.float32)[0])

        # compute detections
        self.get_detected_objects(boxes, detections, scores, num_det)

        # cleanup
        mem_boxes.unmap(info_boxes)
        mem_det.unmap(info_det)
        mem_scores.unmap(info_scores)
        mem_num.unmap(info_num)

        return True

    # ============================================================
    #            HYBRID SMOOTHING + IOU TRACKING
    # ============================================================
    def get_detected_objects(self, boxes, detections, scores, num):
        """
        Stable detection output:
        - raw SSD → IOU tracker (persistent ID)
        - sliding-window smoothing per tracked ID (12 frames)
        """

        raw_objs = []

        # -------------------------
        # Convert raw SSD output
        # -------------------------
        for i in range(num):
            if scores[i] < self.score_threshold:
                continue

            cid = int(detections[i])
            off = self.BOX_SIZE * i

            ymin = boxes[off + 0]
            xmin = boxes[off + 1]
            ymax = boxes[off + 2]
            xmax = boxes[off + 3]

            x = xmin * self.MODEL_WIDTH
            y = ymin * self.MODEL_HEIGHT
            w = (xmax - xmin) * self.MODEL_WIDTH
            h = (ymax - ymin) * self.MODEL_HEIGHT

            raw_objs.append({
                "temp_id": i,
                "class_id": cid,
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "prob": float(scores[i])
            })

        if not raw_objs:
            self.detected_objects = []
            return

        # -------------------------
        # IOU Tracking (persistent IDs)
        # -------------------------
        assignments = self.iou_tracker.match(raw_objs)

        output_objs = []

        # -------------------------
        # Sliding window smoothing (no flicker)
        # -------------------------
        SMOOTH_N = 5  # 12–20 is ideal for SSD Mobilenet

        for obj in raw_objs:
            oid = assignments[obj["temp_id"]]  # stable track ID

            x, y = obj["x"], obj["y"]
            w, h = obj["width"], obj["height"]

            # unique smoothing buffer per track
            uid = f"{oid}_{obj['class_id']}"

            if uid not in self.prev_smoothed:
                self.prev_smoothed[uid] = []

            buf = self.prev_smoothed[uid]
            buf.append([x, y, w, h])

            if len(buf) > SMOOTH_N:
                buf.pop(0)

            self.prev_smoothed[uid] = buf

            # stable mean smoothing
            smx, smy, smw, smh = np.mean(buf, axis=0)

            output_objs.append({
                "id": oid,
                "class_id": obj["class_id"],
                "x": float(smx),
                "y": float(smy),
                "width": float(smw),
                "height": float(smh),
                "prob": obj["prob"]
            })

        self.detected_objects = output_objs

    # ============================================================
    #                 CAIRO OVERLAY DRAWING
    # ============================================================
    def prepare_overlay_cb(self, overlay, caps):
        """Store caps for drawing."""
        self.video_caps = caps

    def draw_overlay_cb(self, overlay, context, timestamp, duration):
        if not self.running or self.video_caps is None:
            return

        scale_h = self.VIDEO_HEIGHT / 1080
        scale_w = self.VIDEO_WIDTH / 1920
        scale_text = max(scale_h, scale_w)

        objs = self.detected_objects

        # Font setup
        context.select_font_face(
            'Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(int(25.0 * scale_text))
        context.set_source_rgb(0, 0, 1)

        draw_count = 0

        for obj in objs:
            label = self.tflite_labels[obj["class_id"]].strip()

            # map box coordinates to display
            x = obj['x'] * self.VIDEO_WIDTH / self.MODEL_WIDTH
            y = obj['y'] * self.VIDEO_HEIGHT / self.MODEL_HEIGHT
            w = obj['width'] * self.VIDEO_WIDTH / self.MODEL_WIDTH
            h = obj['height'] * self.VIDEO_HEIGHT / self.MODEL_HEIGHT

            # jitter snap (<2px)
            if abs(w) < 2: w = 0
            if abs(h) < 2: h = 0

            # draw rectangle
            context.rectangle(x, y, w, h)
            context.set_line_width(3)
            context.stroke()

            # label
            context.move_to(x, y - 5)
            context.show_text(label)

            draw_count += 1
            if draw_count >= self.MAX_OBJECT_DETECTION:
                break

        # initial loading message (first frame only)
        if self.first_frame:
            context.set_source_rgb(1, 1, 1)
            context.move_to(int(50 * scale_w), int(100 * scale_h))
            context.set_font_size(int(50.0 * min(scale_w, scale_h)))
            context.show_text("Loading...")
            self.first_frame = False
        context.fill()


    # ============================================================
    #                 BUS & MESSAGE HANDLING
    # ============================================================
    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            logging.info('received eos message')
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logging.warning('[error] %s : %s', error.message, debug)
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            warn, dbg = message.parse_warning()
            logging.warning("[WARNING] %s : %s", warn.message, dbg)

        elif message.type == Gst.MessageType.QOS:
            data_format, processed, dropped = message.parse_qos_stats()
            logging.debug(
                "[QOS] processed=%d dropped=%d",
                processed, dropped
            )


    # ============================================================
    #                 SET WINDOW TITLE
    # ============================================================
    def set_window_title(self, name, title):
        element = self.pipeline.get_by_name(name)
        if not element:
            return

        pad = element.get_static_pad('sink')
        if not pad:
            return

        tags = Gst.TagList.new_empty()
        tags.add_value(Gst.TagMergeMode.APPEND, 'title', title)
        pad.send_event(Gst.Event.new_tag(tags))


# ============================================================
#            CAMERA DISCOVERY / FORMAT / RESOLUTION
# ============================================================
ALLOWED_RESOLUTIONS = [
    (640, 360), (640, 480),
    (1280, 720), (1920, 1080)
]

def get_video_nodes():
    return glob.glob("/dev/video*")

def is_streaming(dev, num_buffers=3):
    try:
        subprocess.run(
            ["timeout", "5", "gst-launch-1.0", "-q",
             "v4l2src", f"device={dev}", f"num-buffers={num_buffers}",
             "!", "fakesink"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_format_info(dev):
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        return []

    lines = result.stdout.splitlines()
    formats = []
    current_pix = None

    for line in lines:
        line = line.strip()

        if line.startswith("["):
            m = re.search(r"'(\w+)'", line)
            if m:
                current_pix = m.group(1)

        elif "Discrete" in line and "Size:" in line:
            m = re.search(r"(\d+)x(\d+)", line)
            if m and current_pix:
                w, h = map(int, m.groups())
                if (w, h) in ALLOWED_RESOLUTIONS:
                    if current_pix in ("YUYV", "BGRx", "MJPG"):
                        formats.append((w, h, current_pix))

    return formats


def select_webcam():
    print("Available supported video nodes for streaming:")
    print("-------------------------------------")
    nodes = get_video_nodes()
    working = []

    for dev in nodes:
        if is_streaming(dev):
            fmt = get_format_info(dev)
            if fmt:
                print(f"Node {dev}:")
                for i,(w,h,pix) in enumerate(fmt,1):
                    print(f"   [{i}] {w}x{h}, {pix}")
                working.append((dev, fmt))

    if not working:
        print("No valid camera found.")
        return None

    # choose node
    while True:
        c = input(f"Select video node [1-{len(working)}]: ")
        if c.isdigit() and 1 <= int(c) <= len(working):
            dev, fmts = working[int(c)-1]
            break

    print(f"\nAvailable resolutions for {dev}:")
    for i,(w,h,pix) in enumerate(fmts,1):
        print(f"   [{i}] {w}x{h}, {pix}")

    while True:
        r = input(f"Select resolution [1-{len(fmts)}]: ")
        if r.isdigit() and 1 <= int(r) <= len(fmts):
            w, h, pix = fmts[int(r)-1]
            break

    print("\nSelected camera:")
    print("Device:", dev)
    print("Resolution:", w, "x", h)
    print("Pixel Format:", pix)

    return {
        "device": dev,
        "width": w,
        "height": h,
        "pixfmt": pix,
        "framerate": 30
    }


def get_resolution(filepath):
    """Detect width/height of files or RTSP."""
    try:
        result = subprocess.run(
            ["gst-discoverer-1.0", filepath],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True
        )
    except subprocess.CalledProcessError:
        print("[ERROR] gst-discoverer failed")
        return None, None

    width, height = None, None

    for line in result.stdout.splitlines():
        line = line.strip()
        m1 = re.search(r"Width\s*:\s*(\d+)", line)
        m2 = re.search(r"Height\s*:\s*(\d+)", line)

        if m1:
            width = int(m1.group(1))
        if m2:
            height = int(m2.group(1))

    if width and height:
        print(f"[INFO] Detected: {width}x{height}")
        return width, height

    print("[WARNING] Could not detect resolution.")
    return None, None

# ============================================================
#                          MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Detection (NNStreamer + TFLite)")
    parser.add_argument("--video", default="videos/detect_example.mov",help="Video source: 'discover', RTSP URL (rtsp://<ip>:<port>/<path>), or video file path")
    parser.add_argument("--platform", default="imx8mp", help="Target platform")

    parser.add_argument("--backend", choices=["CPU", "NPU"], default="CPU", help="Inference backend")
    parser.add_argument("--model", default="models/ssd_mobilenet_v2_coco_quant_postprocess.tflite",
                        help="TFLite model path")
    parser.add_argument("--labels", default="models/coco_labels.txt", help="Labels file")
    parser.add_argument("--save-video", choices=["yes", "no"], default="no",
                        help="Save output video (default: no)")
    parser.add_argument("--display", choices=["X11", "weston"], default="weston",
                        help="Display backend (default: weston)")
    parser.add_argument("--cam-width", type=int, default=1920,
                        help="Camera input width (default: 1920)")
    parser.add_argument("--cam-height", type=int, default=1080,
                        help="Camera input height (default: 1080)")
    parser.add_argument("--cam-format", default="BGRx",
                        help="Camera pixel format (default: BGRx)")
    
    args = parser.parse_args()
    if args.display == "X11":
        args.save_video = "no"

    # ============================================================
    #                       VIDEO SOURCE
    # ============================================================
    if args.video.lower() == "discover":
        cam = select_webcam()
        if cam is None:
            print("[ERROR] No valid camera found.")
            sys.exit(1)

        args.video = cam["device"]
        w = cam["width"]
        h = cam["height"]
        args.cam_format = cam["pixfmt"]

        # normalize pixel format
        if args.cam_format.upper() in ("YUYV","YUY2"):
            args.cam_format = "YUY2"
        elif args.cam_format.upper() in ("GREY","GRAY8"):
            args.cam_format = "GRAY8"
        else:
            args.cam_format = "BGRx"

        print(f"[INFO] Selected Camera: {args.video}")
        print(f"[INFO] Resolution: {w}x{h}, Format: {args.cam_format}")

            
    elif args.video.startswith("rtsp://") or not args.video.startswith("/dev/video"):
        w, h = get_resolution(args.video)
        if not w or not h:
            w, h = args.cam_width, args.cam_height
            print(f"[WARNING] Could not detect resolution, using defaults {w}x{h}")
        print(f"Stream/Video selected -> {args.video}, resolution: {w}x{h}")
    
    elif args.video.startswith("/dev/video"):
        # Direct camera device: use provided width/height
        w, h = args.cam_width, args.cam_height
        print(f"Direct camera device detected -> {args.video}, resolution {w}x{h}")

    if not w or not h or w <= 0 or h <= 0:
        print(f"[ERROR] Invalid resolution detected: width={w}, height={h}")
        sys.exit(1)

    # Print configuration
    print("========== Configuration ==========")
    print(f"Video Source : {args.video}")
    print(f"Platform     : {args.platform}")
    print(f"Backend      : {args.backend}")
    print(f"Model File   : {args.model}")
    print(f"Label File   : {args.labels}")
    print(f"Save Video   : {args.save_video}")
    print(f"Display      : {args.display}")
    print(f"Width        : {w}")
    print(f"Height       : {h}")
    print(f"Format       : {args.cam_format}")
    print("===================================")

    # ============================================================
    #               START DETECTION ENGINE
    # ============================================================
    engine = ObjectDetection(
        platform=args.platform,
        device=args.video,
        backend=args.backend,
        model=args.model,
        labels=args.labels,
        save_video=args.save_video,
        display=args.display,
        width=w,
        height=h,
        c_format=args.cam_format
    )

    engine.run_code()
