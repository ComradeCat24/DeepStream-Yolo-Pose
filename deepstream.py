import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from ctypes import *
import math
import asyncio
import aiohttp

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = []
CONFIG_INFER = ''
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1280
STREAMMUX_HEIGHT = 720
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

start_time = time.time()
fps_streams = {}
url = os.getenv('URL')
scr2uri = {}
uri_ppl_count = {}
uri_floor_position = {}
uri_homography_mapping = {}


async def async_req():
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=uri_ppl_count) as response:
            # pass
            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])

            html = await response.text()
            print("Body:", html)


class POSTDATA:
    def __init__(self):
        global start_time
        self.start_time = start_time

    def post_data(self):
        end_time = time.time()
        current_time = end_time - self.start_time
        if current_time > PERF_MEASUREMENT_INTERVAL_SEC:
            asyncio.run(async_req())
            self.start_time = time.time()
        else:
            pass


async def async_pos_req():
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=uri_floor_position) as response:
            # pass
            print("Status:", response.status)
            print("Content-type:", response.headers['content-type'])

            html = await response.text()
            print("Body:", html)

class POST_POSITION_DATA:
    def __init__(self):
        global start_time
        self.start_time = start_time

    def post_data(self):
        end_time = time.time()
        current_time = end_time - self.start_time
        if current_time > PERF_MEASUREMENT_INTERVAL_SEC:
            asyncio.run(async_pos_req())
            self.start_time = time.time()
        else:
            pass


class GETFPS:
    def __init__(self, stream_id):
        global start_time
        self.start_time = start_time
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id
        self.total_fps_time = 0
        self.total_frame_count = 0

    def get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        current_time = end_time - self.start_time
        if current_time > PERF_MEASUREMENT_INTERVAL_SEC:
            self.total_fps_time = self.total_fps_time + current_time
            self.total_frame_count = self.total_frame_count + self.frame_count
            current_fps = float(self.frame_count) / current_time
            avg_fps = float(self.total_frame_count) / self.total_fps_time
            sys.stdout.write('DEBUG: FPS of stream %d: %.2f (%.2f)\n' % (self.stream_id + 1, current_fps, avg_fps))
            self.start_time = end_time
            self.frame_count = 0
        else:
            self.frame_count = self.frame_count + 1

posting_data = POSTDATA()

def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))

    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0
    obj_meta.text_params.font_params.font_name = 'Ubuntu'
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = x_offset
    obj_meta.text_params.y_offset = y_offset
    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0


def parse_pose_from_meta(frame_meta, obj_meta):
    num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

    gain = min(obj_meta.mask_params.width / STREAMMUX_WIDTH,
               obj_meta.mask_params.height / STREAMMUX_HEIGHT)
    pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
    pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    for i in range(num_joints):
        data = obj_meta.mask_params.get_mask_array()

        # Extract information for the i-th joint from the mask array
        xc = int((data[i * 3 + 0] - pad_x) / gain)  # X-coordinate
        yc = int((data[i * 3 + 1] - pad_y) / gain)  # Y-coordinate
        confidence = data[i * 3 + 2]  # Confidence score

        if confidence < 0.5:
            continue    # Skip if average confidence is low
            
       # Check if the current number of lines in display metadata is 16
        if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            # Acquire a new display metadata instance from the pool associated with batch metadata
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            # Add the newly acquired display metadata to the current frame's metadata
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        circle_params = display_meta.circle_params[display_meta.num_circles]
        circle_params.xc = abs(xc)  # TEMP FIX
        circle_params.yc = abs(yc)  # TEMP FIX
        circle_params.radius = 6
        circle_params.circle_color.red = 1.0
        circle_params.circle_color.green = 1.0
        circle_params.circle_color.blue = 1.0
        circle_params.circle_color.alpha = 1.0
        circle_params.has_bg_color = 1
        circle_params.bg_color.red = 0.0
        circle_params.bg_color.green = 0.0
        circle_params.bg_color.blue = 1.0
        circle_params.bg_color.alpha = 1.0
        display_meta.num_circles += 1

    for i in range(num_joints + 2):
        data = obj_meta.mask_params.get_mask_array()

        # Extract information for the i-th joint pair from the mask array and skeleton
        x1 = int((data[(skeleton[i][0] - 1) * 3 + 0] - pad_x) / gain)  # X-coordinate for joint 1
        y1 = int((data[(skeleton[i][0] - 1) * 3 + 1] - pad_y) / gain)  # Y-coordinate for joint 1
        confidence1 = data[(skeleton[i][0] - 1) * 3 + 2]  # Confidence for joint 1

        x2 = int((data[(skeleton[i][1] - 1) * 3 + 0] - pad_x) / gain)  # X-coordinate for joint 2
        y2 = int((data[(skeleton[i][1] - 1) * 3 + 1] - pad_y) / gain)  # Y-coordinate for joint 2
        confidence2 = data[(skeleton[i][1] - 1) * 3 + 2]  # Confidence for joint 2

        if confidence1 < 0.5 or confidence2 < 0.5:
            continue    # Skip if average confidence is low

        # Check if the current number of lines in display metadata is 16
        if display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
            # Acquire a new display metadata instance from the pool associated with batch metadata
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            # Add the newly acquired display metadata to the current frame's metadata
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        line_params = display_meta.line_params[display_meta.num_lines]
        line_params.x1 = abs(x1)  # TEMP FIX
        line_params.y1 = abs(y1)  # TEMP FIX
        line_params.x2 = abs(x2)  # TEMP FIX
        line_params.y2 = abs(y2)  # TEMP FIX
        line_params.line_width = 6
        line_params.line_color.red = 0.0
        line_params.line_color.green = 0.0
        line_params.line_color.blue = 1.0
        line_params.line_color.alpha = 1.0
        display_meta.num_lines += 1


def draw_bottom_center_circle(frame_meta, obj_meta):
    # Calculate the center of the bounding box
    center_x = int(obj_meta.rect_params.left + obj_meta.rect_params.width / 2)
    center_y = int(obj_meta.rect_params.top + obj_meta.rect_params.height / 2)

    # Acquire a display metadata instance from the batch metadata
    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)

    # Add a single center point (circle) to the display metadata
    circle_params = display_meta.circle_params[0]
    circle_params.xc = center_x
    circle_params.yc = center_y
    circle_params.radius = 6
    circle_params.circle_color.red = 1.0
    circle_params.circle_color.green = 1.0
    circle_params.circle_color.blue = 1.0
    circle_params.circle_color.alpha = 1.0
    circle_params.has_bg_color = 1
    circle_params.bg_color.red = 0.0
    circle_params.bg_color.green = 0.0
    circle_params.bg_color.blue = 0.0
    circle_params.bg_color.alpha = 1.0
    display_meta.num_circles = 1

    # Add the display metadata to the frame
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    return center_x, center_y


# posting_pos_data = POST_POSITION_DATA()

def floormapping(current_index, mapping_points):
    global uri_homography_mapping
    uri = scr2uri[current_index]

    camera_pred_position = np.array([[mapping_points]], dtype='float32')    # Person's Position
    
    floor_position = cv2.perspectiveTransform(camera_pred_position, uri_homography_mapping[uri]["homography_matrix"])

    uri_floor_position[uri] = floor_position[0][0].tolist()
    timestamp = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
    headers = "URI,Date,Time,X_Position,Y_Position\n"
    position_str = f"{uri},{timestamp},{uri_floor_position[uri][0]},{uri_floor_position[uri][1]}\n"

    print(f"{position_str}")
    try:
        with open("floor_positions.csv", 'a') as file:
            if file.tell() == 0:
                file.write(headers)
            file.write(position_str)
    except Exception as e:
        print(f"ERROR: {e}")

    return floor_position


def tracker_src_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        people_count = 0
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        current_index = frame_meta.source_id

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                people_count += 1
            except StopIteration:
                break

            parse_pose_from_meta(frame_meta, obj_meta)
            set_custom_bbox(obj_meta)
            mapping_points = draw_bottom_center_circle(frame_meta, obj_meta)

            floormapping(current_index, mapping_points)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # fps_streams['stream{0}'.format(current_index)].get_fps()
        # print(f'{current_index}: {people_count}')
        uri = scr2uri[current_index]
        uri_ppl_count[uri] = people_count

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    # print(uri_ppl_count)
    # posting_data.post_data()
    # posting_pos_data.post_data()

    return Gst.PadProbeReturn.OK

def streammux_src_pad_buffer_probe(pad, info, user_data):
    global uri_homography_mapping
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        current_index = frame_meta.source_id

        uri = scr2uri[current_index]

        # MANUAL WORK
        uri_homography_mapping = {
            # '10.185.151.2xx' : {
            #     }
            '10.185.151.213' : {
                "camera_calib_pts" : np.array([[179, 466], [963, 655], [1005, 438], [338, 199], [488, 239], [547, 199], [657, 75], [755, 11]]), 
                "floor_calib_pts" : np.array([[215, 553], [285, 531], [285, 502], [201, 485], [215, 485], [215, 475], [215, 401], [215, 332]])
                },
            # '10.185.151.2xx' : {
            #     }
            }

        for _, data in uri_homography_mapping.items():
            camera_pts = data['camera_calib_pts']
            floor_pts = data['floor_calib_pts']
            homography_matrix, _ = cv2.findHomography(camera_pts, floor_pts, cv2.RANSAC, 5.0)

            data['homography_matrix'] = homography_matrix

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.REMOVE

def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find('decodebin') != -1:
        Object.connect('child-added', decodebin_child_added, user_data)
    if name.find('nvv4l2decoder') != -1:
        Object.set_property('drop-frame-interval', 0)
        Object.set_property('num-extra-surfaces', 1)
        if is_aarch64():
            Object.set_property('enable-max-performance', 1)
        else:
            Object.set_property('cudadec-memtype', 0)
            Object.set_property('gpu-id', GPU_ID)


def cb_newpad(decodebin, pad, user_data):
    streammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)
    if name.find('video') != -1:
        if features.contains('memory:NVMM'):
            if pad.link(streammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write('ERROR: Failed to link source to streammux sink pad\n')
        else:
            sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin')


def create_uridecode_bin(stream_id, uri, streammux):
    uri_name = uri.split("@")[-1]
    print(uri_name)
    bin_name = f'source-bin-{stream_id}'
    bin = Gst.ElementFactory.make('uridecodebin', bin_name)
    if 'rtsp://' in uri:
        pyds.configure_source_for_ntp_sync(hash(bin))
    bin.set_property('uri', uri)
    pad_name = f'sink_{stream_id}'
    streammux_sink_pad = streammux.get_request_pad(pad_name)
    bin.connect('pad-added', cb_newpad, streammux_sink_pad)
    bin.connect('child-added', decodebin_child_added, 0)
    # fps_streams['stream{0}'.format(stream_id)] = GETFPS(stream_id)
    scr2uri[stream_id] = uri_name
    return bin


def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write('DEBUG: EOS\n')
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('WARNING: %s: %s\n' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('ERROR: %s: %s\n' % (err, debug))
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == 'aarch64'


def main():
    Gst.init(None)

    loop = GLib.MainLoop()

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write('ERROR: Failed to create pipeline\n')
        sys.exit(1)

    streammux = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
    if not streammux:
        sys.stderr.write('ERROR: Failed to create nvstreammux\n')
        sys.exit(1)
    pipeline.add(streammux)

    for i, uri in enumerate(SOURCE):
        source_bin = create_uridecode_bin(i, uri, streammux)
        if not source_bin:
            sys.stderr.write(f'ERROR: Failed to create source_bin {i}\n')
            sys.exit(1)
        pipeline.add(source_bin)

    tiler = Gst.ElementFactory.make('nvmultistreamtiler', 'nvtiler')
    if not tiler:
        sys.stderr.write('ERROR: Failed to create nvmultistreamtiler\n')
        sys.exit(1)

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    if not pgie:
        sys.stderr.write('ERROR: Failed to create nvinfer\n')
        sys.exit(1)

    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    if not tracker:
        sys.stderr.write('ERROR: Failed to create nvtracker\n')
        sys.exit(1)

    converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvideoconvert')
    if not converter:
        sys.stderr.write('ERROR: Failed to create nvvideoconvert\n')
        sys.exit(1)

    osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
    if not osd:
        sys.stderr.write('ERROR: Failed to create nvdsosd\n')
        sys.exit(1)

    sink = None
    if is_aarch64():
        sink = Gst.ElementFactory.make('nv3dsink', 'nv3dsink')
        if not sink:
            sys.stderr.write('ERROR: Failed to create nv3dsink\n')
            sys.exit(1)
    else:
        sink = Gst.ElementFactory.make('nveglglessink', 'nveglglessink')
        if not sink:
            sys.stderr.write('ERROR: Failed to create nveglglessink\n')
            sys.exit(1)

    sys.stdout.write('\n')
    sys.stdout.write('SOURCE: %s\n' % SOURCE)
    sys.stdout.write('CONFIG_INFER: %s\n' % CONFIG_INFER)
    sys.stdout.write('STREAMMUX_BATCH_SIZE: %d\n' % STREAMMUX_BATCH_SIZE)
    sys.stdout.write('STREAMMUX_WIDTH: %d\n' % STREAMMUX_WIDTH)
    sys.stdout.write('STREAMMUX_HEIGHT: %d\n' % STREAMMUX_HEIGHT)
    sys.stdout.write('GPU_ID: %d\n' % GPU_ID)
    sys.stdout.write('PERF_MEASUREMENT_INTERVAL_SEC: %d\n' % PERF_MEASUREMENT_INTERVAL_SEC)
    sys.stdout.write('JETSON: %s\n' % ('TRUE' if is_aarch64() else 'FALSE'))
    sys.stdout.write('\n')

    streammux.set_property('batch-size', STREAMMUX_BATCH_SIZE)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('width', STREAMMUX_WIDTH)
    streammux.set_property('height', STREAMMUX_HEIGHT)
    streammux.set_property('enable-padding', 0)
    streammux.set_property('live-source', 1)
    streammux.set_property('sync-inputs', 1)
    streammux.set_property('attach-sys-ts', 1)
    tiler_rows = int(math.sqrt(len(SOURCE)))
    tiler_columns = int(math.ceil(1.0* len(SOURCE)/tiler_rows))
    tiler.set_property('rows', tiler_rows)
    tiler.set_property('columns', tiler_columns)
    tiler.set_property('width', STREAMMUX_WIDTH)
    tiler.set_property('height', STREAMMUX_HEIGHT)
    pgie.set_property('config-file-path', CONFIG_INFER)
    pgie.set_property('qos', 0)
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file',
                         '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
    tracker.set_property('display-tracking-id', 1)
    tracker.set_property('qos', 0)
    osd.set_property('process-mode', int(pyds.MODE_GPU))
    osd.set_property('qos', 0)
    sink.set_property('async', 0)
    sink.set_property('sync', 0)
    sink.set_property('qos', 0)

    if tracker.find_property('enable_batch_process') is not None:
        tracker.set_property('enable_batch_process', 1)

    if tracker.find_property('enable_past_frame') is not None:
        tracker.set_property('enable_past_frame', 1)
                
    if not is_aarch64():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        # streammux.set_property('nvbuf-memory-type', mem_type)
        streammux.set_property('gpu_id', GPU_ID)
        tiler.set_property('nvbuf-memory-type', mem_type)
        tiler.set_property('gpu_id', GPU_ID)
        pgie.set_property('gpu_id', GPU_ID)
        tracker.set_property('gpu_id', GPU_ID)
        converter.set_property('nvbuf-memory-type', mem_type)
        converter.set_property('gpu_id', GPU_ID)
        osd.set_property('gpu_id', GPU_ID)

    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(converter)
    pipeline.add(tiler)
    pipeline.add(osd)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(converter)
    converter.link(tiler)
    tiler.link(osd)
    osd.link(sink)
    
    dot_data = Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    with open("pipeline.dot", "w") as dot_file:
        dot_file.write(dot_data)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    streammux_src_pad = streammux.get_static_pad('src')
    if not streammux_src_pad:
        sys.stderr.write('ERROR: Failed to get streammux src pad\n')
        sys.exit(1)
    else:
        streammux_src_pad.add_probe(Gst.PadProbeType.BUFFER, streammux_src_pad_buffer_probe, 0)

    tracker_src_pad = tracker.get_static_pad('src')
    if not tracker_src_pad:
        sys.stderr.write('ERROR: Failed to get tracker src pad\n')
        sys.exit(1)
    else:
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)


    try:
        print("Starting pipeline \n")
        pipeline.set_state(Gst.State.PLAYING)
        sys.stdout.write('\n')
        loop.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.set_state(Gst.State.NULL)


def parse_args():
    global SOURCE, CONFIG_INFER, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, \
        PERF_MEASUREMENT_INTERVAL_SEC

    parser = argparse.ArgumentParser(description='DeepStream')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('-s', '--source', nargs='+', help='Source stream/file')
    source_group.add_argument('-i', '--input-file', help='Input file to read and store in SOURCE')
    parser.add_argument('-c', '--config-infer', required=True, help='Config infer file')
    parser.add_argument('-b', '--streammux-batch-size', type=int, default=1, help='Streammux batch-size (default: 1)')
    parser.add_argument('-w', '--streammux-width', type=int, default=1280, help='Streammux width (default: 1280)')
    parser.add_argument('-e', '--streammux-height', type=int, default=720, help='Streammux height (default: 720)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('-f', '--fps-interval', type=int, default=5, help='FPS measurement interval (default: 5)')
    args = parser.parse_args()
    if args.source == '' and args.input_file is None:
        sys.stderr.write('ERROR: Source not found\n')
        sys.exit(1)
    if args.config_infer == '' or not os.path.isfile(args.config_infer):
        sys.stderr.write('ERROR: Config infer not found\n')
        sys.exit(1)

    if args.input_file is not None:
        with open(args.input_file, 'r') as file:
            SOURCE = file.read().splitlines()
    else:
        SOURCE = args.source
    STREAMMUX_BATCH_SIZE = len(SOURCE)
    CONFIG_INFER = args.config_infer
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id
    PERF_MEASUREMENT_INTERVAL_SEC = args.fps_interval


if __name__ == '__main__':
    parse_args()
    sys.exit(main())
