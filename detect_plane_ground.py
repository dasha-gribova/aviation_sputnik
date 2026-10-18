import multiprocessing as mp
import time
import cv2
import torch
import torchvision.ops as ops
import numpy as np
import queue
import os
import glob
from ultralytics import YOLO

# Заглушка на случай, если geometry_lib не подгрузится
def calc_scan_areas(scan_area_bbox, window_w_h=(800, 800), overlay=(0.2, 0.2), force_int=True):
    x1, y1, x2, y2 = scan_area_bbox
    w_win, h_win = window_w_h
    step_x = int(w_win * (1 - overlay[0]))
    step_y = int(h_win * (1 - overlay[1]))
    areas = []
    for y in range(y1, y2, step_y):
        for x in range(x1, x2, step_x):
            cx2 = min(x + w_win, x2)
            cy2 = min(y + h_win, y2)
            areas.append([int(x), int(y), int(cx2), int(cy2)])
    return areas


class YoloSahiInference:
    def __init__(self, q_in: mp.Queue, q_out: mp.Queue, q_names: mp.Queue,
                 conf=0.25, path_model="yolov8x-airplane.pt", nms_type="classic"):
        self.q_in = q_in
        self.q_out = q_out
        self.q_names = q_names
        self.conf = conf
        self.path_model = path_model
        self.nms_type = nms_type

        self.imgsize = 960
        self.half_flag = True
        self.device = 0 if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        try:
            print(f"Loading model from {self.path_model} on {self.device}...")
            return YOLO(self.path_model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def process_start(self, daemon=True):
        prc_main = mp.Process(target=self.main_func, args=(), daemon=daemon)
        prc_main.start()
        return prc_main

    def apply_global_nms(self, all_boxes_glob):
        if not all_boxes_glob:
            return []
        boxes = torch.tensor([[x[2], x[3], x[4], x[5]] for x in all_boxes_glob], dtype=torch.float32)
        scores = torch.tensor([x[1] for x in all_boxes_glob], dtype=torch.float32)
        keep_indices = ops.nms(boxes, scores, iou_threshold=0.5)
        return [all_boxes_glob[i] for i in keep_indices.tolist()]

    def main_func(self):
        model = self.load_model()
        if model is None:
            return

        self.q_names.put(model.names)
        print("Inference process started and ready.")

        while True:
            try:
                batch_data = self.q_in.get(timeout=2)
                if batch_data is None:
                    self.q_out.put(None)
                    break

                lst_batch_img, cropp_cord, orig_shape, frame_to_draw = batch_data
                h_orig, w_orig = orig_shape

                start_time = time.time()

                results = model(
                    lst_batch_img,
                    device=self.device,
                    conf=self.conf,
                    imgsz=self.imgsize,
                    half=self.half_flag,
                    verbose=False,
                    augment=False
                )

                all_data = []
                for i, r in enumerate(results):
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        cls = r.boxes.cls.cpu().numpy()

                        x_offset, y_offset = cropp_cord[i]

                        boxes_glob = boxes.copy()
                        boxes_glob[:, [0, 2]] += x_offset
                        boxes_glob[:, [1, 3]] += y_offset

                        boxes_glob[:, [0, 2]] = np.clip(boxes_glob[:, [0, 2]], 0, w_orig)
                        boxes_glob[:, [1, 3]] = np.clip(boxes_glob[:, [1, 3]], 0, h_orig)

                        data = np.column_stack([cls, confs, boxes_glob])
                        all_data.append(data)

                if all_data:
                    all_boxes_glob = np.vstack(all_data).tolist()
                    final_predictions = self.apply_global_nms(all_boxes_glob)
                else:
                    final_predictions = []

                self.q_out.put((final_predictions, frame_to_draw))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"YOLO batch processing failed: {e}", exc_info=True)


def package_frame(q_in: mp.Queue, frame: np.ndarray, crop_w, crop_h, overlay_w, overlay_h):
    h, w = frame.shape[:2]
    scan_areas = calc_scan_areas([0, 0, w, h], (crop_w, crop_h), (overlay_w, overlay_h), force_int=True)

    batch_imgs = []
    batch_offsets = []

    for x1, y1, x2, y2 in scan_areas:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        batch_imgs.append(crop)
        batch_offsets.append((x1, y1))

    batch_imgs.append(frame)
    batch_offsets.append((0, 0))

    q_in.put((batch_imgs, batch_offsets, (h, w), frame))


def producer_process(q_in: mp.Queue, source: str, source_type: str = 'image', fps_limit: float = 30.0):
    crop_w, crop_h = 960, 960
    overlay_w, overlay_h = 0.2, 0.2

    if source_type == 'image':
        if os.path.isdir(source):
            print(f"Producer: Scanning directory '{source}' for images...")
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']:
                image_paths.extend(glob.glob(os.path.join(source, ext)))

            if not image_paths:
                print(f"Producer: No images found in '{source}'!")
                q_in.put(None)
                return

            print(f"Producer: Found {len(image_paths)} images. Processing sequentially...")

            for img_path in sorted(image_paths):
                print(f"Producer: Loading {os.path.basename(img_path)}...")
                img = cv2.imread(img_path)
                if img is None:
                    continue
                package_frame(q_in, img, crop_w, crop_h, overlay_w, overlay_h)

            print("Producer: All images processed. Sending termination signal.")
            q_in.put(None)
        else:
            print(f"Producer: Loading single image '{source}'...")
            img = cv2.imread(source)
            if img is None:
                q_in.put(None)
                return
            package_frame(q_in, img, crop_w, crop_h, overlay_w, overlay_h)
            q_in.put(None)

    elif source_type in ['video', 'camera']:
        print(f"Producer: Opening stream '{source}'...")
        cap = cv2.VideoCapture(source)
        if source_type == 'camera':
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            q_in.put(None)
            return

        frame_interval = 1.0 / fps_limit
        last_frame_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if source_type == 'camera':
                    current_time = time.time()
                    if current_time - last_frame_time < frame_interval:
                        continue
                    last_frame_time = current_time

                while q_in.qsize() >= 2:
                    try: q_in.get_nowait()
                    except: break

                package_frame(q_in, frame, crop_w, crop_h, overlay_w, overlay_h)
        finally:
            cap.release()
            q_in.put(None)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    q_in = mp.Queue(maxsize=3)
    q_out = mp.Queue()
    q_names = mp.Queue()

    # === ВАШИ ПУТИ И КОНФИГ ===
    MODEL_PATH = "/Users/darya/PycharmProjects/aviation_sutnik/best(2).pt"
    SOURCE_TYPE = 'image'
    SOURCE_PATH = "/Users/darya/PycharmProjects/aviation_sutnik/aviation_sutnik"

    inference_worker = YoloSahiInference(
        q_in=q_in, q_out=q_out, q_names=q_names,
        conf=0.5, path_model=MODEL_PATH
    )
    p_inference = inference_worker.process_start(daemon=True)

    p_producer = mp.Process(target=producer_process, args=(q_in, SOURCE_PATH, SOURCE_TYPE, 25.0))
    p_producer.start()

    print("Main: Waiting for results...")
    frame_counter = 0
    while True:
        try:
            result_data = q_out.get(timeout=2)
            if result_data is None:
                break

            frame_counter += 1
            final_detections, frame_to_draw = result_data

            # === ОТРИСОВКА ДЕТЕКЦИЙ ===
            for det in final_detections:
                cls, conf, x1, y1, x2, y2 = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Airplane: {conf:.2f}"
                cv2.putText(frame_to_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # === СЧЕТЧИК В ЛЕВОМ ВЕРХНЕМ УГЛУ ===
            count_text = f"Detections: {len(final_detections)}"
            # Рисуем подложку для текста, чтобы было хорошо видно на любом фоне
            (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame_to_draw, (5, 5), (15 + text_w, 15 + text_h), (0, 0, 0), -1)
            cv2.putText(frame_to_draw, count_text, (10, 10 + text_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Умный даунскейл для отображения
            h_disp, w_disp = frame_to_draw.shape[:2]
            max_disp_width = 1280
            if w_disp > max_disp_width:
                scale = max_disp_width / w_disp
                frame_to_draw = cv2.resize(frame_to_draw, (max_disp_width, int(h_disp * scale)))

            cv2.imshow("SAHI YOLO Global Detections", frame_to_draw)

            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

        except queue.Empty:
            if not p_producer.is_alive() and q_out.empty():
                break
            continue

    p_producer.join()
    p_inference.terminate()
    cv2.destroyAllWindows()
    print(f"Pipeline finished successfully. Total frames processed: {frame_counter}")