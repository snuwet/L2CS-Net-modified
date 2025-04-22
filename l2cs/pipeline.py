import pathlib
from typing import Union, List

import cv2
import numpy as np
import torch, time
import torch.nn as nn
from dataclasses import dataclass
import mediapipe as mp

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer


class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.5,
        convert_face_to_rgb: bool = False
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.convert_face_to_rgb = convert_face_to_rgb

        # Create L2CS model
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        # Create Mediapipe Face Detection if requested
        if self.include_detector:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.confidence_threshold)
            
            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = [idx for idx in range(90)]
            self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        frame_height, frame_width, _ = frame.shape
        input_frame_bgr = frame

        if self.include_detector:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                max_area = 0
                best_detection = None
                best_bbox_abs = None

                for detection in results.detections:
                    try:
                        bboxC = detection.location_data.relative_bounding_box
                        if not all(hasattr(bboxC, attr) for attr in ['xmin', 'ymin', 'width', 'height']):
                             continue

                        x_min_rel, y_min_rel = bboxC.xmin, bboxC.ymin
                        width_rel, height_rel = bboxC.width, bboxC.height

                        if not (0 <= x_min_rel <= 1 and 0 <= y_min_rel <= 1 and width_rel >= 0 and height_rel >= 0):
                             continue

                        bbox = int(x_min_rel * frame_width), int(y_min_rel * frame_height), \
                               int(width_rel * frame_width), int(height_rel * frame_height)
                        area = bbox[2] * bbox[3]
                        if area > max_area:
                            max_area = area
                            best_detection = detection
                            best_bbox_abs = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                    except Exception:
                         continue

                if best_detection and best_bbox_abs:
                    score = best_detection.score[0] if best_detection.score else 0.0
                    box = best_bbox_abs
                    x_min = max(box[0], 0)
                    y_min = max(box[1], 0)
                    x_max = min(box[2], frame_width)
                    y_max = min(box[3], frame_height)

                    if x_max > x_min and y_max > y_min:
                        img = input_frame_bgr[y_min:y_max, x_min:x_max]
                        
                        if img.size > 0:
                            img_resized = cv2.resize(img, (224, 224))
                            face_imgs.append(img_resized)
                            bboxes.append(np.array(box))
                            scores.append(score)
                        else:
                            print("Warning: Cropped image size is zero.")

                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                else:
                    pitch = np.empty((0,1))
                    yaw = np.empty((0,1))
            else:
                pitch = np.empty((0,1))
                yaw = np.empty((0,1))
        else:
            frame_to_predict = input_frame_bgr
            img_resized = cv2.resize(frame_to_predict, (224, 224))
            pitch, yaw = self.predict_gaze(np.expand_dims(img_resized, axis=0))

        if bboxes:
            results_container = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.stack(bboxes),
                landmarks=np.array([]),
                scores=np.stack(scores)
            )
        else:
            results_container = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.array([]),
                landmarks=np.array([]),
                scores=np.array([])
            )
            
        return results_container

    def step_batch(self, frames: Union[np.ndarray, torch.Tensor]) -> List[GazeResultContainer]:
        if isinstance(frames, torch.Tensor):
             frames = frames.cpu().numpy()

        if isinstance(frames, np.ndarray) and frames.ndim == 4:
             frames_list = [frame for frame in frames]
        elif isinstance(frames, list):
             frames_list = frames
        else:
             raise ValueError("Input 'frames' must be a 4D numpy array or a list of 3D numpy arrays.")

        num_frames = len(frames_list)
        batch_results = []

        if self.include_detector:
             all_face_imgs_bgr = []
             batch_bboxes = [[] for _ in range(num_frames)]
             batch_scores = [[] for _ in range(num_frames)]
             face_to_frame_map = []

             for frame_idx, frame in enumerate(frames_list):
                 if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                     print(f"Warning: Skipping invalid frame at index {frame_idx} in batch.")
                     continue

                 frame_height, frame_width, _ = frame.shape
                 input_frame_bgr = frame
                 
                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 results = self.face_detection.process(frame_rgb)
                 
                 best_detection = None
                 best_bbox_abs = None
                 max_area = 0

                 if results.detections:
                     for detection in results.detections:
                         try:
                             bboxC = detection.location_data.relative_bounding_box
                             if not all(hasattr(bboxC, attr) for attr in ['xmin', 'ymin', 'width', 'height']):
                                 continue

                             x_min_rel, y_min_rel = bboxC.xmin, bboxC.ymin
                             width_rel, height_rel = bboxC.width, bboxC.height

                             if not (0 <= x_min_rel <= 1 and 0 <= y_min_rel <= 1 and width_rel >= 0 and height_rel >= 0):
                                 continue

                             bbox = int(x_min_rel * frame_width), int(y_min_rel * frame_height), \
                                    int(width_rel * frame_width), int(height_rel * frame_height)
                             area = bbox[2] * bbox[3]
                             if area > max_area:
                                 max_area = area
                                 best_detection = detection
                                 best_bbox_abs = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                         except Exception as e:
                             print(f"Error processing detection bbox in frame {frame_idx}: {e}")
                             continue

                 if best_detection and best_bbox_abs:
                     score = best_detection.score[0] if best_detection.score else 0.0
                     box = best_bbox_abs
                     x_min = max(box[0], 0)
                     y_min = max(box[1], 0)
                     x_max = min(box[2], frame_width)
                     y_max = min(box[3], frame_height)

                     if x_max > x_min and y_max > y_min:
                         img = input_frame_bgr[y_min:y_max, x_min:x_max]
                         
                         img_resized = cv2.resize(img, (224, 224))
                         all_face_imgs_bgr.append(img_resized)
                         face_to_frame_map.append(frame_idx)
                         batch_bboxes[frame_idx].append(np.array(box))
                         batch_scores[frame_idx].append(score)
                     else:
                         print(f"Warning: Cropped image size is zero after clipping for frame {frame_idx}.")

             batch_pitch = np.empty((0, 1))
             batch_yaw = np.empty((0, 1))
             if all_face_imgs_bgr:
                 face_batch_np = np.stack(all_face_imgs_bgr)
                 batch_pitch, batch_yaw = self.predict_gaze(face_batch_np)

             face_pred_idx = 0
             for frame_idx in range(num_frames):
                 num_faces_in_this_frame = len(batch_bboxes[frame_idx])
                 
                 if num_faces_in_this_frame > 0:
                     if face_pred_idx < len(batch_pitch):
                         frame_pitch = batch_pitch[face_pred_idx : face_pred_idx + num_faces_in_this_frame]
                         frame_yaw = batch_yaw[face_pred_idx : face_pred_idx + num_faces_in_this_frame]
                         
                         result = GazeResultContainer(
                             pitch=frame_pitch,
                             yaw=frame_yaw,
                             bboxes=np.stack(batch_bboxes[frame_idx]),
                             landmarks=np.array([]),
                             scores=np.stack(batch_scores[frame_idx])
                         )
                         face_pred_idx += num_faces_in_this_frame
                     else:
                         print(f"Warning: Prediction missing for detected face in frame {frame_idx}. Creating empty result.")
                         result = GazeResultContainer(pitch=np.empty((0, 1)), yaw=np.empty((0, 1)), bboxes=np.array([]), landmarks=np.array([]), scores=np.array([]))
                 else:
                     result = GazeResultContainer(pitch=np.empty((0, 1)), yaw=np.empty((0, 1)), bboxes=np.array([]), landmarks=np.array([]), scores=np.array([]))
                 
                 batch_results.append(result)

        else:
             processed_frames_bgr = []
             valid_frame_indices = []

             for idx, frame in enumerate(frames_list):
                 if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                     print(f"Warning: Skipping invalid frame at index {idx} in batch (no detector).")
                     continue

                 frame_to_process = frame
                 img_resized = cv2.resize(frame_to_process, (224, 224))
                 processed_frames_bgr.append(img_resized)
                 valid_frame_indices.append(idx)
             
             batch_results = [GazeResultContainer(pitch=np.empty((0, 1)), yaw=np.empty((0, 1)), bboxes=np.array([]), landmarks=np.array([]), scores=np.array([])) for _ in range(num_frames)]

             if processed_frames_bgr:
                 frame_batch_np = np.stack(processed_frames_bgr)
                 batch_pitch, batch_yaw = self.predict_gaze(frame_batch_np)

                 for i, original_idx in enumerate(valid_frame_indices):
                     if i < len(batch_pitch):
                         pitch = batch_pitch[i:i+1]
                         yaw = batch_yaw[i:i+1]
                         batch_results[original_idx] = GazeResultContainer(
                             pitch=pitch,
                             yaw=yaw,
                             bboxes=np.array([]),
                             landmarks=np.array([]),
                             scores=np.array([])
                         )
                     else:
                          print(f"Warning: Prediction missing for frame index {original_idx} (no detector).")

        return batch_results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        if isinstance(frame, np.ndarray):
            input_frame = frame
            if input_frame.ndim == 3:
                 input_frame = np.expand_dims(input_frame, axis=0)
            
            img = prep_input_numpy(input_frame, self.device)

        elif isinstance(frame, torch.Tensor):
            img = frame.to(self.device)
        else:
            raise RuntimeError("Invalid dtype for input to predict_gaze")
    
        with torch.no_grad():
            gaze_pitch, gaze_yaw = self.model(img)
            pitch_predicted = self.softmax(gaze_pitch)
            yaw_predicted = self.softmax(gaze_yaw)
            
            pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
            
            pitch_predicted= pitch_predicted.cpu().numpy()* np.pi/180.0
            yaw_predicted= yaw_predicted.cpu().numpy()* np.pi/180.0

        return pitch_predicted.reshape(-1, 1), yaw_predicted.reshape(-1, 1)

    def __del__(self):
        if hasattr(self, 'face_detection'):
             self.face_detection.close()
