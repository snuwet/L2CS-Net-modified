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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        if self.include_detector:
            results = self.face_detection.process(frame_rgb)

            if results.detections:
                max_area = 0
                best_detection = None
                
                # Find the largest face based on bounding box area
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    bbox = int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height), \
                           int(bboxC.width * frame_width), int(bboxC.height * frame_height)
                    area = bbox[2] * bbox[3]
                    if area > max_area:
                        max_area = area
                        best_detection = detection
                        best_bbox_abs = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

                if best_detection:
                    score = best_detection.score[0]
                    
                    # Use the already calculated absolute bbox for the largest face
                    box = best_bbox_abs
                    
                    # x,y 좌표의 안전한 최소/최대값 추출
                    x_min = max(box[0], 0)
                    y_min = max(box[1], 0) 
                    x_max = min(box[2], frame_width)
                    y_max = min(box[3], frame_height)
                    
                    # 이미지 자르기 (원래 프레임에서)
                    img = frame[y_min:y_max, x_min:x_max]
                    # Check if the cropped image is valid
                    if img.size > 0:
                        if self.convert_face_to_rgb:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        img = cv2.resize(img, (224, 224))
                        face_imgs.append(img)
                        
                        # 데이터 저장 (absolute bbox)
                        bboxes.append(np.array(box)) 
                        # landmarks are available in detection.location_data.relative_keypoints but not added here
                        scores.append(score)
                    else:
                        print("Warning: Cropped image size is zero.")


                # 시선 예측
                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                else:
                    pitch = np.empty((0,1))
                    yaw = np.empty((0,1))

            else:
                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            # If no detector, assume the whole frame is the face
            img_resized = cv2.resize(frame_rgb, (224, 224))
            pitch, yaw = self.predict_gaze(np.expand_dims(img_resized, axis=0)) # Add batch dimension

        # 데이터 저장
        if bboxes:
            results_container = GazeResultContainer(
                pitch=pitch,
                yaw=yaw, 
                bboxes=np.stack(bboxes),
                landmarks=np.array([]),
                scores=np.stack(scores)
            )
        else:
            # If detector included but no faces found, or detector not included
            results_container = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.array([]),
                landmarks=np.array([]),
                scores=np.array([])
            )
            
        return results_container

    def step_batch(self, frames: Union[np.ndarray, torch.Tensor]) -> List[GazeResultContainer]:
        # Ensure input is numpy array
        if isinstance(frames, torch.Tensor):
             # Assuming tensor is (B, H, W, C) and in RGB if coming from elsewhere
             # Or (B, C, H, W). Let's assume (B, H, W, C) BGR like typical cv2 batch
             # Convert to numpy if it's on GPU etc.
             frames = frames.cpu().numpy()
             # If channel first, permute. Example check: if frames.shape[1] == 3: frames = frames.transpose(0, 2, 3, 1)
             # Need to know the expected format of frames tensor

        # Convert frames to list of numpy arrays if it's a single stacked array
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
             frames_list = [frame for frame in frames]
        elif isinstance(frames, list):
             frames_list = frames
        else:
             raise ValueError("Input 'frames' must be a 4D numpy array or a list of 3D numpy arrays.")

        batch_results = []

        for frame in frames_list:
             # Process each frame individually using the step method
             # Ensure frame is a valid image (3D numpy array)
             if isinstance(frame, np.ndarray) and frame.ndim == 3:
                 result = self.step(frame)
                 batch_results.append(result)
             else:
                  # Handle potential invalid frames in the batch if necessary
                  print("Warning: Skipping invalid frame in batch.")
                  # Append an empty result or handle as appropriate
                  batch_results.append(GazeResultContainer(
                     pitch=np.empty((0,1)), yaw=np.empty((0,1)),
                     bboxes=np.array([]), landmarks=np.array([]), scores=np.array([])
                  ))
        
        return batch_results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        processed_frame = None # Variable to hold the frame ready for prep_input_numpy or model
        
        if isinstance(frame, np.ndarray):
            # Ensure input has batch dimension if single image numpy array
            input_frame = frame
            if input_frame.ndim == 3:
                 input_frame = np.expand_dims(input_frame, axis=0)

            # Check the attribute and convert if necessary
            # Assume prep_input_numpy expects BGR
            if self.convert_face_to_rgb: 
                # If attribute is True, input_frame is RGB, convert to BGR
                frame_bgr = cv2.cvtColor(input_frame[0], cv2.COLOR_RGB2BGR) if input_frame.shape[0] == 1 else np.array([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in input_frame])
                processed_frame = frame_bgr
            else:
                # If attribute is False, input_frame is already BGR
                processed_frame = input_frame
            
            img = prep_input_numpy(processed_frame, self.device) # Pass the correctly formatted frame (BGR)

        elif isinstance(frame, torch.Tensor):
            # Assuming tensor inputs need to be handled appropriately upstream
            # If tensors are RGB, they might need conversion depending on the model's training
            img = frame.to(self.device) 
        else:
            raise RuntimeError("Invalid dtype for input")
    
        with torch.no_grad():
            gaze_pitch, gaze_yaw = self.model(img)
            pitch_predicted = self.softmax(gaze_pitch)
            yaw_predicted = self.softmax(gaze_yaw)
            
            pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
            
            pitch_predicted= pitch_predicted.cpu().numpy()* np.pi/180.0
            yaw_predicted= yaw_predicted.cpu().numpy()* np.pi/180.0

        # Return as (N, 1) arrays
        return pitch_predicted.reshape(-1, 1), yaw_predicted.reshape(-1, 1)

    def __del__(self):
        # Release mediapipe resources
        if hasattr(self, 'face_detection'):
             self.face_detection.close()
