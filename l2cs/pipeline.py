import pathlib
from typing import Union, List

import cv2
import numpy as np
import torch, time
import torch.nn as nn
from dataclasses import dataclass
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer


class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.1
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested
        if self.include_detector:

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=0)

            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = [idx for idx in range(90)]
            self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:
        # 컨테이너 생성
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None:
                # 가장 큰 얼굴 찾기
                max_area = 0
                max_face_idx = 0
                
                for i, (box, _, _) in enumerate(faces):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_area:
                        max_area = area
                        max_face_idx = i
                
                # 가장 큰 얼굴만 처리
                box, landmark, score = faces[max_face_idx]
                
                # 임계값 적용
                if score >= self.confidence_threshold:
                    # x,y 좌표의 안전한 최소/최대값 추출
                    x_min = max(int(box[0]), 0)
                    y_min = max(int(box[1]), 0) 
                    x_max = int(box[2])
                    y_max = int(box[3])
                    
                    # 이미지 자르기
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # 데이터 저장
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

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
            pitch, yaw = self.predict_gaze(frame)

        # 데이터 저장
        if bboxes:
            results = GazeResultContainer(
                pitch=pitch,
                yaw=yaw, 
                bboxes=np.stack(bboxes),
                landmarks=np.stack(landmarks),
                scores=np.stack(scores)
            )
        else:
            results = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.array([]),
                landmarks=np.array([]),
                scores=np.array([])
            )

        return results

    def step_batch(self, frames: torch.Tensor) -> List[GazeResultContainer]:
        batch_size = frames.shape[0]
        results = []
        
        if self.include_detector:
            # 얼굴 검출을 배치로 처리
            faces_batch = self.detector(frames)
            
            for i in range(batch_size):
                faces = faces_batch[i]
                face_imgs = []
                bboxes = []
                landmarks = []
                scores = []
                
                if faces is not None:
                    # 가장 큰 얼굴 찾기
                    max_area = 0
                    max_face_idx = 0
                    
                    for j, (box, _, _) in enumerate(faces):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        if area > max_area:
                            max_area = area
                            max_face_idx = j
                    
                    # 가장 큰 얼굴만 처리
                    box, landmark, score = faces[max_face_idx]
                    
                    # 임계값 적용
                    if score >= self.confidence_threshold:
                        # x,y 좌표의 안전한 최소/최대값 추출
                        x_min = max(int(box[0]), 0)
                        y_min = max(int(box[1]), 0)
                        x_max = int(box[2])
                        y_max = int(box[3])
                        
                        # 이미지 자르기
                        img = frames[i][y_min:y_max, x_min:x_max]
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        face_imgs.append(img)
                        
                        # 데이터 저장
                        bboxes.append(box)
                        landmarks.append(landmark)
                        scores.append(score)
                
                # 시선 예측
                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                    results.append(GazeResultContainer(
                        pitch=pitch,
                        yaw=yaw,
                        bboxes=np.stack(bboxes),
                        landmarks=np.stack(landmarks),
                        scores=np.stack(scores)
                    ))
                else:
                    results.append(GazeResultContainer(
                        pitch=np.empty((0,1)),
                        yaw=np.empty((0,1)),
                        bboxes=np.array([]),
                        landmarks=np.array([]),
                        scores=np.array([])
                    ))
        else:
            # 시선 예측을 배치로 처리
            pitch_batch, yaw_batch = self.predict_gaze(frames)
            for i in range(batch_size):
                results.append(GazeResultContainer(
                    pitch=pitch_batch[i:i+1],
                    yaw=yaw_batch[i:i+1],
                    bboxes=np.array([]),
                    landmarks=np.array([]),
                    scores=np.array([])
                ))
        
        return results

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Predict 
        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        return pitch_predicted, yaw_predicted
