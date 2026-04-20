import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class SingleHandDetector:
    def __init__(self, hand_type="Right", min_detection_confidence=0.8, min_tracking_confidence=0.8, selfie=False):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]
        self.joint_names = [
            "WristRoot",
            "Thumb1",
            "Thumb2",
            "Thumb3",
            "ThumbTip",
            "Index1",
            "Index2",
            "Index3",
            "IndexTip",
            "Middle1",
            "Middle2",
            "Middle3",
            "MiddleTip",
            "Ring1",
            "Ring2",
            "Ring3",
            "RingTip",
            "Pinky1",
            "Pinky2",
            "Pinky3",
            "PinkyTip"
        ]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1)

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS, landmark_style, connection_style
            )

        return image

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            # print("not results.multi_hand_landmarks")
            return 0, None, None, None , None

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        # print(f'desired_hand_num: {desired_hand_num}')
        if desired_hand_num < 0:
            return 0, None, None, None , None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # Parse 3d keypoint from MediaPipe hand detector
        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano
        initpoints = joint_pos[[0, 5, 9], :]
        w_points = keypoint_3d_array[[0, 5, 9], :]
        wrist_R = self.compute_rotation_matrix(initpoints, w_points)
        
        # joint_pos = joint_pos - np.array([0,0,0.05])
        # print(f'wrist position: {keypoint_3d_array[0]}')
        # print(f'wrist rotation: {mediapipe_wrist_rot}')
        # print(f'keypoint_3d : {keypoint_3d_array[1]}')
        # print(f'joint_pos : {joint_pos[1]}')
        # print(f'keypoint_2d :{keypoint_2d[1]}')
        print(f'keypoinr_3d:{keypoint_3d_array.shape}')
        return num_box, joint_pos, keypoint_2d, wrist_R  ,keypoint_3d_array

    def detect_from_quest3(self, oculus_reader):
        # Parse 3d keypoint from MediaPipe hand detector
        joint_pos = oculus_reader.get_joint_transformations()[1]
        rot = []
        pos = []
        final_pos = []
        if joint_pos == {}:
            return None
        
        mediapipe_wrist_rot = joint_pos["WristRoot"][:3, :3]
        wrist_position = joint_pos["WristRoot"][:3, 3]
        
        for joint_name in self.joint_names:
            joint_transformation = joint_pos[joint_name]
            # Take the 4th column of the transformation matrix as the position
            pos.append(joint_transformation[:3, 3] - wrist_position)
            # Take the rotation part of the transformation matrix
            rot.append(joint_transformation[:3, :3])
            
            # final_pos.append(pos[-1] @ np.linalg.inv(rot[-1]) @ self.operator2mano)
            final_pos.append(pos[-1] @ mediapipe_wrist_rot @ self.operator2mano)
        # Convert lists to numpy arrays
        pos = np.array(pos) 
        rot = np.array(rot)
        final_pos = np.array(final_pos)        
        
        return final_pos

    @staticmethod
    def parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def compute_rotation_matrix(self ,initpoints, points):
        # 计算质心
        c_init = np.mean(initpoints, axis=0)
        c_points = np.mean(points, axis=0)
        
        # 验证质心模长是否相等
        if not np.isclose(np.linalg.norm(c_init), np.linalg.norm(c_points)):
            raise ValueError("质心模长不匹配，无法求解旋转矩阵。")
        
        # 去中心化
        A = initpoints - c_init
        B = points - c_points
        
        # 计算协方差矩阵
        H = A.T @ B  # 协方差矩阵
        
        # 奇异值分解
        U, S, Vt = np.linalg.svd(H)
        
        # 构造旋转矩阵
        R = Vt.T @ U.T
        
        # 处理反射（确保行列式为1）
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1  # 调整V的最后一列符号
            R = Vt.T @ U.T

        # pre_points = R @ initpoints.T 
        # print(f'pre_points : {pre_points.T}')
        # print(f'points : {points}')
        # exit()
        
        return R