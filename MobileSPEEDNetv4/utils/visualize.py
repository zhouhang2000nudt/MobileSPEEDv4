import cv2 as cv
import numpy as np

from scipy.spatial.transform import Rotation as R

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (0, 0, 0) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    x_min, y_min, x_max, y_max = bbox

    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv.LINE_AA,
    )
    return img

def project(q, r, K):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        # pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        rotation = R.from_quat([q[1], q[2], q[3], q[0]])
        pose_mat = np.hstack((rotation.as_matrix(), np.expand_dims(r, 1)))
        # pose_mat = np.hstack((quat2dcm(q), np.expand_dims(r, 1)))
        p_cam = pose_mat @ points_body

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = K @ points_camera_frame

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y


def visualize_axes(image, q, r, K):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        # no pose label for test
        xa, ya = project(q, r, K)
        image = cv.arrowedLine(image, (int(xa[0]), int(ya[0])), (int(xa[1]), int(ya[1])), (0, 0, 255), 2, tipLength=0.1)  # X轴
        image = cv.arrowedLine(image, (int(xa[0]), int(ya[0])), (int(xa[2]), int(ya[2])), (0, 255, 0), 2, tipLength=0.1)  # Y轴
        image = cv.arrowedLine(image, (int(xa[0]), int(ya[0])), (int(xa[3]), int(ya[3])), (255, 0, 0), 2, tipLength=0.1)  # Z轴

        return image

def visualize(image: np.ndarray, bboxes: np.ndarray, category_ids: np.ndarray, category_id_to_name: dict, ori: np.ndarray, pos: np.ndarray, K: np.ndarray):
    bboxes = bboxes.astype(np.int32)
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    img = visualize_axes(img, np.array(ori), np.array(pos), K)
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()