import numpy as np
from scipy.spatial.transform import Rotation as R

# scipy rotation matrix is using right handed why we use left-handed matrices, this is the reason for inverting euler angles
# https://butterflyofdream.wordpress.com/2016/07/05/converting-rotation-matrices-of-left-handed-coordinate-system/


class Transform:
    """
    Class used to represent transformation within 3D space, rotations , translations and scale.
    Note that this class uses 'row major' format. See https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/row-major-vs-column-major-vector


    The z-axis points forward
    The y-axis points down
    The x-axis points to the right


             ^
           / z
         /          x
        |------------>
        |
        |
        |  y
        v


    Rotate around X” is Z-to-Y
    Rotate around Y” is X-to-Z
    Rotate around Z” is Y-to-X

    """

    def __init__(self, position, orientation_quat):
        self._position = np.array(position)
        self._orientation_obj = R.from_quat(orientation_quat)

    @classmethod
    def fromEuler(cls, position, euler_yxz):
        orientation_quat = R.from_euler('yxz', [euler_yxz[0], euler_yxz[1], euler_yxz[2]], degrees=True).as_quat()
        return cls(position, orientation_quat)

    @classmethod
    def fromMatrix(cls, matrix):
        orientation_quat = R.from_matrix(matrix[:3, :3]).as_quat()
        position = matrix[3, :3]
        return cls(position, orientation_quat)

    @property
    def position(self):
        return self._position

    def set_position(self, position):
        self._position = position

    def set_rotation_quat(self, orientation_quat):
        self._orientation_obj = R.from_quat(orientation_quat)

    @property
    def rotation_quat(self):
        return self._orientation_obj.as_quat()

    @property
    def rotation_matrix(self):
        return self._orientation_obj.as_matrix()

    @property
    def as_matrix(self):
        """Returns transform as 4x4 matrix"""
        matrix = np.identity(4)
        matrix[:3, :3] = self._orientation_obj.as_matrix()
        matrix[3, :3] = self._position
        return matrix

    def transform_points(self, points_list):
        """ Returns the transform points"""
        assert type(points_list) == np.ndarray and points_list.shape[1] == 3, "invalid argument"

        # expand list with one axis
        points = np.ones((points_list.shape[0], 4))
        points[:, :3] = points_list
        # apply transformation
        result = points.dot(self.as_matrix)
        return result[:, :3]

    def apply_transform(self, transform):
        """apply another transformation on the current transform"""
        new_matrix = np.dot(self.as_matrix, transform.as_matrix)
        return Transform.fromMatrix(new_matrix)

    def set_rotation(self, yaw, pitch, roll, degrees=True):
        """set transform orientation using euler angles"""
        self._orientation_obj = R.from_euler('yxz', [yaw, pitch, roll], degrees=True)

    def rotation_euler(self, degrees=True):
        """
        returns yxz euler angles
        """
        euler = self._orientation_obj.as_euler('yxz', degrees=degrees)
        return np.array([euler[0], euler[1], euler[2]])

    def to_inverse(self):
        """
        returns transform corresponding to the inverse of the current transform
        """
        matrix = self.as_matrix
        inv_rotation = np.eye(4)
        inv_rotation[:3, :3] = matrix[:3, :3].T
        inv_translation = np.eye(4)
        inv_translation[3, :3] = -matrix[3, :3]
        return Transform.fromMatrix(inv_translation.dot(inv_rotation))

    def between(self, other):
        """ 
        self and other are in the same reference frame, between returns self relative to other
        """
        # tij, tik -> tjk
        return self.apply_transform(other.to_inverse())
        
    # For call to repr(). Prints object's information
    def __repr__(self):
        return f'Transform (position: {self.position}, orientation: {self.rotation_euler()}'
    # For call to str(). Prints readable form
    def __str__(self):
        return f'Transform (position: {self.position}, orientation: {self.rotation_euler()}'