
import numpy as np
import gtsam
from gtsam import Point3, Pose3, Rot3
from scipy.spatial.transform import Rotation as R
from transform import Transform


class UncertainTransform(Transform):
    # https://arxiv.org/pdf/1906.07795.pdf?fbclid=IwAR0Ue3IAqTlzd3MuLMG-O51NqQ3Lh01WlivsaVSyBvIitDNX6k-gLPnf9VE
    # https://www.cs.ubc.ca/~schmidtm/Courses/540-W19/L14.pdf
    # https://geostatisticslessons.com/lessons/errorellipses
    # https://www2.imm.dtu.dk/pubdb/edoc/imm3312.pdf
    # https://janakiev.com/blog/covariance-matrix/
    # https://bitbucket.org/jmangelson/lie/src/master/src/lie/lie.hpp


    def __init__(self,
                 position,
                 orientation_quat,
                 covariance_matrix):
        """
        :param position: 3D vector describing position, translation from the origin np.array
        :param orientation: 4d np.array describing orientation
        :covariance_matrix: covariance matrix 6x6 (yaw,pitch,roll,x,y,z), in radians
        """
        super().__init__(position, orientation_quat)
        self.covariance_matrix = covariance_matrix

    @classmethod
    def fromStd(cls, position, orientation_quat, position_std, orientation_std_deg):
        """
        :param position: 3D vector describing position, translation from the origin np.array
        :param orientation: 4d np.array describing orientation
        :position_std: a 3D vector containing the standard deviation on each axis of a Normal Distribution centered at position of measurement
        :orientation_std_deg: a 3D vector containing the standard deviation of the euler angles [xyz] using a ND centered at orientation of measurement
        """
        #initialise the cov matrix with uncorrelated axis values
        covariance_matrix = UncertainTransform.___error_to_covariance(position_std, orientation_std_deg)
        return cls(position, orientation_quat, covariance_matrix)


    @classmethod
    def fromMatrix(cls, matrix, position_std, orientation_std_deg,):
        """
        :param matrix: 4x4 matrix containing rotation and translation
        :position_std: a 3D vector containing the standard deviation on each axis of a Normal Distribution centered at position of measurement
        :orientation_std_deg: a 3D vector containing the standard deviation of the euler angles [xyz] using a ND centered at orientation of measurement
        """

        orientation_quat = R.from_matrix(matrix[:3, :3]).as_quat()
        position = matrix[3, :3]
        return cls.fromStd(position, orientation_quat, position_std, orientation_std_deg)
        
    @classmethod
    def fromEuler(cls,
                  position,
                  euler_yxz,
                  position_std,
                  orientation_std_deg):
        """
        :param position: 3D vector describing position, translation from the origin np.array
        :param euler_yxz: yaw,pitch,roll angles in degrees
        :position_std: a 3D vector containing the standard deviation on each axis of a Normal Distribution centered at position of measurement
        :orientation_std_deg: a 3D vector containing the standard deviation of the euler angles [xyz] using a ND centered at orientation of measurement
        """
        orientation_quat = R.from_euler('yxz', [euler_yxz[0], euler_yxz[1], euler_yxz[2]], degrees=True).as_quat()
        return cls.fromStd(position, orientation_quat, position_std, orientation_std_deg)



    def apply_transform(self, other_transform):
        """apply another transformation on the current transform"""
        # compose poses
        new_transform = super().apply_transform(other_transform)

        # composing pose uncertainty
        # https://bitbucket.org/jmangelson/lie/src/master/src/lie/lie.hpp

        if hasattr(other_transform, "covariance_matrix"):
            # get parent covariance 
            E_ij= UncertainTransform.__cov_axis_convert(other_transform.covariance_matrix)
 
            # propagate parrent covariance in child 
            T_jk = UncertainTransform.TransformToGTSAM(self.to_inverse())
            Adj_jk = T_jk.AdjointMap() 
            E_jk= UncertainTransform.__cov_axis_convert(self.covariance_matrix)
            Eik = Adj_jk.dot(E_ij).dot(Adj_jk.T)
            composed_cov =   Eik + E_jk
            covariance_matrix = UncertainTransform.__cov_axis_convert(composed_cov)
        else:
            covariance_matrix=self.covariance_matrix

        return UncertainTransform(new_transform.position, new_transform.rotation_quat, covariance_matrix=covariance_matrix)

    def to_inverse(self):
        """
        returns uncertain transform corresponding to the inverse of the current transform
        """
        # pose inverse
        transform_inv = super().to_inverse()

        # covariance to inverse
        covariance_inv = UncertainTransform.apply_transform_covariance(self.covariance_matrix, transform_inv)
        return UncertainTransform(transform_inv.position, transform_inv.rotation_quat, covariance_matrix=covariance_inv)

        
    def compute_ellipse_from_covariance(self, alph=0.99):
        """
        Computes dominant axis and scale from covariance matrix using only x, z
        :param alph:  {float btw 0 and 1} -- percentage of data inside the ellipse
        returns angle of rotation and size
        """
        # sample positions
        positions = np.array([p.position[[0,2]] for p in self.get_samples(nb_samples=100)])
        position_cov = np.cov(positions[:,1],positions[:,0])

        c = -2 * np.log(1 - alph)  # quantile at alpha of the chi_squarred distr. with df = 2
        Lambda, Q = np.linalg.eig(position_cov)  # eigenvalues and eigenvectors (col. by col.)
        
        ## Compute the attributes of the ellipse
        w, h = 2 * np.sqrt(c * Lambda)
        # compute the value of the angle theta (in degree)
        angle = 180 * np.arctan2(Q[1,0] , Q[0,0]) / np.pi if position_cov[1,0] else 0
        return -angle, h, w


    @staticmethod
    def gausian_product(u_list, c_list):
        """
        If two random variables are independent, it must be that p1(x1)p2(x2) = p(x1,x2)p1(x1)p2(x2)=p(x1,x2),
        therefore, in that case, the product of two Gaussian pdfs will be exactly a Gaussian pdf (the joint pdf)
        The product of two Gaussian pdfs has not to be a Gaussian pdf if the variables are not independent!
        
        :param u_list: means of the Normal Distributions
        :param c_list: variance of the Normal Distributions
        """
        ic_list = []
        ic_sum = 0     
        for idx,c1 in enumerate(c_list):
            ic1 = np.linalg.inv(c1)
            ic_list.append(ic1)
            ic_sum+=ic1

        t = 0
        c = np.linalg.inv(ic_sum)
        for u1, ic1 in zip(u_list, ic_list):
            t+=u1.dot(ic1)
        u = t.dot(c)
        return u, c

    @staticmethod
    def ___error_to_covariance(position_axis_std, orientation_axis_std_deg):
        orientation_std = np.radians(orientation_axis_std_deg)
        orientation_std[orientation_std==0]=1e-7
        position_axis_std=np.array(position_axis_std).astype(float)
        position_axis_std[position_axis_std==0]=1e-7
        covariance_matrix = np.zeros((6,6))
        for i,v in enumerate([
                    orientation_std[0],orientation_std[1],orientation_std[2],
                    position_axis_std[0],position_axis_std[1],position_axis_std[2],
                    ]):
            covariance_matrix[i,i] = v
            
        return covariance_matrix


    def between(self, other):
        """ 
        self and other are in the same reference frame, between returns self relative to other
        """

        self2other = super().between(other)
        covariance_matrix = self.__between_pose_covariance(self,other)
        return UncertainTransform(self2other.position, self2other.rotation_quat, covariance_matrix=covariance_matrix)


    def __between_pose_covariance(self, A, B):

        # get other covariance in origin frame
        T_wB = UncertainTransform.TransformToGTSAM(B)
        Adj_wB_inv = T_wB.inverse().AdjointMap() 
        E_B= UncertainTransform.__cov_axis_convert(B.covariance_matrix) if hasattr(B, "covariance_matrix") else np.zeros((6,6))#*1e-7
        E_wB =  Adj_wB_inv.dot(E_B).dot(Adj_wB_inv.T) 

        # move other covariance in self body frame and sum
        T_wA = UncertainTransform.TransformToGTSAM(A)
        Adj_wA = T_wA.AdjointMap() 
        E_A= UncertainTransform.__cov_axis_convert(A.covariance_matrix) if hasattr(A, "covariance_matrix") else np.zeros((6,6))
        E_AB =  Adj_wA.dot(E_wB).dot(Adj_wA.T) + E_A  
        
        E_AB =  UncertainTransform.__cov_axis_convert(E_AB)
        E_AB[np.isnan(E_AB)]=0
        E_AB+=1e-4
        return E_AB


    @staticmethod
    def merge_measurments(measurements):
        """
        Returns a refined uncertain transform using a different mesurment of the same object in the same reference frame
        """
        positions = [m.position for m in measurements]
        covs = [m.covariance_matrix[3:,3:] for m in measurements]
        position, position_cov = UncertainTransform.gausian_product(positions, covs)

       # compute new orientation mean and variance
        u = [np.radians( m.rotation_euler()[[1,0,2]]) for m in measurements]
        covs = [ m.covariance_matrix[:3,:3] + np.identity(3)*1e-5 for m in measurements]

        orientation_euler_xyz_c, orientation_cov = UncertainTransform.gausian_product(np.cos(u), covs)
        orientation_euler_xyz_s, _=  UncertainTransform.gausian_product(np.sin(u), covs)
        orientation_euler_xyz = np.degrees(np.arctan2(orientation_euler_xyz_s, orientation_euler_xyz_c)) 
        orientation_euler_yxz = orientation_euler_xyz[[1,0,2]]

        # create uncertain transform from new mean and variance
        result =  UncertainTransform.fromEuler(position,
                                        orientation_euler_yxz,
                                        np.ones(3)*1e-6,
                                        np.ones(3)*1e-6)

        result.covariance_matrix[3:,3:] =position_cov
        result.covariance_matrix[:3,:3] = orientation_cov
        return result

    def Logmap(self):
        logmap =  gtsam.Pose3.Logmap(UncertainTransform.TransformToGTSAM(self))
        # convert axis from gtsam to transform
        logmap[[0,1,2,3,4,5]] = logmap[[0,2,1,3,5,4]]
        return logmap
    
    @staticmethod
    def Expmap(logmap):
        # convert axis from transform to gtsam
        logmap[[0,1,2,3,4,5]] = logmap[[0,2,1,3,5,4]]
        pose3 =  gtsam.Pose3.Expmap(logmap)
        return UncertainTransform.GTSAMToUncertainTransform(pose3)

    def get_covariance_in_origin_frame(self):
        return self.to_inverse().covariance_matrix

    def get_samples(self, nb_samples = 1000):
            
        mean =np.array([0,0,0,0,0,0])
        t = Transform(self.position, self.rotation_quat)#.to_inverse()
        samples_transform = []
        cov =  self.covariance_matrix
        for i in range(nb_samples):
            # sample the multivariate distribution from covariance matrix around bodyframe
            sample =(np.random.multivariate_normal(mean, cov, 1)).reshape(-1,1)
            # transform from tangent space to pose
            p = UncertainTransform.Expmap(sample)
            # apply mean value (t) 
            p = p.apply_transform(t)
            samples_transform.append(p)
        return samples_transform


    # For call to repr(). Prints object's information
    def __repr__(self):
        return f'UncertainTransform(position: {self.position}, orientation: {self.rotation_euler()}, error: {np.round(np.sum(self.covariance_matrix, axis=0),2)})' 

    # For call to str(). Prints readable form
    def __str__(self):
        return f'UncertainTransform(position: {self.position}, orientation: {self.rotation_euler()}, error: {np.round(np.sum(self.covariance_matrix, axis=0),2)})' 

    @staticmethod
    def __matrix_gtsam_to_transform(gtsam_matrix):
        ours = np.identity(4)
        r = gtsam_matrix[:3,:3].T
        r[ :,[0,1,2]]=r[ :,[0,2,1]]
        r[[0,1,2],:]=r[[0,2,1],:]
        t = gtsam_matrix[[0,2,1],3]
        ours[:3,:3] = r
        ours[3,:3] = t
        return ours
        
    @staticmethod
    def __matrix_transform_to_gtsam(our_matrix):
        gtsam = np.identity(4)
        r = our_matrix[:3,:3].T
        r[ :,[0,1,2]]=r[ :,[0,2,1]]
        r[[0,1,2],:]=r[[0,2,1],:]
        t = our_matrix[3,[0,2,1]]
        gtsam[:3,:3] = r
        gtsam[:3,3] = t
        return gtsam


    @staticmethod
    def __cov_axis_convert(covariance_mat):
        """
        Convert to transform covariance matrix
        """
        covariance_mat=covariance_mat.T.copy()
        covariance_mat[ :,[0,1,2,3,4,5]]=covariance_mat[ :,[0,2,1,3,5,4]]
        covariance_mat[[0,1,2,3,4,5],:]=covariance_mat[[0,2,1,3,5,4],:]
        return covariance_mat


    @staticmethod
    def TransformToGTSAM(transform):
        """
        Convert Transform to gtsam Pose3
        """
        gtsam_matrix = UncertainTransform.__matrix_transform_to_gtsam(transform.as_matrix)
        translation = gtsam_matrix[:3,3]
        return Pose3(Rot3(gtsam_matrix[:3,:3]), Point3(*translation))


    @staticmethod
    def GTSAMToUncertainTransform(pose3, marginal_covariance=None):
        """
        Convert gtsam Pose3 to Transform
        """
        ours_matrix = UncertainTransform.__matrix_gtsam_to_transform(pose3.matrix())
        t = Transform.fromMatrix(ours_matrix)
        if marginal_covariance is None:
            return t
        else:
            return UncertainTransform(t.position,
                                      t.rotation_quat,
                                      UncertainTransform.__cov_axis_convert(marginal_covariance))
    @staticmethod
    def UncertainTransformToGTSAM(uncertain_transform):
        gtsam_pose = UncertainTransform.TransformToGTSAM(uncertain_transform)
        gtsam_covariance = UncertainTransform.__cov_axis_convert(uncertain_transform.covariance_matrix)+ np.identity(6)*1e-5
        noise_model = gtsam.noiseModel.Gaussian.Covariance(gtsam_covariance)
        return gtsam_pose, noise_model


    @staticmethod
    def apply_transform_covariance(covariance_matrix, bTw_transform):
        bTw = UncertainTransform.TransformToGTSAM(bTw_transform)
        bSb= UncertainTransform.__cov_axis_convert(covariance_matrix)
        adjTbb1_inv = bTw.inverse().AdjointMap()
        origin_frame_cov = adjTbb1_inv.T.dot(bSb).dot(adjTbb1_inv)
        origin_frame_cov =  UncertainTransform.__cov_axis_convert(origin_frame_cov)
        origin_frame_cov[np.isnan(origin_frame_cov)]=+1e-4
        origin_frame_cov+=1e-4
        return origin_frame_cov