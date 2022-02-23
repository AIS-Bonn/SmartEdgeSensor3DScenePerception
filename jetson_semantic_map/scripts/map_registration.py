import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import math
import time
import copy

import open3d as o3d

def multiscale_icp(source, target, voxel_size, max_iter, init_transformation = np.identity(4)):
    current_transformation = init_transformation
    for i, scale in enumerate(range(len(max_iter))): # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = voxel_size[scale] * 1.4 # TODO: voxel_size[0] to keep this constant ?
        print("voxel_size %f" % voxel_size[scale])
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])

        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius = voxel_size[scale] * 2.0, max_nn = 30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                    radius = voxel_size[scale] * 2.0, max_nn = 30))

        result_icp = o3d.pipelines.registration.registration_icp(source_down, target_down,
                        distance_threshold, current_transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iter))
        current_transformation = result_icp.transformation

        if i == len(max_iter)-1:
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, voxel_size[scale] * 1.4,
                    result_icp.transformation)

    return (result_icp.transformation, information_matrix)


def main(args):
    prior_cloud = o3d.io.read_point_cloud('/home/sbultmann/datasets/hall/geometry/halle_lampen.pcd')
    map_cloud = o3d.io.read_point_cloud('/home/sbultmann/datasets/hall/geometry/map_5cm.pcd')
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.706, 0]]
    
    t_base_prior = np.eye(4)
    r_base_prior = Rot.from_euler('xyz', [1.57, 0, 2.26])
    trans_base_prior = np.array([19.14, -0.2, -0.08])
    t_base_prior[:3,:3] = r_base_prior.as_matrix()
    t_base_prior[:3,3] = trans_base_prior
    print(t_base_prior)
    
    map_cloud.paint_uniform_color(colors[0])
    map_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    prior_cloud.transform(t_base_prior)
    prior_cloud.paint_uniform_color([1, 0.706, 0])
    prior_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    o3d.visualization.draw_geometries([map_cloud, prior_cloud])
    
    #voxel_size_data2model_base = 0.40
    #voxel_size_data2model = [voxel_size_data2model_base, voxel_size_data2model_base / 2.0, voxel_size_data2model_base / 4.0, voxel_size_data2model_base / 8.0, voxel_size_data2model_base / 16.0]
    voxel_size_data2model = [0.10, 0.05]
    icp_max_iter_data2model =  [100, 70] #, 50, 30, 14]
    print("Registering pointcloud to robot model..")
    (transform, info) = multiscale_icp(map_cloud, prior_cloud, voxel_size_data2model, icp_max_iter_data2model)
    print(transform)
    print('\n')

    map_cloud.transform(transform)
    o3d.visualization.draw_geometries([map_cloud, prior_cloud])
    
    print('for config file:')
    print('tf_tuner_d455_1_link ( delta ):')
    print('\tlin:')
    print('\t\tx: {}'.format(transform[0,3]))
    print('\t\ty: {}'.format(transform[1,3]))
    print('\t\tz: {}'.format(transform[2,3]))
    rot = Rot.from_matrix(transform[0:3,0:3])
    rpy = rot.as_euler('xyz')
    print('\trot:')
    print('\t\tp: {}'.format(rpy[1]))
    print('\t\tr: {}'.format(rpy[0]))
    print('\t\ty: {}'.format(rpy[2]))
    print('\n')
    
    t_base_d455_1 = np.eye(4)
    r_base_d455_1 = Rot.from_euler('xyz', [3.102179990797158, 0.5741646231940096, -0.7221224548726103])
    trans_base_d455_1 = np.array([-3.369377944271212, 0.7594458749406078, 2.43820935974625])
    t_base_d455_1[:3,:3] = r_base_d455_1.as_matrix()
    t_base_d455_1[:3,3] = trans_base_d455_1
    
    transform_d455_1 = transform @ t_base_d455_1
    print('tf_tuner_d455_1_link:')
    print('\tlin:')
    print('\t\tx: {}'.format(transform_d455_1[0,3]))
    print('\t\ty: {}'.format(transform_d455_1[1,3]))
    print('\t\tz: {}'.format(transform_d455_1[2,3]))
    rot = Rot.from_matrix(transform_d455_1[0:3,0:3])
    rpy = rot.as_euler('xyz')
    print('\trot:')
    print('\t\tp: {}'.format(rpy[1]))
    print('\t\tr: {}'.format(rpy[0]))
    print('\t\ty: {}'.format(rpy[2]))
    print('\n')
    
    
    t_base_d455_1_opt = np.eye(4)
    r_base_d455_1_opt = Rot.from_quat([0.7946510745054201, 0.3768881380093155, 0.1866646779024932, 0.437768545282563]) #qx, qy, qz, qw
    trans_base_d455_1_opt = np.array([-3.331491284286757, 0.8047036926155615, 2.43630491006004])
    t_base_d455_1_opt[:3,:3] = r_base_d455_1_opt.as_matrix()
    t_base_d455_1_opt[:3,3] = trans_base_d455_1_opt
    
    transform_d455_1_opt = transform @ t_base_d455_1_opt
    print('tf_tuner_d455_1_optical:')
    print('\tlin:')
    print('\t\tx: {}'.format(transform_d455_1_opt[0,3]))
    print('\t\ty: {}'.format(transform_d455_1_opt[1,3]))
    print('\t\tz: {}'.format(transform_d455_1_opt[2,3]))
    rot = Rot.from_matrix(transform_d455_1_opt[0:3,0:3])
    rpy = rot.as_euler('xyz')
    print('\trot:')
    print('\t\tp: {}'.format(rpy[1]))
    print('\t\tr: {}'.format(rpy[0]))
    print('\t\ty: {}'.format(rpy[2]))
    print('\n')
    print('{}, {}'.format(transform_d455_1_opt[:3,3], rot.as_quat()))

if __name__ == '__main__':
    main(sys.argv)
