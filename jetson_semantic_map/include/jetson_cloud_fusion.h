#pragma once

#include <ros/ros.h>
#include <jetson_color_maps.h>

#include <sensor_msgs/PointCloud2.h>
#include <jetson_semantic_map/MappOccIdx.h>
#include <tf2_ros/transform_listener.h>
#include <memory>
#include <Eigen/Core>
#include <mutex>
#include <thread>

template<int _N>
class SemanticFusor;

class CloudColoring
{
public:
    CloudColoring(ros::NodeHandle &nh);
    ~CloudColoring();

private:

    template<int NUM_CLASSES>
    void addCloudToFusor( const sensor_msgs::PointCloud2ConstPtr & cloud_msg, const bool prior = false );

    template<int NUM_CLASSES>
    sensor_msgs::PointCloud2Ptr getFusedCloud(jetson_semantic_map::MappOccIdxPtr & map_occ_idx_msg, const bool use_semantic_color = true );

    template<int NUM_CLASSES>
    void fuseSemanticCloudCallback( const sensor_msgs::PointCloud2ConstPtr &cloud);

    template<int NUM_CLASSES>
    void fusePriorCloudCallback( const sensor_msgs::PointCloud2ConstPtr &cloud);

    template<int NUM_CLASSES>
    void fused_publisher_run();

    int num_classes;
    int num_cameras;

    sensor_msgs::PointField field_rgb, field_semantic;

    struct PointRGB{
      union{
        struct{
          uint8_t b;
          uint8_t g;
          uint8_t r;
          uint8_t a;
        };
        float rgb;
      };
    };

    // params
    std::string m_cloud_topic = "";
    std::string m_fusion_cloud_topic = "";
    std::string m_fixed_frame = "";
    bool m_publish_in_fixed_frame = false;

    int m_semantic_window_size = 10;
    bool m_remove_older = false;
    bool m_store_all = false;
    bool m_fuse_semantic_window = false;

    //tf2
    tf2_ros::Buffer m_tfBuffer;
    tf2_ros::TransformListener m_tfListener;
    std::map<std::string, Eigen::Affine3d> m_transforms_base_cam;

    ros::Subscriber m_prior_map_subscriber;
    std::vector<ros::Subscriber> m_fuse_only_subscribers;

    ros::Publisher m_fused_cloud_publisher;
    ros::Publisher m_map_occupied_idx_publisher;

    int m_person_class_idx;
    std::vector<int> m_dynamic_class_idxs;
    bool m_no_ray_tracing;

    std::atomic<bool> m_fusor_updated = false;
    typedef std::shared_ptr<SemanticFusor<SemanticsColorMap::NUM_CLASSES_INDOOR>> SemanticFusorADE20KIndoorPtr;
    SemanticFusorADE20KIndoorPtr m_fusor_ade20Kindoor = nullptr;
    typedef std::shared_ptr<SemanticFusor<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>> SemanticFusorColorOnlyPtr;
    SemanticFusorColorOnlyPtr m_fusor_color_only = nullptr;
    double m_fusor_side_length, m_fusor_voxel_length;

    bool m_shutdown_requested = false;
    std::mutex m_fusor_mutex;
    std::thread m_fused_publisher_thread;
};
