#include "jetson_cloud_fusion.h"
#include "jetson_color_maps.h"
#include "JetsonSemanticFusor.h"
#include <geometry_msgs/TransformStamped.h>
#include <tf2_eigen/tf2_eigen.h>

#include <algorithm>

template<typename T>
std::string t_to_string(const T &t)
{
    std::ostringstream ss;
    ss << t;
    return ss.str();
}

CloudColoring::~CloudColoring()
{
    m_shutdown_requested = true;
    m_fused_publisher_thread.join();
}

CloudColoring::CloudColoring(ros::NodeHandle &nh)
        : m_tfBuffer(), m_tfListener(m_tfBuffer)
{
    nh.param<std::string>("cloud_topic", m_cloud_topic, "/cloud_coloring/semantic_cloud");
    nh.param<std::string>("fusion_output_cloud_topic", m_fusion_cloud_topic, "colorize/fused_cloud");

    nh.param<std::string>("fixed_frame", m_fixed_frame, "field");
    nh.param<bool>("publish_in_fixed_frame", m_publish_in_fixed_frame, false);
    ROS_INFO("fixed_frame: %s, publish in fixed frame: %d.", m_fixed_frame.c_str(), m_publish_in_fixed_frame);

    nh.param<int>("semantic_window_size", m_semantic_window_size, 10);
    nh.param<bool>("fuse_semantic_window", m_fuse_semantic_window, false);
    nh.param<bool>("remove_older", m_remove_older, false);
    nh.param<bool>("store_all", m_store_all, false);
    nh.param("n_classes", num_classes, 32);
    nh.param("n_cameras", num_cameras, 4);

    m_fused_cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>(m_fusion_cloud_topic, 2);
    m_map_occupied_idx_publisher = nh.advertise<jetson_semantic_map::MappOccIdx>("map_occupied_idx", 2);

    nh.param<double>("side_length", m_fusor_side_length, 30);
    nh.param<double>("voxel_length", m_fusor_voxel_length, 0.10);
    nh.param<bool>("no_raytracing", m_no_ray_tracing, false);

    switch(num_classes){
    case SemanticsColorMap::NUM_CLASSES_INDOOR:
      m_prior_map_subscriber = nh.subscribe("/cloud_pcd", 1, &CloudColoring::fusePriorCloudCallback<SemanticsColorMap::NUM_CLASSES_INDOOR>, this );
      for (int cam_idx = 1; cam_idx <= num_cameras; ++cam_idx) { // one subscriber for each jetson board!
        m_fuse_only_subscribers.push_back(nh.subscribe( "/d455_" + std::to_string(cam_idx) + m_cloud_topic, 1, &CloudColoring::fuseSemanticCloudCallback<SemanticsColorMap::NUM_CLASSES_INDOOR>, this ));
      }
      m_person_class_idx = 14;
      m_dynamic_class_idxs = {9, 10};
      m_fusor_ade20Kindoor = SemanticFusor<SemanticsColorMap::NUM_CLASSES_INDOOR>::Create(m_fusor_side_length, m_fusor_voxel_length, m_semantic_window_size, m_remove_older, m_store_all, m_person_class_idx, m_dynamic_class_idxs, m_no_ray_tracing);
      m_fused_publisher_thread = std::thread{std::bind(&CloudColoring::fused_publisher_run<SemanticsColorMap::NUM_CLASSES_INDOOR>, this)};
      break;
    case SemanticsColorMap::NUM_CLASSES_COLOR_ONLY:
      m_prior_map_subscriber = nh.subscribe("/cloud_pcd", 1, &CloudColoring::fusePriorCloudCallback<SemanticsColorMap::NUM_CLASSES_INDOOR>, this );
      m_fuse_only_subscribers.push_back(nh.subscribe( m_cloud_topic, 1, &CloudColoring::fuseSemanticCloudCallback<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>, this ));
      m_person_class_idx = -1;
      m_fusor_color_only = SemanticFusor<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>::Create(m_fusor_side_length, m_fusor_voxel_length, m_semantic_window_size, m_remove_older, m_store_all, m_person_class_idx, m_dynamic_class_idxs, m_no_ray_tracing);
      m_fused_publisher_thread = std::thread{std::bind(&CloudColoring::fused_publisher_run<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>, this)};
      break;
    default:
      return;
    }

    // Fields to add. Offsets need to be set in callback depending on incoming point fields.. !
    field_rgb.name = "rgb";           field_rgb.datatype = sensor_msgs::PointField::FLOAT32;      field_rgb.count = 1;
    field_semantic.name = "semantic"; field_semantic.datatype = sensor_msgs::PointField::FLOAT32; field_semantic.count = num_classes;
}

template<int NUM_CLASSES>
void CloudColoring::addCloudToFusor( const sensor_msgs::PointCloud2ConstPtr & cloud_msg, const bool prior )
{
    typedef std::shared_ptr<SemanticFusor<NUM_CLASSES>> SemanticFusorPtr;
    static SemanticFusorPtr p_fusor = nullptr;
    if ( ! cloud_msg ) return;
    if ( ! p_fusor )
    {
        if constexpr ( NUM_CLASSES == SemanticsColorMap::NUM_CLASSES_INDOOR ) p_fusor = m_fusor_ade20Kindoor;
        if constexpr ( NUM_CLASSES == SemanticsColorMap::NUM_CLASSES_COLOR_ONLY ) p_fusor = m_fusor_color_only;
    }

    // Assume static transforms (static cameras)
    Eigen::Affine3d transform_base_cloud;
    const auto& transform_it = m_transforms_base_cam.find(cloud_msg->header.frame_id);
    if ( transform_it == m_transforms_base_cam.end() ) { // transform not yet acquired
      geometry_msgs::TransformStamped tfTransformCloud;
      try{
          tfTransformCloud = m_tfBuffer.lookupTransform(m_fixed_frame, cloud_msg->header.frame_id, ros::Time(0)); // static transform, so just get the latest available
      } catch (tf2::TransformException& ex){
          ROS_ERROR_STREAM(ex.what());
          return;
      }
      transform_base_cloud = tf2::transformToEigen(tfTransformCloud.transform);
      m_transforms_base_cam[cloud_msg->header.frame_id] = transform_base_cloud;
    }
    else { // transform already acquired
      transform_base_cloud = transform_it->second;
    }

    //std::unique_lock<std::mutex> lock (m_fusor_mutex);
    p_fusor->addCloud( cloud_msg , transform_base_cloud, prior );
    m_fusor_updated = true;
}

template<int NUM_CLASSES>
sensor_msgs::PointCloud2Ptr CloudColoring::getFusedCloud(jetson_semantic_map::MappOccIdxPtr & map_occ_idx_msg, const bool use_semantic_color )
{
    typedef std::shared_ptr<SemanticFusor<NUM_CLASSES>> SemanticFusorPtr;
    static SemanticFusorPtr p_fusor = nullptr;
    if ( ! p_fusor )
    {
        if constexpr ( NUM_CLASSES == SemanticsColorMap::NUM_CLASSES_INDOOR ) p_fusor = m_fusor_ade20Kindoor;
        if constexpr ( NUM_CLASSES == SemanticsColorMap::NUM_CLASSES_COLOR_ONLY ) p_fusor = m_fusor_color_only;
    }
    return p_fusor->getFusedVoxel(map_occ_idx_msg, m_fixed_frame, use_semantic_color);
}

template<int NUM_CLASSES>
void CloudColoring::fuseSemanticCloudCallback ( const sensor_msgs::PointCloud2ConstPtr & cloud )
{
    //ROS_INFO_STREAM("got cloud: " << cloud->header.frame_id);
    if ( cloud->header.frame_id == "" ){
      ROS_WARN_STREAM_THROTTLE(1,"empty frame_id. size: " << cloud->height << " x " << cloud->width );
      return;
    }
    if ( m_fuse_semantic_window && cloud != nullptr ) //&& m_fused_cloud_publisher.getNumSubscribers() > 0 )
    {
        addCloudToFusor<NUM_CLASSES> ( cloud );
//        sensor_msgs::PointCloud2Ptr msg ( new sensor_msgs::PointCloud2 );
//        *msg = *cloud;
//        sensor_msgs::PointCloud2Ptr fused_msg = getFusedCloud<NUM_CLASSES>( msg );
//        if ( fused_msg ) m_fused_cloud_publisher.publish(fused_msg);
//        else ROS_WARN_STREAM("empty fusion.");
    }
}

template<int NUM_CLASSES>
void CloudColoring::fusePriorCloudCallback ( const sensor_msgs::PointCloud2ConstPtr & cloud )
{
    static int num_prior = 0;
    //ROS_INFO_STREAM("got cloud: " << cloud->header.frame_id);
    if ( cloud->header.frame_id == "" ){
      ROS_WARN_STREAM_THROTTLE(1,"empty frame_id. size: " << cloud->height << " x " << cloud->width );
      return;
    }
    if ( m_fuse_semantic_window && cloud != nullptr ) //&& m_fused_cloud_publisher.getNumSubscribers() > 0 )
    {
        addCloudToFusor<NUM_CLASSES> ( cloud, true );
    }

    ++num_prior;

    if(num_prior >= Voxel<NUM_CLASSES>::Nocc)
      m_prior_map_subscriber.shutdown();
}

template<int NUM_CLASSES>
void CloudColoring::fused_publisher_run()
{
    ros::Time start_time = ros::Time::now();
    sensor_msgs::PointCloud2Ptr last_fused_msg = nullptr;
    jetson_semantic_map::MappOccIdxPtr last_map_occ_msg = nullptr;
    const std::chrono::milliseconds time_between_map_publish_ns ( 1000 );
    std::chrono::time_point<std::chrono::system_clock> last_fused_publish = std::chrono::system_clock::now();

    while(!m_shutdown_requested && ros::ok())
    {
        bool fusor_was_updated = false;
        bool fused_cloud_wanted = false;
        sensor_msgs::PointCloud2Ptr new_fused_msg = nullptr;
        jetson_semantic_map::MappOccIdxPtr new_map_occ_msg( m_map_occupied_idx_publisher.getNumSubscribers() > 0 ? new jetson_semantic_map::MappOccIdx() : nullptr) ;
        {
            //std::unique_lock lock{m_fusor_mutex};
            fusor_was_updated = m_fusor_updated;
            fused_cloud_wanted = m_fused_cloud_publisher.getNumSubscribers() > 0 || m_map_occupied_idx_publisher.getNumSubscribers() > 0;
            if ( fusor_was_updated )
            {
                m_fusor_updated = false;
                if ( fused_cloud_wanted ) new_fused_msg = getFusedCloud<NUM_CLASSES>(new_map_occ_msg, true );
                if ( new_fused_msg ) ROS_INFO_STREAM("Got semantic fused map.");
            }
            else
            {
                new_fused_msg = last_fused_msg;
                new_map_occ_msg = last_map_occ_msg;
            }
        }
        const ros::Time now = ros::Time::now();
        const std::chrono::time_point<std::chrono::system_clock> cur_time = std::chrono::system_clock::now();
        const bool publish_fused_again = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - last_fused_publish) > time_between_map_publish_ns;

        if ( fused_cloud_wanted && ( fusor_was_updated || publish_fused_again ) )
        {
            sensor_msgs::PointCloud2::Ptr msg = new_fused_msg ? new_fused_msg : (publish_fused_again && last_fused_msg ? last_fused_msg : nullptr);
            jetson_semantic_map::MappOccIdx::Ptr msg_occ = new_map_occ_msg ? new_map_occ_msg : (publish_fused_again && last_map_occ_msg ? last_map_occ_msg : nullptr);
            if ( msg )
            {
                msg->header.stamp = now;
                m_fused_cloud_publisher.publish(msg);
                ROS_INFO_STREAM("Published fused semantic cloud! " << (msg->header.stamp-start_time).toNSec() << " pts: " << msg->width << " x " << msg->height);
            }
            else ROS_WARN_STREAM_THROTTLE(1,"Empty fusion (semantic).");

            if ( msg_occ )
            {
                msg_occ->header.stamp = now;
                m_map_occupied_idx_publisher.publish(msg_occ);
                ROS_INFO_STREAM("Published occupied map indices! " << (msg_occ->header.stamp-start_time).toNSec() << " pts: " << msg_occ->occ_indices.size());
            }
            else ROS_WARN_STREAM_THROTTLE(1,"Empty map indices (semantic).");

            last_fused_publish = cur_time;
        }
        last_fused_msg = new_fused_msg;
        last_map_occ_msg = new_map_occ_msg;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        //if ( ! fusor_was_updated )
        //    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
