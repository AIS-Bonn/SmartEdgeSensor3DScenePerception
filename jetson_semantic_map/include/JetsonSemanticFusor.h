#pragma once

#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <memory>
#include <deque>
#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
//#include <config_server/parameter.h>
#include <mutex>

#include <jetson_color_maps.h>
#include <jetson_semantic_map/MappOccIdx.h>

template <int _N>
struct SemanticClass
{
    using VecN = Eigen::Matrix<double, _N, 1>;
    typedef std::shared_ptr<SemanticClass> Ptr;

    static Ptr create ( const Ptr & other = nullptr );
    void addLogProb ( const VecN & new_log_prob );
    void addProb ( const VecN & new_prob );
    int getArgMaxClass ( ) const;
    VecN getSemantic( ) const;
    //void add ( const SemanticClass & other );

    bool m_empty = true;
    bool m_use_log_prob = false;
    VecN m_class = VecN::Constant(1./_N); // VecN::Zero();
};

template <int _N>
class Voxel
{
public:
    typedef std::shared_ptr<Voxel > Ptr;
    typedef SemanticClass<_N> SemanticClassType;

    static constexpr int Nocc = 16;
    static constexpr int8_t occupied = 1;
    static constexpr int8_t unknown = 0;
    static constexpr int8_t free = -1;
    static constexpr int minOcc = occupied * 2;
    static constexpr int minFree = free * 2;
    typedef Eigen::Array<int8_t, Nocc, 1> VecNocc;

    static Ptr Create( const int & window_size = 10, const Eigen::Vector3d & first_view_dir = Eigen::Vector3d::UnitX() )
    {
        return std::make_shared<Voxel>(window_size, first_view_dir );
    }
    Voxel( const int & window_size, const Eigen::Vector3d & first_view_dir ) : m_window_size ( window_size ), m_needs_update ( true ), m_valid ( false ), m_first_view_dir ( first_view_dir ) {}

    struct VoxelData
    {
        static constexpr int max_num_fused = 1000;
        static constexpr int min_num_fused = 20;
        SemanticClassType m_semantic;

        int m_num = 0;
        int m_num_color = 0;
        int m_scan_id = -1;

        Eigen::Vector3d m_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d m_sum_color = Eigen::Vector3d::Zero();
        Eigen::Matrix3d m_sum_squares = Eigen::Matrix3d::Zero();

        VoxelData & operator+=(const VoxelData & rhs )
        {
            if (rhs.m_num > 0 && m_num < max_num_fused)
            {
                if ( m_num == 0 )
                {
                    m_sum_squares = rhs.m_sum_squares;
                    m_sum = rhs.m_sum;
                    m_num = rhs.m_num;
                    m_num_color = rhs.m_num_color;
                    m_sum_color = rhs.m_sum_color;
                    m_semantic.m_use_log_prob = rhs.m_semantic.m_use_log_prob;
                }
                else
                {
                    const Eigen::Vector3d deltaS = rhs.m_num * m_sum - m_num * rhs.m_sum;
                    m_sum_squares += rhs.m_sum_squares +
                            1.0 / (m_num * rhs.m_num * (rhs.m_num + m_num)) * deltaS * deltaS.transpose();
                    m_sum += rhs.m_sum;
                    m_num += rhs.m_num;
                }
                m_num_color += rhs.m_num_color;
                m_sum_color += rhs.m_sum_color;

                if(m_semantic.m_use_log_prob != rhs.m_semantic.m_use_log_prob){
                  ROS_WARN("Semantic class add: incoherent probability representation: log_prob (ours): %d, (other): %d", (int)m_semantic.m_use_log_prob, (int)rhs.m_semantic.m_use_log_prob);
                }
                if (m_semantic.m_use_log_prob && rhs.m_semantic.m_use_log_prob)
                  m_semantic.addLogProb(rhs.m_semantic.m_class);
                else if (m_semantic.m_use_log_prob && !rhs.m_semantic.m_use_log_prob)
                  m_semantic.addLogProb(rhs.m_semantic.m_class.array().log());
                else if (!m_semantic.m_use_log_prob && rhs.m_semantic.m_use_log_prob)
                  m_semantic.addProb(rhs.m_semantic.m_class.array().exp());
                else
                  m_semantic.addProb(rhs.m_semantic.m_class);
            }
            return *this;
        }
        void addPoint (const Eigen::Vector3d & new_point )
        {
            if (m_num < max_num_fused)
            {
                if ( m_num == 0 )
                {
                    m_sum = new_point;
                    m_num = 1;
                }
                else
                {
                    const Eigen::Vector3d deltaS = m_sum - m_num * new_point;
                    m_sum_squares += 1.0 / (m_num * (1 + m_num)) * deltaS * deltaS.transpose();
                    m_sum += new_point;
                    m_num += 1;
                }
            }
        }
        void addColor ( const Eigen::Vector3d & new_color )
        {
            if ( m_num_color >= max_num_fused ) return;
            m_sum_color += new_color;
            m_num_color += 1;
        }
    };

    void addNewScan( const int & scan_id = 0 );
    void addNewPoint( const Eigen::Vector3d & new_point, const Eigen::VectorXd * new_semantic = nullptr, const Eigen::Vector3d * new_color = nullptr, const bool prior = false );
    void update();

    void push_occ ( const int8_t & mea )
    {
        m_meas_occ[next_pos_occ] = mea;
        ++next_pos_occ;
        if ( next_pos_occ > Nocc )
            next_pos_occ = 0;
    }
    int8_t count() const
    {
        const int sum = m_meas_occ.sum();
        return sum < minFree ? free : ( sum > minOcc ? occupied : unknown );
    }


    Eigen::Vector3d getMean ( ) const
    {
        if ( m_fused.m_num == 0 )
            return Eigen::Vector3d::Constant(std::numeric_limits<double>::signaling_NaN());
        else
            return m_mean;
    }

    Eigen::Vector3d getMeanColor ( ) const
    {
        if ( !m_valid && m_fused.m_num_color == 0 )
            return Eigen::Vector3d::Zero();
        else
            return m_mean_color;
    }
    Eigen::Vector3d getNormal() const
    {
        if ( !m_valid && m_fused.m_num < VoxelData::min_num_fused )
            return Eigen::Vector3d::Zero();
        else
            return m_normal;
    }

    void setFirstViewDir ( const Eigen::Vector3d & firstViewDir )
    {
        m_first_view_dir = firstViewDir;
    }

    int getNum () const
    {
        return m_fused.m_num;
    }

    int getArgMaxClass() const
    {
        if ( m_fused.m_num == 0 )
            return -1;
        else
            return m_fused.m_semantic.getArgMaxClass();
    }

    typename SemanticClassType::VecN getSemantic() const
    {
      return m_fused.m_semantic.getSemantic();
    }

    void remove_older( const int & last_id )
    {
        while ( m_data.size()>0 && m_data.front().m_scan_id < last_id )
            m_data.pop_front();
    }

    void clear()
    {
        m_data.clear();
        m_valid = false;
        m_needs_update = true;
    }

    const int& getObscureAngleCnt() const
    {
      return m_obscure_angle_counter;
    }

    void incrObscureAngleCnt()
    {
      ++m_obscure_angle_counter;
    }

    void resetObscureAngleCnt()
    {
      m_obscure_angle_counter = 0;
    }
    
    bool isPrior()
    {
        return m_is_prior;
    }

    bool isEmpty() const
    {
        return m_data.empty();
    }

    bool isValid() const
    {
        return m_valid;
    }
    
    float getSmallestEigenvalue ( const bool use_normalized_eigenvalue ) const
    {
        return use_normalized_eigenvalue ? m_normalized_eigenvalue : m_normal_eigenvalue;
    }
    

private:
    std::deque<VoxelData> m_data;
    VecNocc m_meas_occ = VecNocc::Constant(unknown);
    int next_pos_occ = 0;
    VoxelData m_fused;
    Eigen::Vector3d m_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d m_mean_color = Eigen::Vector3d::Zero();
    Eigen::Vector3d m_normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d m_first_view_dir = Eigen::Vector3d::UnitX();
    float m_normal_eigenvalue = std::numeric_limits<float>::signaling_NaN();
    float m_normalized_eigenvalue = std::numeric_limits<float>::signaling_NaN();

    int m_obscure_angle_counter = 0;
    bool m_is_prior = false;
    bool m_valid = false;
    bool m_needs_update = true;
    int m_window_size = 10;
};

template <int _N>
class SemanticFusor
{
public:

    struct Config{ // TODO: hard-coded default values. Original Node uses Nimbro Config Server Params.
        int min_num_points_for_vis = 1;
        int min_num_points_for_vis_dynamic = 1;
        bool use_normalized_eigenvalue = false;
        bool publish_normal = false;
        bool publish_semantic = true;
        float x_max = 13.f, x_min = -13.f;
        float y_max = 7.f, y_min = -7.f;
        float z_max = 2.3f, z_min = -1.f;
//        config_server::Parameter<int> min_num_points_for_vis{"min_num_points_for_vis", 0,1,100,10};
//        config_server::Parameter<int> min_num_points_for_vis_dynamic{"min_num_points_for_vis_dynamic", 0,1,100,1};
//        config_server::Parameter<bool> use_normalized_eigenvalue{"use_normalized_eigenvalue", false};
//        config_server::Parameter<bool> publish_normal{"publish_normal", false};
//        config_server::Parameter<bool> publish_semantic{"publish_semantic", false};
//        config_server::Parameter<float> x_min{"x_min", -100,0.1,100,-20};
//        config_server::Parameter<float> x_max{"x_max", -100,0.1,100,20};
//        config_server::Parameter<float> y_min{"y_min", -100,0.1,100,-20};
//        config_server::Parameter<float> y_max{"y_max", -100,0.1,100,20};
//        config_server::Parameter<float> z_min{"z_min", -10,0.1,10,-2};
//        config_server::Parameter<float> z_max{"z_max", -10,0.1,10,10};
// //        cloud_fusor_node:
// //             min_num_points_for_vis: 1
// //             min_num_points_for_vis_dynamic: 1
// //             publish_normal: 0
// //             publish_semantic: 1
// //             use_normalized_eigenvalue: 0
// //             x_max: 13
// //             x_min: -13
// //             y_max: 7
// //             y_min: -7
// //             z_max: 2.3
// //             z_min: -1
    };
    Config m_config;

    bool m_dont_use_color = false;
    //typedef int IndexType;
    typedef int32_t IndexType;
    typedef Eigen::Matrix<int32_t,3,1> IndexVec3Type;
    typedef std::shared_ptr<SemanticFusor > Ptr;

    static Ptr Create(const double& side_length, const double& voxel_length, const int & window_size, const bool & remove_older = false, const bool & store_all = false, const int & exclude_class = -1, const std::vector<int>& dyn_classes = std::vector<int>(), const bool & no_ray_tracing = false );
    void addCloud ( const sensor_msgs::PointCloud2ConstPtr & cloud, const Eigen::Affine3d & pose_ws, const bool prior = false );

    void bresenham3D( const IndexVec3Type & scan_origin_index,
                      const std::unordered_set<IndexType> & endpoint_cells,
                      std::unordered_set<IndexType> & new_unknown_cells,
                      std::unordered_set<IndexType> & new_free_cells );

    void updateUpdatedIndices(const std::vector<IndexType> &updated_occupied_indices, const std::vector<IndexType> &updated_free_indices, const std::vector<IndexType> &updated_unknown_indices);

    sensor_msgs::PointCloud2Ptr getFusedVoxel(jetson_semantic_map::MappOccIdxPtr & map_occ_idx_msg, const std::string& fixed_frame, const bool use_semantic_color = true );
    int addVoxel2Cloud(const typename Voxel<_N>::Ptr & voxel_ptr, sensor_msgs::PointCloud2Ptr & msg, const size_t & idx,
                       const size_t & point_step, const size_t offset_rgb,const size_t offset_normal,const size_t offset_sev, const size_t offset_semantic,
                       const int & min_num_points, const int & min_num_points_dynamic, const float & min_x, const float & max_x, const float & min_y, const float & max_y, const float & min_z, const float & max_z, const SemanticsColorMap & cmap,
                       const bool use_normalized_eigenvalue, const bool use_semantic_color, const bool with_normal, const bool with_semantic);

    inline IndexVec3Type getMiddleIndexVector( const IndexType & num_cells ) const
    {
        return IndexVec3Type( num_cells>>1, num_cells>>1, num_cells>>1 );
    }

    inline IndexType toCellIndex ( const Eigen::Vector3d & point, const double & voxel_scale, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels ) const;
    inline IndexType toCellIndexFromIndexVector ( const IndexVec3Type & idx_vec, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels ) const;
    inline IndexVec3Type fromCellIndexToIndexVector( const IndexType & idx, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels) const;
    inline IndexVec3Type toCellIndexVector ( const Eigen::Vector3d & point, const double & voxel_scale) const;
    inline Eigen::Vector3d fromCellIndexToCenter(const IndexType& idx, const double & voxel_scale, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels) const;

    SemanticFusor(const double& side_length, const double& voxel_length, const int & window_size, const bool & remove_older = false, const bool & store_all = false, const int & exclude_class = -1, const std::vector<int>& dyn_classes = std::vector<int>(), const bool & no_ray_tracing = false )
        : m_side_length(side_length), m_voxel_length(voxel_length), m_window_size (std::max(1,window_size)), m_remove_older(remove_older), m_store_all(store_all), m_exclude_class(exclude_class), m_dynamic_classes(dyn_classes), m_no_ray_tracing(no_ray_tracing)
    {
      m_voxels.reserve(1e7);
      m_free_indices.reserve(1e7);
      m_occupied_indices.reserve(1e6);
      m_unknown_indices.reserve(1e6);
    }

    //void copyVoxels();
        
        
private:
    const double m_side_length;
    const double m_voxel_length;
    int m_scan_id = -1;
    int m_window_size = 10;
    bool m_remove_older = false;
    bool m_store_all = false;
    int m_exclude_class = -1;
    std::vector<int> m_dynamic_classes;
    bool m_no_ray_tracing;
//    std::vector<IndexType> m_updatedIndices;
    std::mutex m_voxels_mutex; //, m_indices_mutex;
    std::unordered_map<IndexType, typename Voxel<_N>::Ptr> m_voxels;
    //std::unordered_map<IndexType, typename Voxel<_N>::Ptr> m_voxels_copy;
    std::unordered_set<IndexType> m_occupied_indices, m_free_indices, m_unknown_indices;
    //std::unordered_set<IndexType> m_occupied_indices_copy, m_free_indices_copy, m_unknown_indices_copy;
};
