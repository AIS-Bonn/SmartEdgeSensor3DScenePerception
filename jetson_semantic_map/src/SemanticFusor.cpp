#include "JetsonSemanticFusor.h"
#include <ros/ros.h>
#include <cmath>
#include <chrono>

Eigen::Vector3d rgb2hsv ( const Eigen::Vector3d & rgb )
{
    int idxV = 0;
    const double V = rgb.maxCoeff( &idxV);
    const double m = rgb.minCoeff();
    const double Vm = V - m;
    const double S = V > 0 ? std::min<double>(Vm / V,1.) : 0;
    const double d = ( idxV == 0 ? rgb(1)-rgb(2) : ( idxV == 1 ? rgb(2)-rgb(0) : rgb(0)-rgb(1) ) );
    const double H = V == m ? 0 : 120*idxV + 60*d / Vm;
    return Eigen::Vector3d((H < 0 ? H+360 : H) / 360, S, V); // alles in [0,1]
}

Eigen::Vector3d hsv2rgb ( const Eigen::Vector3d & hsv )
{
    const double h6 = hsv(0) * 6;
    const int hi = int( h6 ); // H in [0,1] statt [0,360]
    const double f = h6-hi;
    const double V = hsv(2);
    const double p = V*(1-hsv(1));
    const double q = V*(1-hsv(1)*f);
    const double t = V*(1-hsv(1)*(1-f));
    switch ( hi )
    {
        case 1: return Eigen::Vector3d(q,V,p); break;
        case 2: return Eigen::Vector3d(p,V,t); break;
        case 3: return Eigen::Vector3d(p,q,V); break;
        case 4: return Eigen::Vector3d(t,p,V); break;
        case 5: return Eigen::Vector3d(V,p,q); break;
        default: return Eigen::Vector3d(V,t,p); break;
    }
}

template <int _N>
typename SemanticClass<_N>::Ptr SemanticClass<_N>::create ( const SemanticClass<_N>::Ptr & other )
{
    if ( ! other )
        return std::make_shared<SemanticClass<_N>>( );

    SemanticClass<_N>::Ptr n = std::make_shared<SemanticClass<_N>>();
    n->m_empty = other->m_empty;
    n->m_class = other->m_class;
    n->m_use_log_prob = other->m_use_log_prob;
    return n;
}

template <int _N>
void SemanticClass<_N>::addLogProb ( const SemanticClass<_N>::VecN & new_log_prob )
{
    m_use_log_prob = true;
    if ( m_empty)
    {
        m_class.setConstant(std::log(1./_N));
    }

    m_empty = false;

    m_class += new_log_prob;

    typename SemanticClass<_N>::VecN::Index maxIndex;
    const double log_max = m_class.maxCoeff(&maxIndex);
    double log_sum; // The max element is not part of the exp sum
    if (maxIndex == 0)
      log_sum = log_max + std::log1p(double((m_class.template tail<_N-1>().array()-log_max).exp().sum()));
    else if (maxIndex == _N-1)
      log_sum = log_max + std::log1p(double((m_class.template head<_N-1>().array()-log_max).exp().sum()));
    else
      log_sum = log_max + std::log1p(double((m_class.segment(0,maxIndex).array()-log_max).exp().sum()) + double((m_class.segment(maxIndex+1,_N-maxIndex-1).array()-log_max).exp().sum()));

    m_class.array() -= log_sum;
}

template <int _N>
void SemanticClass<_N>::addProb ( const SemanticClass<_N>::VecN & new_prob )
{
    if ( m_empty || m_class.norm() < 1e-4 )
    {
        m_class.setConstant(1./_N);
    }

    m_empty = false;
    m_use_log_prob = false;

#ifdef sum_it_up
    m_class.array() += new_prob.array();
#else

    //Semantic fusion:
    // CNN output: P(label_i | neues Bild )
    // P(label_i | alle bilder ) =  ( P( label_i | alle alten bilder ) * P( label_i | neues Bild) ) / Z
    // Z = sum ( P( label_i | alle alten bilder ) * P( label_i | neues Bild) )

    m_class.array() *= new_prob.array();
#endif
    m_class /= m_class.sum(); // re-normalize
}

//template <int _N>
//void SemanticClass<_N>::add ( const SemanticClass<_N> & other )
//{
//    if(m_empty)
//        m_use_log_prob = other.m_use_log_prob;
//    else if(m_use_log_prob != other.m_use_log_prob){
//      ROS_WARN("Semantic class add: incoherent probability representation: log_prob (ours): %d, (other): %d", (int)m_use_log_prob, (int)other.m_use_log_prob);
//    }
//    m_empty = false;
//    m_class += other.m_class;
//}

template <int _N>
int SemanticClass<_N>::getArgMaxClass ( ) const
{
    typename SemanticClass<_N>::VecN::Index idx = 0; // assuming background is first class.
    if ( m_empty ) return idx;
    m_class.array().maxCoeff(&idx); // don't need to differentiate between prob and log-prob here, as log is monotonous.
    //LOG(INFO) << "idx: " << idx << " mcl: " << m_class.transpose();
    return idx;
}

template <int _N>
typename SemanticClass<_N>::VecN SemanticClass<_N>::getSemantic ( ) const
{
  if (m_empty)
    return SemanticClass<_N>::VecN::Ones() * (1.0 / _N); // equal probability for all classes
  if ( m_use_log_prob )
    return m_class.array().exp();
  else
    return m_class;
}

template struct SemanticClass<SemanticsColorMap::NUM_CLASSES_INDOOR>;
template struct SemanticClass<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>;

template <int _N>
void Voxel<_N>::addNewScan( const int & scan_id )
{
    if (!m_data.empty() &&  m_data.back().m_scan_id == scan_id ) return; // already added
    while ( m_data.size() >= m_window_size )
        m_data.pop_front();
    m_data.push_back(VoxelData());
    VoxelData & voxel = m_data.back();
    voxel.m_scan_id = scan_id;
    m_needs_update = true;
}

template <int _N>
void Voxel<_N>::addNewPoint( const Eigen::Vector3d & new_point, const Eigen::VectorXd * new_semantic, const Eigen::Vector3d * new_color, const bool prior )
{
    if ( new_semantic != nullptr )
    {
        if ( new_semantic->rows() != _N) { ROS_WARN_STREAM("number of classes missmatch!"); return; }
        const Eigen::VectorXd new_semantic_normalized = *new_semantic / new_semantic->sum();
        //m_data.back().m_semantic.addProb(new_semantic_normalized);
        m_data.back().m_semantic.addLogProb(new_semantic_normalized.array().log());
        m_needs_update = true;
    }
    if ( new_color != nullptr && m_data.back().m_num_color < VoxelData::max_num_fused ){
        m_data.back().addColor(*new_color);
        //++m_data.back().m_num_color;
        //m_data.back().m_sum_color += (*new_color);
        m_needs_update = true;
    }
    if ( m_data.back().m_num >= VoxelData::max_num_fused ) return;
    m_data.back().addPoint(new_point);
    //++(m_data.back().m_num);
    //m_data.back().m_sum += new_point;
    if(!m_is_prior)
        m_is_prior = prior;
    m_needs_update = true;
}

template <int _N>
void Voxel<_N>::update()
{
    if ( ! m_needs_update ) return;
    m_needs_update = false;
    m_fused = VoxelData();
    for ( const VoxelData & voxel : m_data )
    {
        m_fused += voxel;
    }
    // mean etc:
    m_valid = false;
    if ( m_fused.m_num > 0 ) m_mean = m_fused.m_sum / m_fused.m_num;
    if ( m_fused.m_num_color > 0 ) m_mean_color = m_fused.m_sum_color / m_fused.m_num_color;

    if ( m_fused.m_num >= VoxelData::min_num_fused )
    {   //TODO: calculating normals could be made optional..
        const Eigen::Matrix3d cov = (m_fused.m_sum_squares / (m_fused.m_num-1));
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
        const auto eigen_vectors = eig.eigenvectors();
        const auto eigen_values = eig.eigenvalues().normalized();

        if ( (eigen_values.array() > 1e-6).template cast<int>().sum() >= 2 )
            m_valid = true;

        m_normal = eigen_vectors.col(0).normalized().template cast<double>();
        if (m_normal.dot(m_first_view_dir) > 0.0)
            m_normal *= -1.0;
        
        m_normal_eigenvalue = eig.eigenvalues()(0);
        m_normalized_eigenvalue = eigen_values(0);
    }
}

template class Voxel<SemanticsColorMap::NUM_CLASSES_INDOOR>;
template class Voxel<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>;

template <int _N>
typename SemanticFusor<_N>::Ptr SemanticFusor<_N>::Create (const double& side_length, const double& voxel_length, const int & window_size, const bool & remove_older, const bool & store_all , const int &exclude_class, const std::vector<int>& dyn_classes, const bool& no_ray_tracing)
{
    return std::make_shared<SemanticFusor>(side_length, voxel_length, window_size, remove_older, store_all, exclude_class, dyn_classes, no_ray_tracing);
}

template <int _N>
inline typename SemanticFusor<_N>::IndexType SemanticFusor<_N>::toCellIndex ( const Eigen::Vector3d & point, const double & voxel_scale, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels ) const
{
    return toCellIndexFromIndexVector ( toCellIndexVector( point, voxel_scale ), middleIndexVector, num_voxels );
//    const IndexVec3Type idx_vec = (point * voxel_scale).array().floor().cast<IndexType>(); // ((numcells>>1)<<1) // even
//    const IndexVec3Type p = idx_vec + middleIndexVector;
//    const IndexVec3Type num_cells_per_side ( 1, num_voxels, num_voxels * num_voxels);
//    return p.dot(num_cells_per_side);
}

template <int _N>
inline typename SemanticFusor<_N>::IndexVec3Type SemanticFusor<_N>::toCellIndexVector ( const Eigen::Vector3d & point, const double & voxel_scale ) const
{
    return (point * voxel_scale).array().floor().cast<IndexType>(); // ((numcells>>1)<<1) // even
}

template <int _N>
inline typename Eigen::Vector3d SemanticFusor<_N>::fromCellIndexToCenter(const IndexType& idx, const double & voxel_scale, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels) const
{
    return (fromCellIndexToIndexVector(idx, middleIndexVector, num_voxels).template cast<double>() * voxel_scale).array() + voxel_scale / 2;
}

template <int _N>
inline typename SemanticFusor<_N>::IndexType SemanticFusor<_N>::toCellIndexFromIndexVector ( const IndexVec3Type & idx_vec, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels ) const
{
  const IndexVec3Type p = idx_vec + middleIndexVector;
  const IndexVec3Type num_cells_per_side ( 1, num_voxels, num_voxels * num_voxels);
  return p.dot(num_cells_per_side);
}

template <int _N>
inline typename SemanticFusor<_N>::IndexVec3Type SemanticFusor<_N>::fromCellIndexToIndexVector( const IndexType & idx, const IndexVec3Type & middleIndexVector, const IndexType & num_voxels) const
{
    IndexType pm = idx;
    IndexVec3Type center;
    center(0) =  pm % num_voxels;
    pm /= num_voxels;
    center(1) = pm % num_voxels;
    pm /= num_voxels;
    center(2) = pm;
    return center-middleIndexVector;
}


template <int _N>
void SemanticFusor<_N>::bresenham3D(const IndexVec3Type & scan_origin_index, const std::unordered_set<IndexType> & endpoint_cells, std::unordered_set<IndexType> & new_unknown_cells, std::unordered_set<IndexType> & new_free_cells )
{
    constexpr int num_unknown = 10;
    const IndexType num_voxels_per_side = IndexType(m_side_length / m_voxel_length);
   // const double voxel_scale = num_voxels_per_side / m_side_length;
    static const IndexVec3Type middleIndexVector = getMiddleIndexVector ( num_voxels_per_side );

    for ( auto cellIt = endpoint_cells.begin(); cellIt != endpoint_cells.end(); ++cellIt )
    {
        // bresenham 3d: https://gist.github.com/yamamushi/5823518
        IndexVec3Type point(scan_origin_index[0],scan_origin_index[1],scan_origin_index[2]);
        const IndexVec3Type end_point = fromCellIndexToIndexVector(*cellIt, middleIndexVector, num_voxels_per_side);

        const int dx = end_point[0] - point[0];
        const int dy = end_point[1] - point[1];
        const int dz = end_point[2] - point[2];
        const int x_inc = (dx < 0) ? -1 : 1;
        const int y_inc = (dy < 0) ? -1 : 1;
        const int z_inc = (dz < 0) ? -1 : 1;
        const int l = std::abs(dx);
        const int m = std::abs(dy);
        const int n = std::abs(dz);
        const int dx2 = l << 1;
        const int dy2 = m << 1;
        const int dz2 = n << 1;
        if ((l >= m) && (l >= n)) {
            int err_1 = dy2 - l;
            int err_2 = dz2 - l;
            for (int i = 0; i < l; ++i) {
                //draw_free_cells[m_params.toCellIndexFromIndexVector ( point )] = point;
                //draw_free_cells.insert(m_params.toCellIndexFromIndexVector ( point ));
                new_free_cells.insert(toCellIndexFromIndexVector ( point, middleIndexVector, num_voxels_per_side ));

                if (err_1 > 0) {
                    point[1] += y_inc;
                    err_1 -= dx2;
                }
                if (err_2 > 0) {
                    point[2] += z_inc;
                    err_2 -= dx2;
                }
                err_1 += dy2;
                err_2 += dz2;
                point[0] += x_inc;
            }
            for (int i = 0; i < num_unknown; ++i) {
                //draw_unknown_cells[m_params.toCellIndexFromIndexVector ( point )] = point;
                //draw_unknown_cells.insert(m_params.toCellIndexFromIndexVector ( point ));
                new_unknown_cells.insert(toCellIndexFromIndexVector ( point, middleIndexVector, num_voxels_per_side ));

                if (err_1 > 0) {
                    point[1] += y_inc;
                    err_1 -= dx2;
                }
                if (err_2 > 0) {
                    point[2] += z_inc;
                    err_2 -= dx2;
                }
                err_1 += dy2;
                err_2 += dz2;
                point[0] += x_inc;
            }
        } else if ((m >= l) && (m >= n)) {
            int err_1 = dx2 - m;
            int err_2 = dz2 - m;
            for (int i = 0; i < m; ++i) {
                //draw_free_cells[m_params.toCellIndexFromIndexVector ( point )] = point;
                //draw_free_cells.insert(m_params.toCellIndexFromIndexVector ( point ));
                new_free_cells.insert(toCellIndexFromIndexVector ( point, middleIndexVector, num_voxels_per_side ));

                if (err_1 > 0) {
                    point[0] += x_inc;
                    err_1 -= dy2;
                }
                if (err_2 > 0) {
                    point[2] += z_inc;
                    err_2 -= dy2;
                }
                err_1 += dx2;
                err_2 += dz2;
                point[1] += y_inc;
            }
            for (int i = 0; i < num_unknown; ++i) {
                //draw_unknown_cells[m_params.toCellIndexFromIndexVector ( point )] = point;
                //draw_unknown_cells.insert(m_params.toCellIndexFromIndexVector ( point ));
                new_unknown_cells.insert(toCellIndexFromIndexVector ( point, middleIndexVector, num_voxels_per_side ));

                if (err_1 > 0) {
                    point[0] += x_inc;
                    err_1 -= dy2;
                }
                if (err_2 > 0) {
                    point[2] += z_inc;
                    err_2 -= dy2;
                }
                err_1 += dx2;
                err_2 += dz2;
                point[1] += y_inc;
            }
        } else {
            int err_1 = dy2 - n;
            int err_2 = dx2 - n;
            for (int i = 0; i < n; ++i) {
                //draw_free_cells[m_params.toCellIndexFromIndexVector ( point )] = point;
                //draw_free_cells.insert(m_params.toCellIndexFromIndexVector ( point ));
                new_free_cells.insert(toCellIndexFromIndexVector ( point, middleIndexVector, num_voxels_per_side ));

                if (err_1 > 0) {
                    point[1] += y_inc;
                    err_1 -= dz2;
                }
                if (err_2 > 0) {
                    point[0] += x_inc;
                    err_2 -= dz2;
                }
                err_1 += dy2;
                err_2 += dx2;
                point[2] += z_inc;
            }
            for (int i = 0; i < num_unknown; ++i) {
                //draw_unknown_cells[m_params.toCellIndexFromIndexVector ( point )] = point;
                //draw_unknown_cells.insert(m_params.toCellIndexFromIndexVector ( point ));
                new_unknown_cells.insert(toCellIndexFromIndexVector ( point, middleIndexVector, num_voxels_per_side ));

                if (err_1 > 0) {
                    point[1] += y_inc;
                    err_1 -= dz2;
                }
                if (err_2 > 0) {
                    point[0] += x_inc;
                    err_2 -= dz2;
                }
                err_1 += dy2;
                err_2 += dx2;
                point[2] += z_inc;
            }
        }
        //reached the original point, not necessary here, since we will set this one outside
    }
}

template <int _N>
void SemanticFusor<_N>::updateUpdatedIndices(const std::vector<IndexType>& updated_occupied_indices, const std::vector<IndexType>& updated_free_indices, const std::vector<IndexType>& updated_unknown_indices)
{
    // now add indices to the occupied / unknown / free ones
    std::vector<IndexType> reset_semantic_indices;
    reset_semantic_indices.reserve(updated_free_indices.size() / 10);
    {
        std::scoped_lock lock(m_voxels_mutex);
        for ( const IndexType & idx : updated_occupied_indices )
        {
            m_occupied_indices.insert(idx);
            auto uit = m_unknown_indices.find(idx);
            if ( uit != m_unknown_indices.end() ) m_unknown_indices.erase(uit);
            auto fit = m_free_indices.find(idx);
            if ( fit != m_free_indices.end() ) m_free_indices.erase(uit);
        }
        for ( const IndexType & idx : updated_unknown_indices )
        {
            m_unknown_indices.insert(idx);
            auto oit = m_occupied_indices.find(idx);
            if ( oit != m_occupied_indices.end() ){ m_occupied_indices.erase(oit); reset_semantic_indices.emplace_back(idx); } // reset VoxelData
            auto fit = m_free_indices.find(idx);
            if ( fit != m_free_indices.end() ) m_free_indices.erase(fit);
        }
        for ( const IndexType & idx : updated_free_indices )
        {
            m_free_indices.insert(idx);
            auto oit = m_occupied_indices.find(idx);
            if ( oit != m_occupied_indices.end() ){ m_occupied_indices.erase(oit); reset_semantic_indices.emplace_back(idx); } // reset VoxelData
            auto uit = m_unknown_indices.find(idx);
            if ( uit != m_unknown_indices.end() ) m_unknown_indices.erase(uit);
        }

        for ( const IndexType & idx : reset_semantic_indices ){
          const auto & voxel_ptr = m_voxels[idx];
          if(voxel_ptr){
            voxel_ptr->clear();
          }
        }
    }

    ROS_INFO_STREAM_THROTTLE(1, " oi: " << m_occupied_indices.size() << " fi: " << m_free_indices.size() << " ui: " << m_unknown_indices.size() << " reset: " << reset_semantic_indices.size());
}

template <int _N>
void SemanticFusor<_N>::addCloud ( const sensor_msgs::PointCloud2ConstPtr & cloud_msg, const Eigen::Affine3d & pose_ws, const bool prior )
{
    //ros::Time start_add_cloud = ros::Time::now();
    std::chrono::high_resolution_clock::time_point start_add_cloud = std::chrono::high_resolution_clock::now();
    if ( ! m_store_all )
        ++m_scan_id;
    uint32_t offset_x = 0;
    uint32_t offset_y = 0;
    uint32_t offset_z = 0;
    uint32_t offset_intensity = 0;
    uint32_t offset_rgb = 0;
    uint32_t offset_semantic = 0;
    bool has_rgb = false;
    int cnt_semantic = 0;
    for( size_t i = 0; i < cloud_msg->fields.size(); ++i )
    {
        if ( cloud_msg->fields[i].name=="x" ) offset_x = cloud_msg->fields[i].offset;
        if ( cloud_msg->fields[i].name=="y" ) offset_y = cloud_msg->fields[i].offset;
        if ( cloud_msg->fields[i].name=="z" ) offset_z = cloud_msg->fields[i].offset;
        if ( cloud_msg->fields[i].name=="intensity" ) offset_intensity = cloud_msg->fields[i].offset;
        if ( cloud_msg->fields[i].name=="rgb" ) { offset_rgb = cloud_msg->fields[i].offset; has_rgb = true; }
        if ( cloud_msg->fields[i].name=="semantic" ) {offset_semantic = cloud_msg->fields[i].offset; cnt_semantic = cloud_msg->fields[i].count;}
    }
    if ( ( has_rgb || m_dont_use_color ) && _N == SemanticsColorMap::NUM_CLASSES_COLOR_ONLY ) // fuse with rgb only !
    {
        cnt_semantic = _N;
    }
    if(!prior && cnt_semantic != _N){
      ROS_ERROR("incoherent number of semantic classes: expected %d, got %d, ABORTING!", _N, cnt_semantic);
      return;
    }
    const size_t num_points = cloud_msg->data.size() / cloud_msg->point_step;
    //const size_t semantic_length = _N * sizeof(float);

    const IndexType num_voxels_per_side = IndexType(m_side_length / m_voxel_length);
    const double voxel_scale = num_voxels_per_side / m_side_length;
    static const IndexVec3Type middleIndexVector = getMiddleIndexVector ( num_voxels_per_side );

    ROS_INFO_STREAM_THROTTLE(1,"side: " << m_side_length << " vxPS: " << num_voxels_per_side << " vs: " << voxel_scale << " mi: " << middleIndexVector.transpose());

    Eigen::VectorXd new_semantic;
    Eigen::Vector3d new_color;
    Eigen::VectorXd * new_semantic_ptr = nullptr;
    std::unordered_set<IndexType> new_occ_indices;

    if ( !m_no_ray_tracing )
      new_occ_indices.reserve(num_points);

    {
      std::scoped_lock lock (m_voxels_mutex);
      for ( size_t idx = 0; idx < num_points; ++idx )
      {
          const size_t point_offset = idx * cloud_msg->point_step;
          const Eigen::Map<const Eigen::Vector3f> point ( (const float*) &cloud_msg->data[point_offset+offset_x] );
          const Eigen::Vector3d new_point = pose_ws * point.cast<double>();
          Eigen::VectorXd::Index maxIndex = 0;

          if constexpr ( _N != SemanticsColorMap::NUM_CLASSES_COLOR_ONLY )
          {
            if(!prior && offset_semantic > 0){
              const Eigen::Map<const Eigen::VectorXf> new_semantic_map ( (const float*) &cloud_msg->data[point_offset+offset_semantic], _N, 1);
              new_semantic = new_semantic_map.cast<double>();
              new_semantic_ptr = &new_semantic;
              new_semantic.maxCoeff(&maxIndex);
            }
          }

          const IndexType voxel_idx = toCellIndex(new_point, voxel_scale, middleIndexVector, num_voxels_per_side);
          if ( ! m_voxels[voxel_idx] ) m_voxels[voxel_idx] = Voxel<_N>::Create(m_window_size, (pose_ws.translation()-new_point).normalized());

//          if(maxIndex != m_exclude_class){ // Don't add person points to cloud..
            const auto & voxel_ptr = m_voxels[voxel_idx];
            voxel_ptr->addNewScan( m_scan_id );
            voxel_ptr->addNewPoint( new_point, new_semantic_ptr, nullptr, prior );
//          }

          if ( !m_no_ray_tracing )
            new_occ_indices.insert(voxel_idx); // use point for raytracing independent of semantic class
      }

      if ( m_remove_older && ! m_store_all )
      {
          std::vector<IndexType> toBeRemoved; toBeRemoved.reserve(m_voxels.size());
          for ( const std::pair<const IndexType, typename Voxel<_N>::Ptr> & idx_voxel_ptr : m_voxels )
          {
              const typename Voxel<_N>::Ptr & voxel_ptr = idx_voxel_ptr.second;
              if ( ! voxel_ptr ) { continue; }
              Voxel<_N> & voxel = *voxel_ptr;
              voxel.remove_older ( m_scan_id-m_window_size );
              if ( voxel.isEmpty() )
                  toBeRemoved.emplace_back(idx_voxel_ptr.first);
          }
          ROS_INFO_STREAM_THROTTLE(1,"should remove now: " << toBeRemoved.size());
          for ( const IndexType & id : toBeRemoved )
              m_voxels.erase(id); // TODO this could erase previously added voxel in case of exclude_class...
      }
      ROS_INFO_STREAM_THROTTLE(1,"adding took: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_add_cloud).count() << " np: " << num_points << " now has: " << m_voxels.size());
    }

    if ( m_no_ray_tracing ) return;

    std::chrono::high_resolution_clock::time_point start_raytrace = std::chrono::high_resolution_clock::now();
    std::unordered_set<IndexType> new_free_indices, new_unknown_indices;
    const size_t num_new_occ = new_occ_indices.size();

    if(!prior){
      new_free_indices.reserve(30*num_new_occ);
      new_unknown_indices.reserve(num_new_occ);
      const IndexVec3Type scan_origin_index = toCellIndexVector(pose_ws.translation(), voxel_scale);
      bresenham3D( scan_origin_index, new_occ_indices, new_unknown_indices, new_free_indices  );
      std::chrono::high_resolution_clock::time_point end_raytrace = std::chrono::high_resolution_clock::now();
      ROS_INFO_STREAM_THROTTLE(1, "bresenham3D took: " << std::chrono::duration<double>(end_raytrace - start_raytrace).count() << " new_occ: " << num_new_occ << ", new_free: " << new_free_indices.size() << ", new_unk: " << new_unknown_indices.size());

      // cleanup: remove measured ones
      for ( auto cellIt = new_occ_indices.begin(); cellIt != new_occ_indices.end(); ++cellIt )
      {
          new_free_indices.erase(*cellIt);
          new_unknown_indices.erase(*cellIt);
      }
      ROS_INFO_STREAM_THROTTLE(1, "cleanup: remove occ: new_unk: " << new_unknown_indices.size() << " new_free: " << new_free_indices.size());

      // remove free ones from unknown
      for ( auto cellIt = new_free_indices.begin(); cellIt != new_free_indices.end(); ++cellIt )
      {
          new_unknown_indices.erase(*cellIt);
      }
      ROS_INFO_STREAM_THROTTLE(1, "cleanup: remove free: new_unk: " << new_unknown_indices.size() << " new_free: " << new_free_indices.size());

      std::chrono::high_resolution_clock::time_point end_cleanup = std::chrono::high_resolution_clock::now();
      ROS_INFO_STREAM_THROTTLE(1, "cleanup took: " << std::chrono::duration<double>(end_cleanup - end_raytrace).count());
    }

    std::chrono::high_resolution_clock::time_point start_update = std::chrono::high_resolution_clock::now();
    const Eigen::Vector3d origin = pose_ws.translation();
    constexpr float obscure_angle_threshold = 0.3f;

    size_t new_idx_cnt = new_occ_indices.size() + new_free_indices.size() + new_unknown_indices.size();
    std::vector<IndexType> updated_occupied_indices, updated_free_indices, updated_unknown_indices;
    updated_occupied_indices.reserve(std::max(num_new_occ, new_idx_cnt / 5));
    updated_unknown_indices.reserve(new_idx_cnt / 5);
    updated_free_indices.reserve(new_idx_cnt);
    int num_occ = 0, num_free = 0, num_unk = 0;

    std::chrono::high_resolution_clock::time_point mutex_time;
    {
      std::scoped_lock lock (m_voxels_mutex);
      mutex_time = std::chrono::high_resolution_clock::now();
      for ( auto cellIt = new_occ_indices.begin(); cellIt != new_occ_indices.end(); ++cellIt)
      {
          const IndexType idx = *cellIt;
          if ( ! m_voxels[idx] ) m_voxels[idx] = Voxel<_N>::Create(m_window_size, (origin - fromCellIndexToCenter(idx, voxel_scale, middleIndexVector, num_voxels_per_side)).normalized()); // TODO: Init check should not be necessary, as already handled above.
          const auto & voxel_ptr = m_voxels[idx];
          voxel_ptr->push_occ(Voxel<_N>::occupied);
          voxel_ptr->resetObscureAngleCnt();

          const int8_t state = voxel_ptr->count();
          switch(state){
            case Voxel<_N>::free : ++num_free; updated_free_indices.emplace_back(idx); break;
            case Voxel<_N>::occupied : ++num_occ; updated_occupied_indices.emplace_back(idx); break;
            default:
            case Voxel<_N>::unknown : ++num_unk; updated_unknown_indices.emplace_back(idx); break;
          }
      }
      for ( auto cellIt = new_free_indices.begin(); cellIt != new_free_indices.end(); ++cellIt)
      {
          const IndexType idx = *cellIt;
          Eigen::Vector3d view_direction = (origin - fromCellIndexToCenter(idx, voxel_scale, middleIndexVector, num_voxels_per_side)).normalized();
          if ( ! m_voxels[idx] ) m_voxels[idx] = Voxel<_N>::Create(m_window_size, view_direction);

          const auto & voxel_ptr = m_voxels[idx];
          voxel_ptr->update(); // update to make sure normals are up to date..
          if ( (voxel_ptr->isPrior() || voxel_ptr->getObscureAngleCnt() < Voxel<_N>::Nocc) && voxel_ptr->isValid() && std::abs(view_direction.dot(voxel_ptr->getNormal().normalized())) < obscure_angle_threshold ){
              voxel_ptr->incrObscureAngleCnt();
              continue;
          }

          voxel_ptr->push_occ(Voxel<_N>::free);

          const int8_t state = voxel_ptr->count();
          switch(state){
            case Voxel<_N>::free : ++num_free; updated_free_indices.emplace_back(idx); break;
            case Voxel<_N>::occupied : ++num_occ; updated_occupied_indices.emplace_back(idx); break;
            default:
            case Voxel<_N>::unknown : ++num_unk; updated_unknown_indices.emplace_back(idx); break;
          }
      }
      for ( auto cellIt = new_unknown_indices.begin(); cellIt != new_unknown_indices.end(); ++cellIt)
      {
          const IndexType idx = *cellIt;
          if ( ! m_voxels[idx] ) m_voxels[idx] = Voxel<_N>::Create(m_window_size, (origin - fromCellIndexToCenter(idx, voxel_scale, middleIndexVector, num_voxels_per_side)).normalized());
          const auto & voxel_ptr = m_voxels[idx];
          if(voxel_ptr->isPrior())
              voxel_ptr->push_occ(Voxel<_N>::occupied);
          else
              voxel_ptr->push_occ(Voxel<_N>::unknown);
          voxel_ptr->resetObscureAngleCnt();

          const int8_t state = voxel_ptr->count();
          switch(state){
            case Voxel<_N>::free : ++num_free; updated_free_indices.emplace_back(idx); break;
            case Voxel<_N>::occupied : ++num_occ; updated_occupied_indices.emplace_back(idx); break;
            default:
            case Voxel<_N>::unknown : ++num_unk; updated_unknown_indices.emplace_back(idx); break;
          }
      }
    }

    ROS_INFO_STREAM_THROTTLE(1, "#o: " << num_occ << " #f: " << num_free << " #u: "<< num_unk);
    std::chrono::high_resolution_clock::time_point occ_update = std::chrono::high_resolution_clock::now();

    updateUpdatedIndices(updated_occupied_indices, updated_free_indices, updated_unknown_indices);
    std::chrono::high_resolution_clock::time_point end_update = std::chrono::high_resolution_clock::now();
    ROS_INFO_STREAM_THROTTLE(1, "update took: " << std::chrono::duration<double>(end_update - start_update).count() << "thereof: acquire mutex: " << std::chrono::duration<double>(mutex_time - start_update).count()
                             << ", occ. update: " << std::chrono::duration<double>(occ_update - mutex_time).count() << ", idx update: " << std::chrono::duration<double>(end_update - occ_update).count());
}

template <int _N>
sensor_msgs::PointCloud2Ptr SemanticFusor<_N>::getFusedVoxel(jetson_semantic_map::MappOccIdxPtr & map_occ_idx_msg, const std::string &fixed_frame, const bool use_semantic_color )
{
    std::chrono::high_resolution_clock::time_point fused_start = std::chrono::high_resolution_clock::now();
    
    static SemanticsColorMap cmap(_N);

    sensor_msgs::PointCloud2Ptr msg ( new sensor_msgs::PointCloud2() );
    if(map_occ_idx_msg){
      map_occ_idx_msg->header.frame_id = fixed_frame;
      map_occ_idx_msg->side_length = m_side_length;
      map_occ_idx_msg->voxel_length = m_voxel_length;
    }

    // add fields
    constexpr int num_fields_n = 8;
    constexpr int num_fields_o = 4;
    const bool with_normals = m_config.publish_normal;
    const int num_fields = with_normals ? num_fields_n : num_fields_o;
    msg->fields.resize(num_fields); // x,y,z,rgb
    msg->fields[0].name = "x";
    msg->fields[1].name = "y";
    msg->fields[2].name = "z";
    msg->fields[3].name = "rgb";
    if ( with_normals )
    {
        msg->fields[4].name = "normal_x";
        msg->fields[5].name = "normal_y";
        msg->fields[6].name = "normal_z";
        msg->fields[7].name = "sev"; // smallest eigenvalue
    }
    for ( size_t i = 0; i < num_fields; ++i )
    {
        msg->fields[i].datatype = sensor_msgs::PointField::FLOAT32;
        msg->fields[i].count = 1;
        msg->fields[i].offset = i*sizeof(float);
    }

    const bool with_semantic = m_config.publish_semantic;
    if(with_semantic){
      sensor_msgs::PointField field_semantic;
      field_semantic.name = "semantic";
      field_semantic.datatype = sensor_msgs::PointField::FLOAT32;
      field_semantic.count = _N;
      field_semantic.offset = num_fields*sizeof(float);
      msg->fields.push_back(field_semantic);
    }

    const int min_num_points = m_config.min_num_points_for_vis;
    const int min_num_points_dynamic = m_config.min_num_points_for_vis_dynamic;
    const float min_x = m_config.x_min;
    const float max_x = m_config.x_max;
    const float min_y = m_config.y_min;
    const float max_y = m_config.y_max;
    const float min_z = m_config.z_min;
    const float max_z = m_config.z_max;
    const bool use_normalized_eigenvalue = m_config.use_normalized_eigenvalue;
    size_t idx = 0, num_nullptrs = 0, num_toofew = 0;

    size_t num_points;
    const size_t point_step = with_semantic ? (num_fields + _N) * sizeof(float) : num_fields*sizeof(float);
    const size_t offset_rgb = msg->fields[3].offset;
    const size_t offset_normal = with_normals ? msg->fields[4].offset : 0;
    const size_t offset_sev = with_normals ? msg->fields[7].offset : 0;
    const size_t offset_semantic = with_semantic ? msg->fields.back().offset : 0;
    msg->header.frame_id = fixed_frame;
    msg->is_dense = false;

    {
      std::scoped_lock lock(m_voxels_mutex);
      num_points = m_no_ray_tracing ? m_voxels.size() : m_occupied_indices.size();
      if(map_occ_idx_msg){
        map_occ_idx_msg->occ_indices.reserve(num_points);
      }

      msg->data.resize(num_points * point_step, 0);
      msg->width = num_points;
      msg->height = 1;
      msg->point_step = point_step;
      msg->row_step = point_step * msg->width;

      if(m_no_ray_tracing){
        for ( const std::pair<const IndexType, typename Voxel<_N>::Ptr> & idx_voxel_ptr : m_voxels ){
            const typename Voxel<_N>::Ptr & voxel_ptr = idx_voxel_ptr.second;
            if ( ! voxel_ptr ) { ++idx; ++num_nullptrs; continue; }

            switch(addVoxel2Cloud(voxel_ptr, msg, idx,
                                  point_step, offset_rgb, offset_normal, offset_sev, offset_semantic,
                                  min_num_points, min_num_points_dynamic, min_x, max_x, min_y, max_y, min_z, max_z, cmap,
                                  use_normalized_eigenvalue, use_semantic_color, with_normals, with_semantic))
            {
              case 0: // point successfully added to output
                if(map_occ_idx_msg)
                  map_occ_idx_msg->occ_indices.emplace_back(idx_voxel_ptr.first);
                break;
              case -1: // too few observations
                ++num_toofew;
                break;
              default: // other error, point not added..
                break;
            }

            ++idx;
        }
      }

      else {
        for(const auto& cellIdx : m_occupied_indices) // generate output map-cloud only from occupied voxels
        {
            const typename Voxel<_N>::Ptr & voxel_ptr = m_voxels[cellIdx];
            if ( ! voxel_ptr ) { ++idx; ++num_nullptrs; continue; }

            switch(addVoxel2Cloud(voxel_ptr, msg, idx,
                                  point_step, offset_rgb, offset_normal, offset_sev, offset_semantic,
                                  min_num_points, min_num_points_dynamic, min_x, max_x, min_y, max_y, min_z, max_z, cmap,
                                  use_normalized_eigenvalue, use_semantic_color, with_normals, with_semantic))
            {
              case 0: // point successfully added to output
                if(map_occ_idx_msg)
                  map_occ_idx_msg->occ_indices.emplace_back(cellIdx);
                break;
              case -1: // too few observations
                ++num_toofew;
                break;
              default: // other error, point not added..
                break;
            }

            ++idx;
        }
      }
    }

    ROS_INFO_STREAM_THROTTLE(1,"getting fused took: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - fused_start).count() << " for " << num_points << " with nulls: " << num_nullptrs << " too few: " << num_toofew);
    return msg;
}

template <int _N>
int SemanticFusor<_N>::addVoxel2Cloud(const typename Voxel<_N>::Ptr & voxel_ptr, sensor_msgs::PointCloud2Ptr & msg, const size_t & idx,
                   const size_t & point_step, const size_t offset_rgb, const size_t offset_normal, const size_t offset_sev, const size_t offset_semantic,
                                      const int & min_num_points, const int & min_num_points_dynamic, const float &min_x, const float &max_x, const float &min_y, const float &max_y, const float & min_z, const float & max_z, const SemanticsColorMap & cmap,
                                      const bool use_normalized_eigenvalue, const bool use_semantic_color, const bool with_normals, const bool with_semantic)
{
    Voxel<_N> & voxel = *voxel_ptr;
    voxel.update();
    const int semantic_class = voxel.getArgMaxClass();

    if(semantic_class == m_exclude_class)
      return -2;

    bool is_dynamic_class = false;
    for(const auto& class_idx : m_dynamic_classes){
      if(semantic_class == class_idx){
        is_dynamic_class = true;
        break;
      }
    }

    if ( ((is_dynamic_class && voxel.getNum() < min_num_points_dynamic) || (!is_dynamic_class && voxel.getNum() < min_num_points)) || ( with_normals && !voxel.isValid() ) ) { return -1; }

    const size_t point_offset = idx * point_step;
    const Eigen::Vector3d pos = voxel.getMean();

    if ( pos.x() < min_x || pos.x() > max_x || pos.y() < min_y || pos.y() > max_y || pos.z() < min_z || pos.z() > max_z )  { return -2; }

    Eigen::Map<Eigen::Vector3f>((float*)&msg->data[point_offset]) = pos.cast<float>();

    Eigen::Matrix<uint8_t, 4,1> pt_rgb = Eigen::Matrix<uint8_t,4,1>::Zero();
    if ( ! use_semantic_color )
    {
        pt_rgb.head<3>() = (hsv2rgb(voxel.getMeanColor()) * 255).template cast<int>().cwiseMin(255).cwiseMax(0).template cast<uint8_t>();
    }
    else
    {
        if ( semantic_class >= 0 && semantic_class < _N )
            pt_rgb.head<3>() = cmap.colormap.row(semantic_class).transpose().reverse();
        else
            pt_rgb.setZero();
    }
    Eigen::Map<Eigen::Matrix<uint8_t, 4,1> > ( (uint8_t*) &msg->data[point_offset+offset_rgb], 4, 1) = pt_rgb; //bgra

    if ( with_normals )
    {
        const Eigen::Vector3d normal = voxel.getNormal();
        Eigen::Map<Eigen::Vector3f>((float*)&msg->data[point_offset+offset_normal]) = normal.cast<float>();
        *reinterpret_cast<float*>(&msg->data[point_offset+offset_sev]) = voxel.getSmallestEigenvalue( use_normalized_eigenvalue );
    }

    if(with_semantic){
      const Eigen::Matrix<float, _N, 1> semantic = voxel.getSemantic().template cast<float>();
      memcpy(&msg->data[point_offset + offset_semantic], semantic.data(), _N * sizeof(float));
    }

    return 0;
}

template class SemanticFusor<SemanticsColorMap::NUM_CLASSES_INDOOR>;
template class SemanticFusor<SemanticsColorMap::NUM_CLASSES_COLOR_ONLY>;
