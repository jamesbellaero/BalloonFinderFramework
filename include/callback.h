#pragma once

#include <cv.h>
#include <Eigen/Core>
#include <vector>
#include <map>

enum Color {red=0, blue=1};

typedef struct {
    /* X, Y, Z */
    Eigen::Vector3d balloonLocation;

    double balloonRadius;

    /* Red=0, Blue=1 */
    Color color;
} BalloonInfo;

typedef struct {
    std::vector<BalloonInfo> info_vec;

    /*
    * Position of the primary antenna in ECEF
    * (x, y, z)
    */
    Eigen::Vector3d quad_pos;

    /*
    * Covariance of the primary antenna's position in ECEF
    */
    Eigen::Matrix3d quad_pos_cov;

    /* 
    * Attitude specified by (Elevation, Azimuth, and Roll) 
    * Elevation - Angle in radians with respect to horizon. Positive is above horizon.
    * Azimuth   - Angle in radians with respect to true north. Positive is clockwise (towards East)
    * Roll      - Standard roll. Assumed to be zero. No true measurement for roll.
    */
    Eigen::Vector3d quad_att;

    /*
    * Covariance of the quad's attitude. 
    * Angles are assumed independent.
    * Roll is assumed to be 0.030461742 = 10 degrees, 1 sigma
    */
    Eigen::Matrix3d quad_att_cov;

} Observation;

const std::vector<BalloonInfo> processImage(const cv::Mat& img);
const std::map<Color, Eigen::Vector3d> estimatePosition(const std::vector<Observation>& database);
const Eigen::Matrix3d getRotationMat(const double axis,const double angle);
const Eigen::Matrix3d Recef2enu(const Eigen::Vector3d& r);