#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <math.h>

#include "callback.h"
#include "sensorParams.h"

/*
* Process a single image and return a vector of observations.  Observations are
* defined by the struct BalloonInfo. Look how BalloonInfo is defined in
* include/callback.h
*/
const std::vector<BalloonInfo> processImage(const cv::Mat& img) {
    /* Sensor params in: sensorParams */

    cv::Scalar redLower(150,100,100);
    cv::Scalar redUpper(200,255,255);
    cv::Scalar blueLower(100,100,100);
    cv::Scalar blueUpper(140,255,255); 
    double tol = .5;

    cv::Mat blurred,hsv,maskB,maskErodedB,maskDilatedB,maskR,maskR1,maskR2,maskErodedR,maskDilatedR;

    cv::GaussianBlur(img,blurred,cv::Size(11,11),0);
    cv::cvtColor(blurred,hsv,cv::COLOR_BGR2HSV);

    //BLUE
    cv::inRange(hsv,blueLower,blueUpper,maskB);
    cv::erode(maskB,maskErodedB,cv::noArray(),cv::Point(-1,-1),1);
    cv::dilate(maskErodedB,maskDilatedB,cv::noArray(),cv::Point(-1,-1),1);

    std::vector<std::vector<cv::Point>> contoursB;
    cv::findContours(maskDilatedB,contoursB,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    int radiusThreshold = 30;
    std::vector<std::vector<cv::Point>> candidateContoursB;
    for(int i=0;i<contoursB.size();i++){
      cv::Point2f center;
      float radius;
      cv::minEnclosingCircle(contoursB[i],center,radius);
      if(radius>radiusThreshold){
        candidateContoursB.push_back(contoursB[i]);
      }
    }

    //RED
    cv::inRange(hsv,redLower,redUpper,maskR1);
    cv::inRange(hsv,cv::Scalar(0,100,100),cv::Scalar(20,255,255),maskR2);
    maskR = maskR1|maskR2;

    cv::erode(maskR,maskErodedR,cv::noArray(),cv::Point(-1,-1),1);
    cv::dilate(maskErodedR,maskDilatedR,cv::noArray(),cv::Point(-1,-1),1);

    std::vector<std::vector<cv::Point>> contoursR;
    cv::findContours(maskDilatedR,contoursR,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

    radiusThreshold = 30;
    std::vector<std::vector<cv::Point>> candidateContoursR;
    for(int i=0;i<contoursR.size();i++){
      cv::Point2f center;
      float radius;
      cv::minEnclosingCircle(contoursR[i],center,radius);
      if(radius>radiusThreshold){
        candidateContoursR.push_back(contoursR[i]);
      }
    }

    //Return the circle locations
    std::vector<BalloonInfo> balloons;
    //BLUE
    if(candidateContoursB.size()>0){
      
      double maxArea = 0;
      std::vector<cv::Point> contB;
      for(int i=0;i<candidateContoursB.size();i++){
        if(cv::contourArea(candidateContoursB[i])>maxArea){
          contB = candidateContoursB[i];
          maxArea=cv::contourArea(contB);
        }
      }
      cv::Point2f center;
      float radius;
      cv::minEnclosingCircle(contB,center,radius);
      double areaCirc = 3.14159*radius*radius;
      if(areaCirc*(1-tol)<maxArea and areaCirc*(1+tol)>maxArea){
        BalloonInfo infoB;
        infoB.balloonLocation = Eigen::Vector3d(center.x*1.12e-6,center.y*1.12e-6,0);
        infoB.balloonRadius = radius;
        infoB.color = blue;
        balloons.push_back(infoB);
        // std::cout<<"Found blue contours"<<std::endl;
      }
    }

    //RED
    if(candidateContoursR.size()>0){
      
      double maxArea = 0;
      std::vector<cv::Point> contR;
      for(int i=0;i<candidateContoursR.size();i++){
        if(cv::contourArea(candidateContoursR[i])>maxArea){
          contR = candidateContoursR[i];
          maxArea=cv::contourArea(contR);
        }
      }
      cv::Point2f center;
      float radius;
      cv::minEnclosingCircle(contR,center,radius);
      double areaCirc = 3.14159*radius*radius;
      if(areaCirc*(1-tol)<maxArea and areaCirc*(1+tol)>maxArea){
        BalloonInfo infoR;
        infoR.balloonLocation = Eigen::Vector3d(center.x,center.y,0);
        infoR.balloonRadius = radius;
        infoR.color = red;
        balloons.push_back(infoR);
        //std::cout<<"Found red contours"<<std::endl;
      }
    }
    return balloons;
}

/*
* Take a database of image observations and estimate the 3D position of the balloons
*/
const std::map<Color, Eigen::Vector3d> estimatePosition(const std::vector<Observation>& database) {
    //First split up into red and blue observations
    std::vector<Observation> databaseRed, databaseBlue;
    for(int i=0;i<database.size();i++){
        if(database[i].info_vec[0].color==red){
            databaseRed.push_back(database[i]);
        }else{
            databaseBlue.push_back(database[i]);
        }
    }

    Eigen::Vector3d locRed = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d locBlue = Eigen::Vector3d(0, 0, 0);
    //RED FIRST
    int N = databaseRed.size();
    if(N>1){
        //Initialize stuff
        Eigen::MatrixXd H =Eigen::MatrixXd::Zero(2*N,4);
        Eigen::MatrixXd R =Eigen::MatrixXd::Zero(2*N,2*N);

        Eigen::Matrix<double,3,4> K =Eigen::MatrixXd::Zero(3,4);
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                K(i,j)=sensorParams.camera_matrix.at<double>(i,j);
            }
        } 
        // Build up H matrix, R matrix
        for(int i=0;i<databaseRed.size();i++){
            //Convert quad_att to rotation matrix
            Eigen::Matrix3d RBI = getRotationMat(2,databaseBlue[i].quad_att(1))*getRotationMat(3,databaseBlue[i].quad_att(0));
            Eigen::Matrix3d RCB = getRotationMat(2,-sensorParams.camera_angle);
            Eigen::Matrix3d RCI = RCB*RBI;
            //Obtain position of camera center
            Eigen::Vector3d rcI = Recef2enu(databaseBlue[i].quad_pos)*(databaseBlue[i].quad_pos - sensorParams.riG) + RBI.transpose()*sensorParams.rcB;
            //Build up T matrix
            Eigen::Matrix<double,4,4> T =Eigen::MatrixXd::Zero(4,4);
            T.block<3,3>(0,0) = RCI;
            T.block<3,1>(0,3) = RCI*rcI;
            T(3,3) = 1;
            //Generate P matrix
            Eigen::Matrix<double,3,4> P = K*T;
            //Fill H matrix
            H.row(i*2) = databaseRed[i].info_vec[0].balloonLocation(0)*1.12e-6*P.row(2)-P.row(0);
            H.row(i*2+1) = databaseRed[i].info_vec[0].balloonLocation(1)*1.12e-6*P.row(2)-P.row(1);
            //Fill covariance matrix
            R.block<2,2>(i*2,i*2) = 20*20*1.12e-6*1.12e-6*Eigen::Matrix2d::Identity();
        }
        Eigen::MatrixXd Hr = H.block(0,0,2*N,3);
        Eigen::VectorXd z = -H.col(3);
        Eigen::MatrixXd Rinv = R.inverse();
        Eigen::MatrixXd Re = (Hr.transpose()*Rinv*Hr).inverse();
        locRed = Re*Hr.transpose()*Rinv*z;
    }
    //NOW BLUE
    N = databaseBlue.size();
    if(N>1){
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2*N,4);
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(2*N,2*N);

        Eigen::Matrix<double,3,4> K = Eigen::MatrixXd::Zero(3,4);
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                K(i,j)=sensorParams.camera_matrix.at<double>(i,j);
            }
        } 
        // Build up H matrix, R matrix
        for(int i=0;i<databaseBlue.size();i++){
            //Convert quad_att to rotation mat
            Eigen::Matrix3d RBI = getRotationMat(2,databaseBlue[i].quad_att(1))*getRotationMat(3,databaseBlue[i].quad_att(0));
            Eigen::Matrix3d RCB = getRotationMat(2,-sensorParams.camera_angle);
            Eigen::Matrix3d RCI = RCB*RBI;
            //Obtain position of camera center
            Eigen::Vector3d rcI = Recef2enu(databaseBlue[i].quad_pos)*(databaseBlue[i].quad_pos - sensorParams.riG) + RBI.transpose()*sensorParams.rcB;
            //Build up T matrix
            Eigen::Matrix<double,4,4> T = Eigen::MatrixXd::Zero(4,4);
            T.block<3,3>(0,0) = RCI;
            T.block<3,1>(0,3) = RCI*rcI;
            T(3,3) = 1;
            //Generate P matrix
            Eigen::Matrix<double,3,4> P = K*T;
            //Fill H matrix
            H.row(i*2) = databaseBlue[i].info_vec[0].balloonLocation(0)*1.12e-6*P.row(2)-P.row(0);
            H.row(i*2+1) = databaseBlue[i].info_vec[0].balloonLocation(1)*1.12e-6*P.row(2)-P.row(1);
            //Fill covariance matrix
            R.block<2,2>(i*2,i*2) = 20*20*1.12e-6*1.12e-6*Eigen::Matrix2d::Identity();
        }
        Eigen::MatrixXd Hr = H.block(0,0,2*N,3);
        Eigen::VectorXd z = -H.col(3);
        Eigen::MatrixXd Rinv = R.inverse();
        Eigen::MatrixXd Re = (Hr.transpose()*Rinv*Hr).inverse();
        locBlue = Re*Hr.transpose()*Rinv*z;
    }
    
    
    std::map<Color, Eigen::Vector3d> balloon_positions;
    balloon_positions[red] = locRed;
    balloon_positions[blue] = locBlue;
    return balloon_positions;
}
const Eigen::Matrix3d getRotationMat(const double axis,const double angle){
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if(axis == 1){
        R << 1,0,0,0,cos(angle),sin(angle),0,-sin(angle),cos(angle);
    }else if(axis==2){
        R << cos(angle),0,-sin(angle),0,1,0,sin(angle),0,cos(angle);
    }else if(axis==3){
        R << cos(angle),sin(angle),0,-sin(angle),cos(angle),0,0,0,1;
    }
    return R;
}
const Eigen::Matrix3d Recef2enu(const Eigen::Vector3d& r){
    double f = 1/298.257223563;
    double a = 6378137;
    double b = 6356752;
    double e = sqrt(f*(2-f));
    double ep = sqrt(e*e/(1-e*e));
    double rho = r.norm();
    double theta = atan2(a*r(2),rho*b);
    double lon = atan2(r(1),r(0));
    double lat = atan2(r(2)+ep*ep*b*pow(sin(theta),3),rho-e*e*a*pow(cos(theta),3));
    double N = a/sqrt(1-e*e*sin(lat)*sin(lat));
    double alt = rho/cos(lat)-N;
    double x = rho*sin(90-lat)*cos(lon);
    double y = rho*sin(90-lat)*sin(lon);
    double z = rho*cos(90-lat);
    Eigen::Vector3d v_vert; v_vert << x,y,z;
    Eigen::Vector3d v_east; v_east << -y,x,0;v_east.normalize();
    Eigen::Vector3d v_north = v_vert.cross(v_east); v_north.normalize();

    Eigen::Matrix3d R;
    R.row(0) = v_east.transpose();
    R.row(1) = v_north.transpose();
    R.row(2) = v_vert.transpose();
    return R;
}
