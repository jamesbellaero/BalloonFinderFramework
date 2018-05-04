#include <opencv2/opencv.hpp>
#include <vector>

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
        infoB.balloonLocation = Eigen::Vector3d(center.x*1.12e-6,center.y*1.12e-6,sensorParams.f);
        infoB.balloonRadius = radius;
        infoB.color = blue;
        balloons.push_back(infoB);
        std::cout<<"Found blue contours"<<std::endl;
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
        std::cout<<"Found red contours"<<std::endl;
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
        if(database[i].color==red){
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
        Eigen::Matrix<double,2*N,4> H = MatrixXd::Zero(2*N,4);
        Eigen::Matrix<double,2*N,2*N> R = MatrixXd::Zero(2*N,2*N);

        Eigen::Matrix<double,3,4> K = MatrixXd::Zero(3,4);
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                K(i,j)=sensorParams.camera_matrix(i,j);
            }
        } 
        // Build up H matrix, R matrix
        for(int i=0;i<databaseRed.size();i++){
            //Convert quad_att to rotation matrix
            Eigen::Matrix3d RCI = Matrix3d.Identity();
            //Obtain position of camera center
            Eigen::Vector3d rcI = databaseRed[i].quadPos;
            //Build up T matrix
            Eigen::Matrix<double,4,4> T = MatrixXd::Zero(4,4);
            T.block<3,3>(0,0) = RCI;
            T.block<3,1>(0,3) = RCI*rcI;
            T(3,3) = 1;
            //Generate P matrix
            Eigen::Matrix<double,3,4> P = K*T;
            //Fill H matrix
            H.row(i*2) = databaseRed[i].info_vec.balloonLocation(0)*1.12e-6*P.row(2)-P.row(0);
            H.row(i*2+1) = databaseRed[i].info_vec.balloonLocation(1)*1.12e-6*P.row(2)-P.row(1);
            //Fill covariance matrix
            R.block<2,2>(i*2,i*2) = 20*20*1.12e-6*1.12e-6*Eigen::Matrix2d.Identity();
        }
        Eigen::MatrixXd Hr = H.block<2*N,3>(0,0);
        Eigen::VectorXd z = -H.col(3);
        Eigen::MatrixXd Rinv = R.inverse();
        Eigen::MatrixXd Re = (Hr.transpose()*Rinv*Hr).inverse();
        locRed = Re*Hr.transpose()*Rinv*z;
    }
    //NOW BLUE
    N = databaseBlue.size();
    if(N>1){
        Eigen::Matrix<double,2*N,4> H = MatrixXd::Zero(2*N,4);
        Eigen::Matrix<double,2*N,2*N> R = MatrixXd::Zero(2*N,2*N);

        Eigen::Matrix<double,3,4> K = MatrixXd::Zero(3,4);
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                K(i,j)=sensorParams.camera_matrix(i,j);
            }
        } 
        // Build up H matrix, R matrix
        for(int i=0;i<databaseBlue.size();i++){
            //Convert quad_att to rotation matrix
            Eigen::Matrix3d RCI = Matrix3d.Identity();
            //Obtain position of camera center
            Eigen::Vector3d rcI = databaseBlue[i].quadPos;
            //Build up T matrix
            Eigen::Matrix<double,4,4> T = MatrixXd::Zero(4,4);
            T.block<3,3>(0,0) = RCI;
            T.block<3,1>(0,3) = RCI*rcI;
            T(3,3) = 1;
            //Generate P matrix
            Eigen::Matrix<double,3,4> P = K*T;
            //Fill H matrix
            H.row(i*2) = databaseBlue[i].info_vec.balloonLocation(0)*1.12e-6*P.row(2)-P.row(0);
            H.row(i*2+1) = databaseBlue[i].info_vec.balloonLocation(1)*1.12e-6*P.row(2)-P.row(1);
            //Fill covariance matrix
            R.block<2,2>(i*2,i*2) = 20*20*1.12e-6*1.12e-6*Eigen::Matrix2d.Identity();
        }
        Eigen::MatrixXd Hr = H.block<2*N,3>(0,0);
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
