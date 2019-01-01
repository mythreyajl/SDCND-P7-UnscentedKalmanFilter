#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
    n_x_ = 5;
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.75;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values
   */
    n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  n_x_ = 5;
  weights_ = VectorXd(2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  is_initialized_ = false;
  n_radar_ = 3;
  n_lidar_ = 2;
  NIS_lidar_ = 0.0;
  NIS_radar_ = 0.0;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /*****************************************************************************
     *  Initialize
     ****************************************************************************/
    if(!is_initialized_) { // Initialize state for first measurement
        // Initialize state
        x_ << 1, 1, 0.2, 1, 0.1;
        
        // Initialize covariance
        P_ << 0.15, 0, 0, 0, 0,
              0, 0.15, 0, 0, 0,
              0,    0, 1, 0, 0,
              0,    0, 0, 1, 0,
              0,    0, 0, 0, 1;
        
        // Initialize state according to measurements first encountered
        if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER && use_laser_) {
            x_(0) = meas_package.raw_measurements_[0];
            x_(1) = meas_package.raw_measurements_[1];
        } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR && use_radar_) {
            const float& rho = meas_package.raw_measurements_[0];
            const float& theta = meas_package.raw_measurements_[1];
            x_(0) = rho * cos(theta);
            x_(1) = rho * sin(theta);
        }
        
        // Update state of measurements
        previous_timestamp_ = meas_package.timestamp_;
        is_initialized_ = true;
        
        return;
    }
    
    /*****************************************************************************
     *  Set timestamps
     ****************************************************************************/
    float dt = (meas_package.timestamp_ - previous_timestamp_)/1000000.0;
    previous_timestamp_ = meas_package.timestamp_;
    
    /*****************************************************************************
     *  Predict state and sigma points
     ****************************************************************************/
    Prediction(dt);
    
    /*****************************************************************************
     *  Update state according to measurements
     ****************************************************************************/
    if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER && use_laser_) {
        UpdateLidar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR && use_radar_) {
        UpdateRadar(meas_package);
    }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
    VectorXd x_aug = VectorXd(n_aug_);
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    // create augmented mean state
    x_aug.block(0, 0, n_x_, 1) = x_;
    x_aug(n_x_) = 0.0;
    x_aug(n_x_+1) = 0.0;
    
    // create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.block(0, 0, n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;
    
    // create square root matrix
    MatrixXd A = P_aug.llt().matrixL();
    
    // create augmented sigma points
    Xsig_aug.fill(0.0);
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
    }
    
    // Predict sigma points
    for(int p = 0; p < 2 * n_aug_ + 1; p++) {
        // Aliases for convenience
        double& px   = Xsig_aug(0, p);
        double& py   = Xsig_aug(1, p);
        double& v    = Xsig_aug(2, p);
        double& phi  = Xsig_aug(3, p);
        double& phid = Xsig_aug(4, p);
        double& nu_a = Xsig_aug(5, p);
        double& nu_p = Xsig_aug(6, p);
        
        // Predictions
        double px_p, py_p;
        if(fabs(phid) > 0.001) { // predict sigma points
            px_p = px + (v/phid) * (sin(phi + phid * delta_t) - sin(phi));
            py_p = py + (v/phid) * (-cos(phi + phid * delta_t) + cos(phi));
        } else { // avoid division by zero
            px_p = px + v * cos(phi) * delta_t;
            py_p = py + v * sin(phi) * delta_t;
        }
        px_p += 0.5 * delta_t * delta_t * cos(phi) * nu_a;
        py_p += 0.5 * delta_t * delta_t * sin(phi) * nu_a;
        
        double v_p = v + delta_t * nu_a;
        double phi_p = phi + phid * delta_t + 0.5 * delta_t * delta_t * nu_p;
        double phid_p = phid + delta_t * nu_p;
        
        // write predicted sigma points into right column
        Xsig_pred_(0, p) = px_p;
        Xsig_pred_(1, p) = py_p;
        Xsig_pred_(2, p) = v_p;
        Xsig_pred_(3, p) = phi_p;
        Xsig_pred_(4, p) = phid_p;
    }
    
    // Set weights
    double weight = lambda_ / (lambda_ + n_aug_);
    weights_(0) = weight;
    weight = 0.5/(lambda_+n_aug_);
    for(int p=1; p<2*n_aug_+1; p++)
        weights_(p) = weight;
    
    // predict state mean
    x_.fill(0.0);
    for(int sig=0; sig<2*n_aug_+1; sig++)
        x_ += weights_(sig) * Xsig_pred_.col(sig);
    
    // predict state covariance matrix
    P_.fill(0.0);
    for(int sig=0; sig<2*n_aug_+1; sig++) {
        MatrixXd x_diff = Xsig_pred_.col(sig) - x_;
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        P_ += weights_(sig) * x_diff * x_diff.transpose();
    }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    MatrixXd Zsig = MatrixXd(n_lidar_, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd(n_lidar_);
    MatrixXd S = MatrixXd(n_lidar_, n_lidar_);
    
    for(int p=0; p<2*n_aug_+1; p++) {
        double& px  = Xsig_pred_(0, p);
        double& py  = Xsig_pred_(1, p);
        Zsig(0, p) = px;
        Zsig(1, p) = py;
    }
    
    // calculate mean predicted measurement
    z_pred.fill(0.0);
    for(int p=0; p<2*n_aug_+1; p++)
        z_pred = z_pred + weights_(p) * Zsig.col(p);
    
    // calculate innovation covariance matrix S
    MatrixXd R = MatrixXd(n_lidar_, n_lidar_);
    R <<  std_laspx_*std_laspx_, 0,
          0,std_laspy_*std_laspy_;
    
    S.fill(0.0);
    for(int p=0; p<2*n_aug_+1; p++) {
        VectorXd z_diff = Zsig.col(p) - z_pred;
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        S = S + weights_(p) * z_diff * z_diff.transpose();
    }
    S = S + R;
    
    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_lidar_);
    
    // calculate cross correlation matrix
    Tc.fill(0.0);
    for(int s=0; s<2*n_aug_+1; s++) {
        VectorXd x_diff = Xsig_pred_.col(s) - x_;
        VectorXd z_diff = Zsig.col(s) - z_pred;
        
        // angle normalization - state prediction
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        Tc += weights_(s) * x_diff * z_diff.transpose();
    }
    
    // calculate Kalman gain K;
    MatrixXd K = MatrixXd(n_x_, n_lidar_);
    K = Tc * S.inverse();
    
    // update state mean and covariance matrix
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;
    x_ += K * z_diff;
    P_ += -K * S * K.transpose();
    
    NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
    
    
    lidar_count++;
    if(NIS_lidar_ > 5.991) {
        //std::cout << "NIS_lidar_: " << NIS_lidar_ << "count: " << ++NIS_lidar_exceed_count << "/" << lidar_count << std::endl;
    }
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_radar_);
  MatrixXd S = MatrixXd(n_radar_, n_radar_);
    
    for(int p=0; p<2*n_aug_+1; p++) {
        double px  = Xsig_pred_(0, p);
        double py  = Xsig_pred_(1, p);
        double v   = Xsig_pred_(2, p);
        double phi = Xsig_pred_(3, p);
        
        double rho = sqrt(px*px + py*py);
        Zsig(0, p) = rho;
        Zsig(1, p) = atan2(py, px);
        Zsig(2, p) = v*(px*cos(phi) + py*sin(phi))/rho;
    }
    
    // calculate mean predicted measurement
    z_pred.fill(0.0);
    for(int p=0; p<2*n_aug_+1; p++)
        z_pred = z_pred + weights_(p) * Zsig.col(p);
    
    // calculate innovation covariance matrix S
    MatrixXd R = MatrixXd(n_radar_, n_radar_);
    R <<  std_radr_*std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0,std_radrd_*std_radrd_;
    S.fill(0.0);
    for(int p=0; p<2*n_aug_+1; p++) {
        VectorXd z_diff = Zsig.col(p) - z_pred;
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        S = S + weights_(p) * z_diff * z_diff.transpose();
    }
    S = S + R;
    
    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_radar_);
    
    // calculate cross correlation matrix
    Tc.fill(0.0);
    for(int s=0; s<2*n_aug_+1; s++) {
        VectorXd x_diff = Xsig_pred_.col(s) - x_;
        VectorXd z_diff = Zsig.col(s) - z_pred;
        
        // angle normalization - state prediction
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        
        // angle normalization - measurement prediction
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
        
        Tc += weights_(s) * x_diff * z_diff.transpose();
    }
    
    // calculate Kalman gain K;
    MatrixXd K = MatrixXd(n_x_, n_radar_);
    K = Tc * S.inverse();
    
    // update state mean and covariance matrix
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;
    x_ += K * z_diff;
    P_ += -K * S * K.transpose();
    
    // NIS measurements update
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
    
    radar_count++;
    if(NIS_radar_ > 7.815) {
       // std::cout << "NIS_radar_: " << NIS_radar_ << "count: " << ++NIS_radar_exceed_count << "/" << radar_count << std::endl;
    }
}
