#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // Initializing RMSE vector
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    
    if( estimations.size() == 0) {
        std::cout << "Invalid estimations size." << std::endl;
        return rmse;
    }

    // Size mismatch handling
    if( estimations.size() != ground_truth.size() ) {
        std::cout << "Mismatched vector sizes" << std::endl;
        return rmse;
    }
    
    // Square Error
    for( size_t i=0;  i<estimations.size(); i++) {
        auto& est = estimations[i];
        auto& gt = ground_truth[i];
        VectorXd diff = est - gt;
        VectorXd diff2 = diff.array() * diff.array();
        rmse += diff2;
    }
    // Mean Square Error
    rmse /= estimations.size();
    
    // Root Mean Square Error
    rmse = rmse.array().sqrt();
    
    return rmse;
}
