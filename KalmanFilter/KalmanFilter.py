import numpy as np

class KalmanFilter:
    def __init__(self):
        self.state_estimate = np.array([0, 0, 0])
        self.error_covariance = np.eye(3)

    def set_state_estimate(self, mean):
        self.state_estimate = np.array(mean)

    def get_state_estimate(self):
        return self.state_estimate

    def set_error_covariance(self, covariance):
        self.error_covariance = np.array(covariance)

    def get_error_covariance(self):
        return self.error_covariance
    
    def predict_new_state(self, A, B, u, Q):
        self.predict_new_state_estimate(A, B, u)
        self.predict_new_error_covariance(A, Q)

    def predict_new_state_estimate(self, A, B, u):
        self.state_estimate = np.dot(A, self.state_estimate.T) + np.dot(B, u.T)

    def predict_new_error_covariance(self, A, Q):
        self.error_covariance = A @ self.error_covariance @ A.T + Q

    def correct_state(self, z, H, R):
        kalman_gain = self.compute_and_get_kalman_gain(H, R)
        self.update_state_estimate(kalman_gain, z, H)
        self.update_error_covariance(kalman_gain, H)

    def compute_and_get_kalman_gain(self, H, R):
        innovation_covariance = self.compute_and_get_innovation_covariance(H, R)
        return np.dot(self.error_covariance, np.dot(H.T, np.linalg.inv(innovation_covariance)))
        
    def compute_and_get_innovation_covariance(self, H, R):
        return np.dot(H, np.dot(self.error_covariance, H.T)) + R
    
    def update_state_estimate(self, K, z, H):
        measurement_residual = self.compute_and_get_measurement_residual(z, H)
        self.state_estimate = self.state_estimate + np.dot(K, measurement_residual)

    def compute_and_get_measurement_residual(self, z, H):
        return z - np.dot(H, self.state_estimate)

    def update_error_covariance(self, K, H):
        kalman_gain_adjustment = self.compute_and_get_kalman_gain_adjustment(K, H)
        self.error_covariance = np.dot(kalman_gain_adjustment, self.error_covariance)

    def compute_and_get_kalman_gain_adjustment(self, K, H):
        return np.eye(3) - np.dot(K, H)
