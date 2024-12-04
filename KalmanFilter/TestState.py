import unittest
import numpy as np
from KalmanFilter import KalmanFilter  

class TestState(unittest.TestCase):
    def test_default_initialization(self):
        state = KalmanFilter()
        self.assertEqual(state.get_state_estimate().tolist(), [0, 0, 0])
        self.assertTrue(np.allclose(state.get_error_covariance(), np.eye(3)))

    def test_set_and_get_state_estimate(self):
        state = KalmanFilter()
        new_mean = [1, 2, 3]
        state.set_state_estimate(new_mean)
        self.assertEqual(state.get_state_estimate().tolist(), new_mean)

    def test_set_and_get_error_covariance(self):
        state = KalmanFilter()
        new_covariance = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        state.set_error_covariance(new_covariance)
        self.assertEqual(state.get_error_covariance().tolist(), new_covariance)

    def test_state_estimate_update(self):
        state = KalmanFilter()
        A = np.eye(3)  
        B = np.eye(3) 
        u = np.array([1, 0, 0])  
        Q = np.zeros((3, 3))  

        state.predict_new_state(A, B, u, Q)
        self.assertEqual(state.get_state_estimate().tolist(), [1, 0, 0])

    def test_error_covariance_update(self):
        state = KalmanFilter()
        A = np.eye(3)  
        B = np.eye(3)  
        u = np.array([0, 0, 0])  
        Q = np.eye(3) * 0.1  

        state.predict_new_state(A, B, u, Q)
        expected_covariance = np.eye(3) * 1.1  
        self.assertTrue(np.allclose(state.get_error_covariance(), expected_covariance))

    def test_no_movement(self):
        state = KalmanFilter()
        A = np.eye(3)  
        B = np.eye(3)  
        u = np.array([0, 0, 0]) 
        Q = np.zeros((3, 3))  

        state.predict_new_state(A, B, u, Q)
        self.assertEqual(state.get_state_estimate().tolist(), [0, 0, 0])
        self.assertTrue(np.allclose(state.get_error_covariance(), np.eye(3)))

    def test_basic_update(self):
        state = KalmanFilter()
        H = np.eye(3) 
        R = np.eye(3) * 0.1  
        z = np.array([1, 1, 0])  

        state.correct_state(z, H, R)
        self.assertTrue(np.allclose(state.get_state_estimate(), [1, 1, 0], atol=0.1))

    def test_error_covariance_update(self):
        state = KalmanFilter()
        H = np.eye(3)  
        R = np.eye(3) * 0.1  
        z = np.array([1, 1, 0])  

        initial_covariance = state.get_error_covariance()
        state.correct_state(z, H, R)
        updated_covariance = state.get_error_covariance()

        self.assertTrue(np.trace(updated_covariance) < np.trace(initial_covariance))

    def test_compute_and_get_innovation_covariance(self):
        state = KalmanFilter()
        state.set_error_covariance(np.eye(3))  
        H = np.eye(3)  
        R = np.eye(3) * 0.1  

        S = state.compute_and_get_innovation_covariance(H, R)
        expected_S = np.eye(3) + R 
        self.assertTrue(np.allclose(S, expected_S))
        
    def test_compute_and_get_kalman_gain(self):
        state = KalmanFilter()
        state.set_error_covariance(np.eye(3))  
        H = np.eye(3)  
        R = np.eye(3) * 0.1  

        K = state.compute_and_get_kalman_gain(H, R)
        S = np.eye(3) + R 
        expected_K = np.dot(np.eye(3), np.dot(H.T, np.linalg.inv(S)))
        self.assertTrue(np.allclose(K, expected_K))

    def test_compute_and_get_measurement_residual(self):
        state = KalmanFilter()
        state.set_state_estimate([0, 0, 0])  
        z = np.array([1, 1, 1])  
        H = np.eye(3)  

        residual = state.compute_and_get_measurement_residual(z, H)
        expected_residual = z - np.dot(H, [0, 0, 0])
        self.assertTrue(np.allclose(residual, expected_residual))

    def test_update_state_estimate(self):
        state = KalmanFilter()
        state.set_state_estimate([0, 0, 0])  
        z = np.array([1, 1, 1])  
        H = np.eye(3) 
        K = np.eye(3) * 0.5  

        state.update_state_estimate(K, z, H)
        expected_state = np.array([0.5, 0.5, 0.5])  
        self.assertTrue(np.allclose(state.get_state_estimate(), expected_state))

    def test_update_error_covariance(self):
        state = KalmanFilter()
        state.set_error_covariance(np.eye(3))  
        K = np.eye(3) * 0.5  
        H = np.eye(3)  

        state.update_error_covariance(K, H)
        expected_covariance = (np.eye(3) - np.dot(K, H)) @ np.eye(3)
        self.assertTrue(np.allclose(state.get_error_covariance(), expected_covariance))


if __name__ == "__main__":
    unittest.main()
