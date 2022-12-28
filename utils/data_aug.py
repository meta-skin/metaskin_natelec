"""
This code is data augmentation methods for sequential sensor data
"""

import random
import torch
from scipy.interpolate import CubicSpline

class DataAugmentation():
    def __init__(self, device, params, window_size=16):
        self.device = device
        self.window_size = window_size
        self.params = params

    def jittering(self, sensor):
        new_sensor = sensor + \
            torch.normal(self.params.jittering_mean, self.params.jittering_std, size=sensor.shape).to(self.device)
        return new_sensor

    def shifting(self, sensor):
        randomNormalShift = torch.normal(0, 0.1, size=(
            sensor.size(0), 1)).repeat(1, sensor.size(1)).to(self.device)
        new_sensor = sensor + randomNormalShift
        return new_sensor

    def cropping(self, sensor):
        # trying with fixed crop size: 8
        # get random crop starting point
        cropST = [random.randint(0, self.window_size//2)
                  for _ in range(len(sensor))]
        # iteratively crop sensor values in each batched sample
        sensor = torch.cat([sensor[i][cropST[i]:cropST[i] + self.window_size //
                           2].reshape(1, -1) for i in range(len(sensor))], 0)
        start, end = sensor[:, :-1], sensor[:, 1:]
        interp_sensor = torch.lerp(start, end, 0.5)
        interp_sensor_end = torch.lerp(
            sensor[:, -2].reshape(-1, 1), sensor[:, -1].reshape(-1, 1), 1.1)
        interp_sensor = torch.cat([interp_sensor, interp_sensor_end], 1)
        resize_sensor = []
        for i in range(self.window_size):
            if(i % 2 == 0):
                resize_sensor.append(sensor[:, i//2].reshape(-1, 1))
            else:
                resize_sensor.append(interp_sensor[:, i//2].reshape(-1, 1))
        resize_sensor = torch.cat(resize_sensor, 1)
        return resize_sensor

    def scaling(self, sensor, ):
        alpha = torch.tensor([random.uniform(0.7, 1.2) for _ in range(
            len(sensor))]).reshape(-1, 1).repeat(1, sensor.size(1)).to(self.device)

        grad_sensor = sensor - \
            sensor[:, 0].reshape(-1, 1).repeat(1, sensor.size(1))
        grad_sensor = grad_sensor * alpha
        new_sensor = sensor[:, 0].reshape(-1, 1).repeat(1, sensor.size(1))
        new_sensor = new_sensor + grad_sensor
        return new_sensor

    def magwraping(self, sensor):
        def generateRandomCurves(X, sigma=0.01, knot=4):
            xx = (torch.ones((X.shape[1], 1)) * (torch.arange(0,
                  X.shape[0], (X.shape[0] - 1)/(knot+1)))).transpose()
            yy = torch.normal(1.0, sigma,
                                  size=(knot + 2, X.shape[1]))
            x_range = torch.arange(X.shape[0])
            cs_x = CubicSpline(xx[:, 0], yy[:, 0])
            return torch.tensor(cs_x(x_range)).transpose()
        mag = generateRandomCurves(sensor[0].reshape(-1, 1))
        new_sensor = sensor * mag.float().to(self.device)
        return new_sensor



