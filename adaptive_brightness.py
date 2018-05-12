import cv2
import numpy as np
from subprocess import call
import time
import datetime
import psutil


def to_range(value, minimum, maximum):
    return min(max(value, minimum), maximum)


def log(output):
    print output


class LightSensor:

    def __init__(self, camera_port=0):
        self.camera = None
        self.camera_port = camera_port
        self.enabled = False

    @staticmethod
    def __set_auto_exposure(auto_exposure_on):
        call(["v4l2-ctl", "--set-ctrl", "exposure_auto_priority=" + str(int(auto_exposure_on))])

    def enable(self):
        self.camera = cv2.VideoCapture(self.camera_port)
        LightSensor.__set_auto_exposure(False)
        self.enabled = True

    def get(self):
        if self.enabled:
            _, frame = self.camera.read()
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            channels = cv2.split(yuv)
            return np.mean(channels[0])
        else:
            return 0.0

    def disable(self):
        LightSensor.__set_auto_exposure(True)
        if self.enabled:
            self.camera.release()


class Backlight:

    def __init__(self):
        pass

    def set_brightness(self, percentage):
        percentage = int(to_range(round(percentage), 0, 100))
        log("Setting brightness to " + str(percentage))
        call(['gdbus', 'call', '--session', '--dest', 'org.gnome.SettingsDaemon.Power',
              '--object-path', '/org/gnome/SettingsDaemon/Power', '--method', 'org.freedesktop.DBus.Properties.Set',
              'org.gnome.SettingsDaemon.Power.Screen', 'Brightness', '<int32 ' + str(percentage) + '>'])


class Battery:

    def __init__(self):
        pass

    def get_percent(self):
        return psutil.sensors_battery().percent

    def is_plugged_in(self):
        return psutil.sensors_battery().power_plugged


class Clock:

    def __init__(self):
        pass

    def get_as_float(self):
        time_of_day = self.get()
        return time_of_day.hour + time_of_day.minute / 60.0

    def get(self):
        return datetime.datetime.now().time()


class LowPassFilter:

    def __init__(self, filter_coef):
        self.filter_coef = to_range(filter_coef, 0, 1)
        self.last_value = 0.0

    def filter(self, value):
        self.last_value = self.filter_coef * self.last_value + (1 - self.filter_coef) * value
        return self.last_value


class AdaptiveBrightness:

    def __init__(self, light_sensor=LightSensor(), backlight=Backlight()):
        self.light_sensor = light_sensor
        self.backlight = backlight

    def get_light(self):
        self.light_sensor.enable()
        light = self.light_sensor.get()
        log("Read light as " + str(int(round(light))))
        self.light_sensor.disable()
        return light

    def set_brightness(self, percentage):
        self.backlight.set_brightness(percentage)


class SimpleAdaptiveBrightness(AdaptiveBrightness):

    def __init__(self, brightness_compensation, change_threshold=6, light_sensor=LightSensor(), backlight=Backlight()):
        AdaptiveBrightness.__init__(self, light_sensor, backlight)
        self.brightness_compensation = brightness_compensation
        self.last_change = -1
        self.change_threshold = change_threshold

    def run(self):
        light = self.get_light()
        if self.last_change == -1 or abs(light - self.last_change) > self.change_threshold:
            self.set_brightness(light * self.brightness_compensation)
            self.last_change = light


if __name__ == "__main__":
    adaptive_brightness = SimpleAdaptiveBrightness(0.5)
    while True:
        adaptive_brightness.run()
        time.sleep(1)
