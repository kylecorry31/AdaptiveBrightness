import numpy as np
import cv2
from subprocess import call, check_output
import time
import datetime
import psutil
import tensorflow as tf
import argparse


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
        try:
            self.camera = cv2.VideoCapture(self.camera_port)
            LightSensor.__set_auto_exposure(False)
            self.enabled = True
        except Exception:
            self.enabled = False

    def get(self):
        if self.enabled:
            try:
                _, frame = self.camera.read()
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                channels = cv2.split(yuv)
                return np.mean(channels[0])
            except Exception:
                return 0.0
        else:
            return 0.0

    def disable(self):
        LightSensor.__set_auto_exposure(True)
        if self.enabled:
            self.camera.release()


class Backlight:

    def __init__(self, lerp_time=2000.0):
        self.lerp_time = float(lerp_time)

    def set_brightness(self, percentage):
        try:
            percentage = int(to_range(round(percentage), 0, 100))
            log("Setting brightness to " + str(percentage))
            if self.lerp_time == 0:
                call(['gdbus', 'call', '--session', '--dest', 'org.gnome.SettingsDaemon.Power',
                      '--object-path', '/org/gnome/SettingsDaemon/Power', '--method',
                      'org.freedesktop.DBus.Properties.Set',
                      'org.gnome.SettingsDaemon.Power.Screen', 'Brightness',
                      '<int32 ' + str(percentage) + '>'])
            start_time = time.time()
            current_percentage = self.get_brightness()
            lerp = lambda t: int(to_range(round((percentage - current_percentage) * t + current_percentage), 0, 100))
            time_diff = (time.time() - start_time) * 1000.0
            while time_diff < self.lerp_time:
                call(['gdbus', 'call', '--session', '--dest', 'org.gnome.SettingsDaemon.Power',
                      '--object-path', '/org/gnome/SettingsDaemon/Power', '--method', 'org.freedesktop.DBus.Properties.Set',
                      'org.gnome.SettingsDaemon.Power.Screen', 'Brightness', '<int32 ' + str(lerp(time_diff / self.lerp_time)) + '>'])
                time_diff = (time.time() - start_time) * 1000.0
        except Exception:
            pass

    def get_brightness(self):
        try:
            output = check_output(['gdbus', 'call', '--session', '--dest', 'org.gnome.SettingsDaemon.Power',
                                   '--object-path', '/org/gnome/SettingsDaemon/Power', '--method',
                                   'org.freedesktop.DBus.Properties.Get',
                                   'org.gnome.SettingsDaemon.Power.Screen', 'Brightness'])

            number = ""

            for char in output:
                if char.isdigit():
                    number += char

            return int(number)
        except Exception:
            return 0


class Battery:

    def __init__(self):
        pass

    def get_percent(self):
        return psutil.sensors_battery().percent

    def is_plugged_in(self):
        return psutil.sensors_battery().power_plugged


class Calendar:

    def __init__(self):
        pass

    def get_day_of_week(self):
        return datetime.datetime.now().weekday()


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


class RangedAdaptiveBrightness(AdaptiveBrightness):

    def __init__(self, base_point, change_threshold=6, light_sensor=LightSensor(), backlight=Backlight()):
        AdaptiveBrightness.__init__(self, light_sensor, backlight)
        self.base_point = base_point
        self.last_change = -1
        self.change_threshold = change_threshold

    def run(self):
        light = self.get_light()
        if self.last_change == -1 or abs(light - self.last_change) > self.change_threshold:
            light_percent = light / 255.0
            min_brightness = max(0, self.base_point - 10)
            max_brightness = min(100, self.base_point + 20)
            self.set_brightness(light_percent * (max_brightness - min_brightness) + min_brightness)
            self.last_change = light


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


class ExponentAdaptiveBrightness(AdaptiveBrightness):
    def __init__(self, brightness_compensation, change_threshold=6, light_sensor=LightSensor(), backlight=Backlight()):
        AdaptiveBrightness.__init__(self, light_sensor, backlight)
        self.brightness_compensation = brightness_compensation
        self.last_change = -1
        self.change_threshold = change_threshold

    def run(self):
        light = self.get_light()
        if self.last_change == -1 or abs(light - self.last_change) > self.change_threshold:
            self.set_brightness((light / 255.0) ** (3 - (1 + self.brightness_compensation)) * 100)
            self.last_change = light


class MLAdaptiveBrightness(AdaptiveBrightness):

    def __init__(self, change_threshold=6, light_sensor=LightSensor(), backlight=Backlight()):
        AdaptiveBrightness.__init__(self, light_sensor, backlight)
        tf.enable_eager_execution()
        self.change_threshold = change_threshold
        self.last_change = -1
        self.data = []
        self.learning_rate = 0.01
        self.num_steps = 10
        self.batch_size = 10
        self.my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.my_optimizer, 5.0)
        self.model = tf.keras.models.load_model(
            "model",
            custom_objects=None,
            compile=False
        )
        self.model.compile(optimizer=self.my_optimizer, loss=tf.keras.losses.mean_squared_error)
        self.last_brightness = self.backlight.get_brightness()

        for i in range(256):
            print(self.model.predict(np.array([i]))[0][0])


    def run(self):
        light = self.get_light()
        current_brightness = self.backlight.get_brightness()
        if current_brightness != self.last_brightness:
            self.learn(self.last_change, current_brightness)
            self.last_brightness = self.backlight.get_brightness()
        if self.last_change == -1 or abs(light - self.last_change) > self.change_threshold:
            self.set_brightness(self.model.predict(np.array([light]))[0][0])
            self.last_change = light
            self.last_brightness = self.backlight.get_brightness()

    def learn(self, light, brightness):
        remove_list = []
        for value in self.data:
            if value[0] == light:
                remove_list.append(value)
        for value in remove_list:
            self.data.remove(value)
        self.data.append([light, brightness])
        features = np.array([x[0] for x in self.data])
        labels = np.array([x[1] for x in self.data])
        self.model.fit(features, labels, batch_size=self.batch_size, epochs=self.num_steps)
        tf.keras.models.save_model(
            self.model,
            "model",
            overwrite=True,
            include_optimizer=False
        )


class AdaptiveBrightnessFactory:

    def __init__(self):
        pass

    def get(self, algorithm, brightness_compensation=0.4):
        if algorithm == "ml":
            return MLAdaptiveBrightness()
        elif algorithm == "simple":
            return SimpleAdaptiveBrightness(brightness_compensation)
        elif algorithm == "exponent":
            return ExponentAdaptiveBrightness(brightness_compensation)
        elif algorithm == "ranged":
            return RangedAdaptiveBrightness(brightness_compensation)
        return None


class UserContext:

    def __init__(self, ambient_light, time_of_day, day_of_week, battery_percent, on_charger):
        self.ambient_light = ambient_light  # Between 0 and 100
        self.time_of_day = time_of_day  # Between 0 and 24
        self.day_of_week = day_of_week  # Monday = 0, ..., Sunday = 6
        self.battery_percent = battery_percent  # Between 0 and 100
        self.on_charger = on_charger  # True or False

    @staticmethod
    def generate(light_sensor=LightSensor(), clock=Clock(), calendar=Calendar(), battery=Battery()):
        light_sensor.enable()
        ctx = UserContext(light_sensor.get() / 255.0 * 100, clock.get_as_float(), calendar.get_day_of_week(),
                          battery.get_percent(), battery.is_plugged_in())
        light_sensor.disable()
        return ctx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", "-f",
                        help="The frequency in seconds which the adaptive brightness will update. Default is 30s.",
                        type=int)
    parser.add_argument("--algorithm", "-a", help="The algorithm to use. Default is ml.", type=str,
                        choices=["ml", "simple", "exponent", "ranged"])
    parser.add_argument("--brightness_compensation", "-b",
                        help="The brightness compensation factor if using the simple algorithm. Strictly positive, normally less than 1.",
                        type=float)
    args = parser.parse_args()

    if not args.frequency:
        args.frequency = 30

    if not args.algorithm:
        args.algorithm = "ml"

    if args.brightness_compensation is None:
        if args.algorithm == "ranged":
            args.brightness_compensation = 50
        else:
            args.brightness_compensation = 0.4

    algorithm_factory = AdaptiveBrightnessFactory()
    adaptive_brightness = algorithm_factory.get(args.algorithm, args.brightness_compensation)
    while True:
        adaptive_brightness.run()
        time.sleep(args.frequency)
