import datetime
import time
from src.pipeline.interface_pipeline_component import PipelineComponent
from collections import deque


class GestureFireController(PipelineComponent):
    def __init__(self, gesture_queue_size=4, gesture_fire_threshold=30):
        self.input_type = str  # ? str or int?
        self.output_type = str
        self.gesture_queue_size = gesture_queue_size
        self.gesture_fire_threshold = gesture_fire_threshold

        self.gesture_queue = deque(maxlen=self.gesture_queue_size)
        for _ in range(self.gesture_queue_size):
            self.gesture_queue.append("idle")
        self.frames_since_last_gesture = gesture_fire_threshold  # init as threshold so first gesture fires

    def run(self, input_data):
        gesture = input_data

        self.gesture_queue.append(gesture)
        #print(self.gesture_queue)
        #print(f"Number of gestures in queue: {len(set(self.gesture_queue))}")
        #print(f"Frames since last gesture: {self.frames_since_last_gesture}")
        if len(set(self.gesture_queue)) == 1 and self.gesture_queue[0] != "idle":  # check if all past "gesture_queue_size" gestures are the same
            if self.frames_since_last_gesture >= self.gesture_fire_threshold:
                self.frames_since_last_gesture = 0
                return gesture

        self.frames_since_last_gesture += 1
        return "idle"


if __name__ == "__main__":
    gesture_fire_controller = GestureFireController()
    print(f'Gesture 0: {gesture_fire_controller.run("idle")}\n')
    print(f'Gesture 1: {gesture_fire_controller.run("idle")}\n')
    print(f'Gesture 2: {gesture_fire_controller.run("idle")}\n')
    print(f'Gesture 3: {gesture_fire_controller.run("idle")}\n')
    print(f'Gesture 4: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 5: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 6: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 7: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 8: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 9: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 10: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 11: {gesture_fire_controller.run("idle")}\n')
    print(f'Gesture 12: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 13: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 14: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 15: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 16: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 17: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 18: {gesture_fire_controller.run("rotate")}\n')
    for _ in range(50):
        print(f'Gesture Loop: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 19: {gesture_fire_controller.run("rotate")}\n')
    print(f'Gesture 20: {gesture_fire_controller.run("rotate")}\n')



