# Gesture Classifer
## Description
A Python Project in the course of blabla (Marvin Roth, Julius Schneller, Johannes Schraud)
- [Tetris by Tech With Tim](https://github.com/techwithtim/Tetris-Game)
- you need python 3.10

***

## Installation and Setup
1. Clone this GIT repository with one of the following commands:
    ```
    git clone git@gitlab2.informatik.uni-wuerzburg.de:hci/teaching/courses/machine-learning/student-submissions/ws23/Team-9/gesture-classifier.git
    ```
    ```
    git clone https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws23/Team-9/gesture-classifier.git
    ```
2.  Execute the following commands to set up a new virtual environment, install the necessary requirements, and add the project-root to your `PYTHONPATH`-variable:
    ```
    python3 -m venv venv
    . venv/bin/activate
    python -m pip install -r requirements.txt
    . venv/bin/activate
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```



## Usage
### General
1. Make sure you are inside the `gesture-classifier` directory.
2. Use the first following command to check if the virtual environment is activated - otherwise use the second following command to activate it:
    ```
    echo $VIRTUAL_ENV
    ```
    ```
    . venv/bin/activate
    ```
3. Use the first following command to check if the project-root is in the `PYTHONPATH`-variable - otherwise use the second following command to add it:
    ```
    echo $PYTHONPATH
    ```
    ```
    export PYTHONPATH=$PWD:$PYTHONPATH
    ```
   
### Prediction Mode
1. Execute the following command to start the Prediction Mode:
    ```
    python src/prediction_mode.py
    ```
    * (If your camera is not being detected you might have to change the camera-index in `src/prediction_mode.py`.)
  
### Test Mode
1. Move your `input_csv` file to the `resource/performance_evaluation_resources` directory.
2. Execute the following command to start the Test Mode:
    ```
    python src/test_mode.py \
        --input_csv <path/to/your/input.csv> \
        --output_csv_name <output_name>
    ```
    e.g.
    ```
    python src/test_mode.py \
        --input_csv resource/performance_evaluation_resources/demo_video_frames_rotate.csv \
        --output_csv_name emitted_events.csv
    ```
   
### Performance Mode
1. Move your `events_csv` file and `ground_truth_csv` file to the `resource/performance_evaluation_resources` directory.
2. Execute the following command to start the Performance Mode:
    ```
    python src/performance_mode.py \
        --events_csv path/to/your/events.csv \
        --ground_truth_csv path/to/your/ground_truth.csv
    ```
    e.g.
    ```
    python src/performance_mode.py \
        --events_csv resource/performance_evaluation_resources/emitted_events.csv \
        --ground_truth_csv resource/performance_evaluation_resources/demo_video_csv_with_ground_truth_rotate.csv
    ```
   