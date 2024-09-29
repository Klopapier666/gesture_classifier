import argparse
import subprocess
# Example parameters:
#   --input_csv=demo_data/demo_video_csv_with_ground_truth_rotate.csv
#   --output_csv_name=log_emitted_events_output_csv_name.csv


parser = argparse.ArgumentParser()
parser.add_argument("--events_csv",
                    help="CSV file containing a column 'events' with the events predicted by your model",
                    required=True)
parser.add_argument("--ground_truth_csv",
                    help="CSV file containing a column 'ground_truth' with the correct gesture for each frame (may be the same as 'events_csv')",
                    required=True)

args = parser.parse_known_args()[0]
print(args.events_csv)
print(args.ground_truth_csv)


command1 = [
    "python3",
    "src/util/performance_score/calculator.py",
    "--events_csv",
    args.events_csv,
    "--ground_truth_csv",
    args.ground_truth_csv
]

command2 = [
    "python3",
    "src/util/performance_score/events_visualization.py",
    "--events_csv",
    args.events_csv,
    "--ground_truth_csv",
    args.ground_truth_csv
]

print("Executing calculator.py...")
process1 = subprocess.run(
    command1,
    capture_output=True,
    text=True
)
print(process1.stdout)
print(process1.stderr)

print("Executing events_visualization.py...")
process2 = subprocess.run(
    command2,
    capture_output=True,
    text=True
)
print(process2.stdout)
print(process2.stderr)
