import argparse
import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_args():
    parser = argparse.ArgumentParser(description=('Export the tensorboard trainingsdata to csv'))
    parser.add_argument('tensorboard_folder', type=str, help="folder containing the tensorboard logs")
    parser.add_argument('output_folder', type=str, help="output folder where the csv should go")
    parser.add_argument('--scalar', type=str,
                        default='', help="name of the scalar to export (if not given all are exported)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # create output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    root_path = args.tensorboard_folder
    runs = os.listdir(root_path)

    for run in runs:
        path=f"{root_path}/{run}/"
        try:
            event_acc = EventAccumulator(path)
            event_acc.Reload()

            if args.scalar == '':
                tags = event_acc.Tags()["scalars"]
            else:
                tags = [args.scalar]

            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"value": values, "step": step}
                r = pd.DataFrame(r)
                tag = tag.replace("/", "_") # to avoid interpreting tag names as directories
                r.to_csv(f"{args.output_folder}/{run}_{tag}.csv", index_label='index')
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
            traceback.print_exc()
            exit(1)
