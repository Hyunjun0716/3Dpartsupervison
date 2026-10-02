"""
Check training progress from tensorboard logs
"""

from utils.paths import project_path
import argparse
import os
import glob
from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser(description='Inspect the latest training log')
parser.add_argument('--log_root', default=project_path('logs_gen'))
args = parser.parse_args()

# Find latest log directory
log_dirs = sorted(glob.glob(os.path.join(args.log_root, 'GEN_*')))
latest_log = log_dirs[-1] if log_dirs else None

if latest_log:
    print(f"Checking logs from: {latest_log}")

    # Load tensorboard events
    ea = event_accumulator.EventAccumulator(latest_log)
    ea.Reload()

    print("\nAvailable tags:")
    print(ea.Tags())

    # Check scalar tags
    if 'scalars' in ea.Tags():
        print("\nScalar metrics:")
        for tag in ea.Tags()['scalars']:
            print(f"  - {tag}")

        # Get loss values
        if 'train/loss' in ea.Tags()['scalars']:
            loss_events = ea.Scalars('train/loss')
            print(f"\n\nLoss progression (first 20 iterations):")
            for i, event in enumerate(loss_events[:20]):
                print(f"  Step {event.step}: {event.value:.4f}")

            if len(loss_events) > 20:
                print(f"\nLoss progression (last 10 iterations):")
                for event in loss_events[-10:]:
                    print(f"  Step {event.step}: {event.value:.4f}")

            # Calculate loss change
            if len(loss_events) >= 2:
                initial_loss = loss_events[0].value
                current_loss = loss_events[-1].value
                change = initial_loss - current_loss
                change_pct = (change / initial_loss) * 100
                print(f"\nLoss change: {initial_loss:.4f} -> {current_loss:.4f}")
                print(f"Reduction: {change:.4f} ({change_pct:.2f}%)")
else:
    print("No log directory found")
