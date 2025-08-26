import os.path

import matplotlib.pyplot as plt
import numpy as np
from daily import do_meta, data_path

# Example data (replace these with your actual data)
values = do_meta()

values = sorted(values, key=lambda x: x["date"])


event_names = []
for i, value in enumerate(values):
    if i % 2 == 0:
        event_names.append(f"{value['date']} - "
                           f"{value['type']} "
                           f"{value['method']} - "
                           f"{value['community'] if 'community' in value else value['community1'] + ' - ' + value['community2']}")
time_points = np.arange(-28, 28)  # Assuming we have time points from -5 to +5 for pre and post-event

ts_values = [value["values"] for value in values]

# Example time series data (random data for illustration)
np.random.seed(42)  # For reproducibility
print(len(ts_values[0]))
old_users = []
new_users = []
for i in range(len(ts_values)):
    if i % 2 == 0:
        old_users.append(ts_values[i])
    else:
        new_users.append(ts_values[i])

# Example mean, max, and min data
means_old = np.array([np.mean(old) for old in old_users])
max_old = np.array([np.max(old) for old in old_users])
min_old = np.array([np.min(old) for old in old_users])

means_new = np.array([np.mean(new) for new in new_users])
max_new = np.array([np.max(new) for new in new_users])
min_new = np.array([np.min(new) for new in new_users])

fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(15, 25))  # 7 subplots in a grid
axs = axs.flatten()

# Plot first 6 events
for i, event in enumerate(event_names):
    ax = axs[i]
    ax.plot(time_points, old_users[i], label='Old Users', color='blue')
    ax.plot(time_points, new_users[i], label='New Users', color='green')

    # Plotting mean with error bars representing min and max, shifted to the side
    ax.errorbar([-29], [means_old[i]],
                yerr=[[means_old[i] - min_old[i]], [max_old[i] - means_old[i]]],
                fmt='o', color='blue', capsize=5, label='Mean Old Users', alpha=1)

    ax.errorbar([-29], [means_new[i]],
                yerr=[[means_new[i] - min_new[i]], [max_new[i] - means_new[i]]],
                fmt='o', color='green', capsize=5, label='Mean New Users', alpha=1)

    ax.set_title(event)
    ax.axvline(x=0, color='red', linestyle='--')  # Vertical line at the event point
    ax.legend()

# Add the final event to the middle of the bottom row
"""for i in range(12, 14):
    ax = axs[i+1]  # This correctly places the last event in the middle
    ax.plot(time_points, old_users[i], label='Old Users', color='blue')
    ax.plot(time_points, new_users[i], label='New Users', color='green')

    # Plotting mean with error bars representing min and max, shifted to the side
    ax.errorbar([-29], [means_old[i]],
                yerr=[[means_old[i] - min_old[i]], [max_old[i] - means_old[i]]],
                fmt='o', color='blue', capsize=5, label='Mean Old Users', alpha=0.7)

    ax.errorbar([-29], [means_new[i]],
                yerr=[[means_new[i] - min_new[i]], [max_new[i] - means_new[i]]],
                fmt='o', color='green', capsize=5, label='Mean New Users', alpha=0.7)

    ax.set_title(event_names[i])
    ax.axvline(x=0, color='red', linestyle='--')  # Vertical line at the event point
    ax.legend()

# Hide the bottom-left and bottom-right subplots (completely remove axes)
axs[12].axis('off')
axs[15].axis('off')
#axs[8].axis('off')  # Hide the bottom-right"""
axs[13].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(data_path, "old_new_users.png"))
plt.show()