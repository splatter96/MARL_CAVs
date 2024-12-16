import numpy as np
from matplotlib import pyplot as plt

simulation_frequency = 15  # [Hz] frequency of the main traffic simulation
radar_frequency = 2000  # [Hz] frequency of radar overlap calculation

frame_times = [
    20.0 / 1000,
    9.0 / 1000,
    20.0 / 1000,
]  # [s] duration of a single radar frame

duty_cycles = [0.06, 0.03, 0.03]  # [% / 100]

cycle_offsets = [
    0 / 1000,
    5 / 1000,
    10 / 1000,
]  # [s] offset from start of simulation of the first on period of the radar

end_time = 10  # [s]

radar_steps_per_frame = int(frame_times[0] * radar_frequency)
timestep = 1 / simulation_frequency
radar_frames_per_timestep = int(timestep / frame_times[0])

print(f"Radar steps per radar frame {radar_steps_per_frame}")
print(f"Minimal possible duty cycle {frame_times[0]/radar_steps_per_frame}")


def is_on(t, duty_cycle, offset, frame_time):
    return ((t - offset) % frame_time) < frame_time * duty_cycle


frames = 0
overlaps = 0
overlap = False

t = 0
while t < end_time:
    t2 = t
    for _ in range(radar_frames_per_timestep):
        for _ in range(radar_steps_per_frame):
            for i in range(len(duty_cycles) - 1):
                if is_on(t, duty_cycles[0], cycle_offsets[0], frame_times[0]) and is_on(
                    t, duty_cycles[i + 1], cycle_offsets[i + 1], frame_times[i + 1]
                ):
                    overlap = True
            # as soon as one overlap occurs entire frame can be discarded
            if overlap:
                overlaps += 1
                overlap = False
                break

            t2 += 1 / radar_frequency
            # print(f"{t2=}")

        frames += 1
    t += timestep
    # print(f"{t=}")

print(f"Calculated {frames}")
print(f"Overlaps prob {overlaps/(frames)}")


T = np.linspace(0, 0.2, 10000)
plt.plot(T, is_on(T, duty_cycles[0], cycle_offsets[0], frame_times[0]), label="ego")

for i in range(len(duty_cycles) - 1):
    plt.plot(T, is_on(T, duty_cycles[i + 1], cycle_offsets[i + 1], frame_times[i + 1]))

plt.legend(loc="upper left")
plt.show()
