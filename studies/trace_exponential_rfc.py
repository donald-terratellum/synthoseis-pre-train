## -----------------------------------------------
### model reflectivity with bandlimited exponential spikes
### -----------------------------------------------
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew


def flat_hanning_taper(n: int, pct_flat: float = 0.95) -> np.ndarray:
	"""Return a symmetric flat-top Hanning taper of length n."""
	if n <= 0:
		raise ValueError("n must be > 0")
	if not (0.0 <= pct_flat <= 1.0):
		raise ValueError("pct_flat must be in [0, 1]")

	if pct_flat >= 1.0:
		return np.ones(n, dtype=np.float32)
	if pct_flat <= 0.0:
		return np.hanning(n).astype(np.float32)

	taper_len = int(round((1.0 - pct_flat) * n / 2.0))
	taper_len = min(max(taper_len, 0), n // 2)
	if taper_len == 0:
		return np.ones(n, dtype=np.float32)

	taper = np.ones(n, dtype=np.float32)
	ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_len, dtype=np.float32)))
	taper[:taper_len] = ramp
	taper[-taper_len:] = ramp[::-1]
	return taper


def count_peaks_troughs(x: np.ndarray) -> tuple[int, int, int]:
	"""Count strict local peaks and troughs using immediate neighbors."""
	if x.size < 3:
		return 0, 0, 0
	center = x[1:-1]
	left = x[:-2]
	right = x[2:]
	peaks = int(np.sum((center > left) & (center > right)))
	troughs = int(np.sum((center < left) & (center < right)))
	return peaks, troughs, peaks + troughs


def simulate_one(plot: bool = False, pct_flat: float = 0.95) -> tuple[float, float, float, int]:
	"""Run one synthetic trace simulation and return CSV row values."""
	avg_thickness = float(np.random.uniform(low=35, high=60))
	low = float(np.random.uniform(low=3, high=8))
	high = float(np.random.uniform(low=25, high=60))

	# set up layer boundaries at 10x sampling
	trace = np.zeros(10000, dtype=np.float32)
	thickness = np.random.normal(loc=avg_thickness, scale=7.0, size=10000).astype(np.int16)
	boundaries = np.hstack((0, thickness)).cumsum()
	boundaries = boundaries[boundaries < trace.size]
	sign = np.random.choice([-1.0, 1.0], size=boundaries.size)
	ampl = np.random.exponential(scale=1000, size=boundaries.size) * sign
	trace[boundaries] = ampl

	# bandpass the trace using butterworth filter
	dt = 0.004 / 10.0
	fs = 1.0 / dt
	b, a = butter(4, [low, high], btype="bandpass", fs=fs)

	# apply a flat Hanning taper to reduce edge effects before zero-phase filtering
	trace_tapered = trace * flat_hanning_taper(trace.size, pct_flat=pct_flat)
	trace_f = filtfilt(b, a, trace_tapered)
	trace_f_d = trace_f[::10]

	peaks, troughs, peak_trough_total = count_peaks_troughs(trace_f_d)

	if plot:
		plot_data = trace_f_d
		std = float(np.std(plot_data))
		plot_data_z = plot_data / std if std > 0.0 else plot_data
		stats_text = (
			"min: {mn:.3f}\n"
			"mean: {mu:.3f}\n"
			"max: {mx:.3f}\n"
			"std: {sd:.3f}\n"
			"skew: {sk:.3f}\n"
			"kurtosis: {ku:.3f}\n"
			"peaks: {pk}\n"
			"troughs: {tr}\n"
			"peaks+troughs: {pt}"
		).format(
			mn=float(np.min(plot_data_z)),
			mu=float(np.mean(plot_data_z)),
			mx=float(np.max(plot_data_z)),
			sd=float(np.std(plot_data_z)),
			sk=float(skew(plot_data_z, bias=False)),
			ku=float(kurtosis(plot_data_z, fisher=True, bias=False)),
			pk=peaks,
			tr=troughs,
			pt=peak_trough_total,
		)

		fig, ax = plt.subplots(num=1, figsize=(10, 5))
		ax.grid(True)
		ax.plot(plot_data)
		ax.text(
			0.02,
			0.98,
			stats_text,
			transform=ax.transAxes,
			va="top",
			ha="left",
			bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
		)
		fig.savefig("trace_exponential_rfc.png", dpi=300)
		plt.show()

	return avg_thickness, low, high, peak_trough_total


def run_batch(n_runs: int = 200) -> None:
	"""Run multiple no-plot simulations and append rows to CSV."""
	out_csv = Path.cwd() / "trace_exponential_rfc_stats.csv"
	file_exists = out_csv.exists()

	with out_csv.open("a", newline="") as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["avg_thickness", "low", "high", "peaks_plus_troughs"])
		for _ in range(n_runs):
			avg_thickness, low, high, peak_trough_total = simulate_one(plot=False)
			writer.writerow([avg_thickness, low, high, peak_trough_total])

	print(f"Appended {n_runs} rows to {out_csv}")


if __name__ == "__main__":
	run_batch(n_runs=200)
