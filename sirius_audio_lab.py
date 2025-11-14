import tkinter as tk
from tkinter import ttk
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SiriusAudioLab:
    """Main application class for Sirius Audio Lab."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("🎵 Sirius Audio Lab")
        self.root.geometry("1200x800")

        # Audio parameters
        self.sample_rate = 44_100
        self.blocksize = 1024
        self.channels = 1
        self.audio_stream: Optional[sd.InputStream] = None
        self.recording = False
        self.audio_data: list[float] = []
        self.file_counter = 1

        # Analysis data
        self.current_freq = 0.0
        self.current_note = "Silence"
        self.portal_strength = 0.0
        self.freq_history: deque[float] = deque(maxlen=100)

        # Portal frequency metadata
        self.portal_freqs = [432, 528, 639, 741, 852]
        self.portal_names = ["Heart", "DNA Repair", "Connection", "Awakening", "Spiritual"]

        self.setup_gui()
        self.start_audio_stream()

    def setup_gui(self) -> None:
        """Configure the main graphical interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info_frame = ttk.LabelFrame(main_frame, text="Audio Analysis", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.freq_label = ttk.Label(info_frame, text="Frequency: 0 Hz", font=("Arial", 12))
        self.freq_label.pack(side=tk.LEFT, padx=20)

        self.note_label = ttk.Label(info_frame, text="Note: Silence", font=("Arial", 12))
        self.note_label.pack(side=tk.LEFT, padx=20)

        self.portal_label = ttk.Label(info_frame, text="Portal: None", font=("Arial", 12))
        self.portal_label.pack(side=tk.LEFT, padx=20)

        btn_frame = ttk.Frame(info_frame)
        btn_frame.pack(side=tk.RIGHT)

        self.record_btn = ttk.Button(btn_frame, text="⏺ Start Recording", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="⚙️ Settings", command=self.show_settings).pack(side=tk.LEFT, padx=5)

        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        self.setup_spectrogram(viz_frame)
        self.setup_cube_visualization(viz_frame)

    def setup_spectrogram(self, parent: ttk.Frame) -> None:
        """Create spectrogram plots."""
        spec_frame = ttk.LabelFrame(parent, text="Spectrum Analysis", padding=10)
        spec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, spec_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax1.set_title("Real-time Spectrum")
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude")
        self.ax1.set_xlim(0, 2000)
        self.ax1.set_ylim(0, 1)

        self.ax2.set_title("Frequency History")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Frequency (Hz)")
        self.ax2.set_ylim(0, 1000)

        (self.spectrum_line,) = self.ax1.plot([], [], "b-", linewidth=2)
        (self.freq_line,) = self.ax2.plot([], [], "r-", linewidth=2)

    def setup_cube_visualization(self, parent: ttk.Frame) -> None:
        """Create pseudo 3D cube visualization."""
        cube_frame = ttk.LabelFrame(parent, text="CodexCube 9×9×9 Visualization", padding=10)
        cube_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.cube_canvas = tk.Canvas(cube_frame, bg="black", width=400, height=400)
        self.cube_canvas.pack(fill=tk.BOTH, expand=True)

        portal_info = ttk.Label(cube_frame, text="Portal Frequencies:", font=("Arial", 10))
        portal_info.pack(pady=5)

        self.portal_text = tk.Text(cube_frame, height=6, width=50, font=("Arial", 9))
        self.portal_text.pack(fill=tk.X, pady=5)
        self.update_portal_info()

    def update_portal_info(self) -> None:
        info = "🌌 PORTAL FREQUENCIES 🌌\n\n"
        for freq, name in zip(self.portal_freqs, self.portal_names):
            info += f"• {freq} Hz - {name}\n"
        info += f"\nCurrent: {self.current_freq:.1f} Hz - {self.current_note}"

        self.portal_text.delete(1.0, tk.END)
        self.portal_text.insert(1.0, info)

    def start_audio_stream(self) -> None:
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=self.channels,
                callback=self.audio_callback,
            )
            self.audio_stream.start()
            print("✅ Audio stream started")
        except Exception as exc:  # pragma: no cover - GUI feedback
            print(f"❌ Audio stream error: {exc}")

    def audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"Audio status: {status}")

        audio_chunk = indata[:, 0]
        freq = self.analyze_frequency(audio_chunk)

        self.current_freq = freq
        self.current_note = self.freq_to_note(freq)
        self.portal_strength = self.calculate_portal_strength(freq)
        self.freq_history.append(freq)

        if self.recording:
            self.audio_data.extend(audio_chunk.tolist())

        self.root.after(0, self.update_gui)

    def analyze_frequency(self, audio_chunk: np.ndarray) -> float:
        if np.max(np.abs(audio_chunk)) < 0.01:
            return 0.0

        window = np.hanning(len(audio_chunk))
        windowed = audio_chunk * window

        spectrum = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1 / self.sample_rate)
        magnitudes = np.abs(spectrum)

        min_freq_idx = int(50 * len(freqs) / (self.sample_rate / 2))
        if len(magnitudes[min_freq_idx:]) > 0:
            dominant_idx = np.argmax(magnitudes[min_freq_idx:]) + min_freq_idx
            return float(freqs[dominant_idx])
        return 0.0

    def freq_to_note(self, freq: float) -> str:
        if freq < 50:
            return "Silence"

        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        A4 = 440

        n = 12 * np.log2(freq / A4) + 69
        note_index = int(round(n)) % 12
        octave = int(round(n)) // 12 - 1

        return f"{note_names[note_index]}{octave}"

    def calculate_portal_strength(self, freq: float) -> float:
        if freq == 0:
            return 0.0

        min_diff = min(abs(freq - pf) for pf in self.portal_freqs)
        return max(0.0, 1 - min_diff / 50)

    def update_gui(self) -> None:
        self.freq_label.config(text=f"Frequency: {self.current_freq:.1f} Hz")
        self.note_label.config(text=f"Note: {self.current_note}")

        if self.current_freq > 0:
            closest_idx = int(np.argmin([abs(self.current_freq - pf) for pf in self.portal_freqs]))
            closest_portal = self.portal_names[closest_idx]
            strength = self.portal_strength
            self.portal_label.config(text=f"Portal: {closest_portal} ({strength:.2f})")
        else:
            self.portal_label.config(text="Portal: None")

        self.update_plots()
        self.draw_cube()
        self.update_portal_info()

    def update_plots(self) -> None:
        self.ax1.clear()
        self.ax2.clear()

        freqs = np.linspace(0, 2000, 100)
        if self.current_freq > 0:
            spectrum = np.sin(freqs * self.current_freq / 1000)
        else:
            spectrum = np.zeros(100)

        self.ax1.plot(freqs, spectrum, "b-", linewidth=2)
        self.ax1.set_title("Real-time Spectrum")
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("Magnitude")
        self.ax1.set_xlim(0, 2000)
        self.ax1.set_ylim(0, 1)

        if self.freq_history:
            times = range(len(self.freq_history))
            self.ax2.plot(times, list(self.freq_history), "r-", linewidth=2)
            self.ax2.set_title("Frequency History")
            self.ax2.set_xlabel("Time")
            self.ax2.set_ylabel("Frequency (Hz)")
            self.ax2.set_ylim(0, 1000)

        self.canvas.draw()

    def draw_cube(self) -> None:
        self.cube_canvas.delete("all")

        width = self.cube_canvas.winfo_width()
        height = self.cube_canvas.winfo_height()

        if width <= 1 or height <= 1:
            return

        center_x, center_y = width // 2, height // 2
        size = min(width, height) * 0.3

        color = self.calculate_cube_color()
        self.draw_pseudo_3d_cube(center_x, center_y, size, color)

    def calculate_cube_color(self) -> str:
        if self.current_freq == 0:
            return "#333333"

        base = int(50 + min(self.current_freq / 1000, 1) * 205)
        portal_boost = int(self.portal_strength * 200)

        red = min(base + portal_boost, 255)
        green = int(base * 0.5)
        blue = 255 - base // 2

        return f"#{red:02x}{green:02x}{blue:02x}"

    def draw_pseudo_3d_cube(self, cx: float, cy: float, size: float, color: str) -> None:
        front_points = [
            cx - size,
            cy - size,
            cx + size,
            cy - size,
            cx + size,
            cy + size,
            cx - size,
            cy + size,
        ]

        side_shift = size * 0.7
        side_points = [
            cx + size,
            cy - size,
            cx + size + side_shift,
            cy - size - side_shift,
            cx + size + side_shift,
            cy + size - side_shift,
            cx + size,
            cy + size,
        ]

        top_points = [
            cx - size,
            cy - size,
            cx,
            cy - size - side_shift,
            cx + size + side_shift,
            cy - size - side_shift,
            cx + size,
            cy - size,
        ]

        self.cube_canvas.create_polygon(front_points, fill=color, outline="white", width=2)
        self.cube_canvas.create_polygon(side_points, fill=self.adjust_brightness(color, -30), outline="white", width=1)
        self.cube_canvas.create_polygon(top_points, fill=self.adjust_brightness(color, 20), outline="white", width=1)

        self.draw_cube_grid(cx, cy, size, color)

    def draw_cube_grid(self, cx: float, cy: float, size: float, color: str) -> None:
        grid_color = self.adjust_brightness(color, 50)

        for i in range(1, 9):
            pos = -size + (i * size * 2 / 9)

            self.cube_canvas.create_line(cx - size, cy + pos, cx + size, cy + pos, fill=grid_color, width=1)
            self.cube_canvas.create_line(cx + pos, cy - size, cx + pos, cy + size, fill=grid_color, width=1)

    def adjust_brightness(self, color: str, amount: int) -> str:
        # Placeholder implementation – production code should modify the colour value.
        return color

    def toggle_recording(self) -> None:
        self.recording = not self.recording

        if self.recording:
            self.record_btn.config(text="⏹ Stop Recording")
            self.audio_data = []
            print("🎙️ Recording started...")
        else:
            self.record_btn.config(text="⏺ Start Recording")
            self.save_recording()
            print("💾 Recording saved")

    def save_recording(self) -> None:
        if self.audio_data:
            filename = f"sirius_audio_{self.file_counter}.wav"
            audio_array = np.array(self.audio_data, dtype=np.float32)
            sf.write(filename, audio_array, self.sample_rate)
            print(f"✅ Saved: {filename}")
            self.file_counter += 1

    def show_settings(self) -> None:
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Audio Settings")
        settings_window.geometry("300x200")

        ttk.Label(settings_window, text="Audio Device:").pack(pady=5)
        devices = sd.query_devices()
        device_names = [
            f"{device['name']} ({device['max_input_channels']}ch)"
            for device in devices
            if device["max_input_channels"] > 0
        ]

        device_var = tk.StringVar()
        device_combo = ttk.Combobox(settings_window, textvariable=device_var, values=device_names, width=40)
        device_combo.pack(pady=5)

        ttk.Button(settings_window, text="Apply", command=lambda: self.apply_settings(device_var.get())).pack(pady=10)

    def apply_settings(self, device_name: str) -> None:
        print(f"Applying audio device: {device_name}")

    def __del__(self) -> None:
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()


def main() -> None:
    root = tk.Tk()
    app = SiriusAudioLab(root)
    root.mainloop()


if __name__ == "__main__":
    main()
