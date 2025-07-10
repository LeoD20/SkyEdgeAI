import tkinter as tk
import tkinter.ttk as ttk  # for Notebook
from tkinter import messagebox
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from ultralytics import *
from environment import *
from channel_model import *
from task_managment import *
import random
import contextlib
import sys
import logging
from tqdm import tqdm
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SimulatorGUI:
    def __init__(self, root):
        self.active_figure = None
        self.result_categories = None
        self.result_buttons = None
        self.result_display = None
        self.user_energy_array = None
        self.uav_energy_array = None
        self.current_step = 0
        self.root = root
        self.root.title("UAV Simulator")
        self.root.configure(bg='white')

        self.input_list = []
        self.input_sizes = []
        self.processing_times = []
        self.output_sizes = []

        self.uav = None
        self.users = []

        self.uav_positions = []

        self.energy_vect = []
        self.snr_values = []
        self.completed_tasks = []
        self.dropped_tasks = []
        self.total_tasks = []
        self.bandwidth_utilization = []

        self.generated_tasks = []
        self.offloaded_tasks = []
        self.local_tasks = []
        self.completed_local = []
        self.completed_remote = []
        self.dropped_local = []
        self.dropped_remote = []
        self.fairness = []
        self.energy_saved_all = []

        self.latency_local_values = []
        self.latency_remote_values = []

        self.user_energy_all = []

        self.avg_battery_usage = []

        self.threshold_snr = -9.478

        self.results = None

        self.model = YOLO("yolov3.pt")  # Load YOLOv3 model
        logging.getLogger("ultralytics").setLevel(logging.CRITICAL)  # Disable printing output
        tqdm.disable = True

        # Center the window
        window_width = 1200
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(window_width, window_height)  # Prevent resizing below this

        # ===== TITLE (always visible) =====
        self.title_label = tk.Label(root, text="SkyEdgeAI", font=("Helvetica", 24, "bold"), bg="white")
        self.title_label.pack(pady=(30, 10))

        # ===== INITIAL BUTTONS =====
        self.start_btn = tk.Button(root, text="Start New Simulation", font=("Helvetica", 14), width=25, command=self.start_simulation)
        self.start_btn.pack(pady=(10, 5))

        self.load_btn = tk.Button(root, text="Load Simulation Data", font=("Helvetica", 14), width=25, command=self.load_results)
        self.load_btn.pack(pady=(5, 30))

        # ===== CLOSE BUTTON (always visible) =====
        self.close_btn = tk.Button(root, text="Close", font=("Helvetica", 12), command=self.root.destroy)
        self.close_btn.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

        # ===== CONFIRM BUTTON (hidden initially) =====
        # Frame to hold action buttons
        self.button_frame = tk.Frame(root, bg='white')

        self.save_config_btn = tk.Button(self.button_frame, text="Save to JSON", font=("Helvetica", 12),
                                  command=self.save_config_to_json)
        self.load_config_json_btn = tk.Button(self.button_frame, text="Load from JSON", font=("Helvetica", 12),
                                       command=self.load_config_from_json)
        self.confirm_btn = tk.Button(self.button_frame, text="Confirm Parameters", font=("Helvetica", 12),
                                     command=self.confirm_parameters)
        self.run_btn = tk.Button(
            self.button_frame,
            text="▶ Run",
            font=("Helvetica", 12, "bold"),
            fg="white",
            bg="green",
            command=self.run_simulation  # You can define this method as needed
        )

        self.progressbar = ttk.Progressbar(self.root, mode="determinate", length=400)
        # Inference mode variable (calculate is default)
        self.inference_mode = tk.StringVar(value="calculate")

        # ===== Parameters and Entry Tracking =====
        self.param_entries = {}
        self.updated_params = None

        self.inference_mode = tk.StringVar(value="calculate")  # Default
        self.inference_settings = {
            "mode": "calculate",
            "input_json_filename": "",
            "image_folder_path": "",
            "output_json_filename": ""
        }

        self.parameters = {
            "UAV Parameters": {
                "Speed [m/s]": 5,
                "Altitude [m]": 50,
                "Radius [m]": 75,
                "Mass [kg]": 4,
                "Battery Capacity [kJ]": 300,
                "Computational Frequency [GHz]": 4.8,
                "Power Usage [W]": 5,
                "Processing Storage [Gb]": 16,
                "Queue Storage [Tb]": 8,
                "Processing Units [Unit]": 2
            },
            "Local Parameters": {
                "Battery Capacity [kJ]": 40,
                "Computational Frequency [GHz]": 1.5,
                "Power Usage [W]": 2,
                "Processing Storage [Mb]": 800,
                "Local Queue Storage [Gb]": 8,
                "Offloader Queue Storage [Gb]": 8
            },
            "Task Parameters": {
                "Deadline [s]": 3,
                "Latency Sensitive Deadline [s]": 1.2,
                "Latency Threshold [s]": 1.5,
                "Energy Threshold [J]": 2.5,
            },
            "Channel Parameters": {
                "Total Bandwidth [MHz]": 400,
                "Noise Power Density [10log(mW/Hz)]": -143,
                "Carrier Frequency [GHz]": 28,
                "Rician K-Factor [dB]": 10,
                "Transmission Power [W]": 1,
            },
            "Simulation Parameters": {
                "Number of Users": 10,
                "Total Time [s]": 120,
                "Step Size [ms]": 1,
                "Area Size [m]": 300,
                "Generation probability [Task/s]": 1,
            }
        }

        # UAV Parameters
        self.uav_speed = self.parameters["UAV Parameters"]["Speed [m/s]"]
        self.radius = self.parameters["UAV Parameters"]["Radius [m]"]
        self.uav_altitude = self.parameters["UAV Parameters"]["Altitude [m]"]
        self.uav_mass = self.parameters["UAV Parameters"]["Mass [kg]"]
        self.uav_available_energy = self.parameters["UAV Parameters"]["Battery Capacity [kJ]"] * 1e3
        self.f_uav = self.parameters["UAV Parameters"]["Computational Frequency [GHz]"] * 1e9
        self.uav_processing_storage = self.parameters["UAV Parameters"]["Processing Storage [Gb]"] * 1e9
        self.uav_queue_storage = self.parameters["UAV Parameters"]["Queue Storage [Tb]"] * 1e12
        self.processors = self.parameters["UAV Parameters"]["Processing Units [Unit]"]
        self.power_uav = self.parameters["UAV Parameters"]["Power Usage [W]"] * self.processors

        # Local Parameters
        self.user_available_energy = self.parameters["Local Parameters"]["Battery Capacity [kJ]"] * 1e3
        self.f_user = self.parameters["Local Parameters"]["Computational Frequency [GHz]"] * 1e9
        self.power_user = self.parameters["Local Parameters"]["Power Usage [W]"]
        self.local_processing_storage = self.parameters["Local Parameters"]["Processing Storage [Mb]"] * 1e6
        self.local_queue_storage = self.parameters["Local Parameters"]["Local Queue Storage [Gb]"] * 1e9
        self.offloader_queue_storage = self.parameters["Local Parameters"]["Offloader Queue Storage [Gb]"] * 1e9

        # Task Parameters
        self.latency_deadline = self.parameters["Task Parameters"]["Deadline [s]"]
        self.latency_sensitive_deadline = self.parameters["Task Parameters"]["Latency Sensitive Deadline [s]"]
        self.latency_threshold = self.parameters["Task Parameters"]["Latency Threshold [s]"]
        self.energy_threshold = self.parameters["Task Parameters"]["Energy Threshold [J]"]

        # Channel Parameters
        self.total_bandwidth = self.parameters["Channel Parameters"]["Total Bandwidth [MHz]"] * 1e6
        self.noise_power_density = 10**((self.parameters["Channel Parameters"]["Noise Power Density [10log(mW/Hz)]"]/10)-3)
        self.carrier_freq = self.parameters["Channel Parameters"]["Carrier Frequency [GHz]"] * 1e9
        self.K_factor_constant = 10**(self.parameters["Channel Parameters"]["Rician K-Factor [dB]"]/10)
        self.tx_power = self.parameters["Channel Parameters"]["Transmission Power [W]"]

        # Simulation Parameters
        self.N_users = self.parameters["Simulation Parameters"]["Number of Users"]
        self.total_time = self.parameters["Simulation Parameters"]["Total Time [s]"]
        self.time_step = self.parameters["Simulation Parameters"]["Step Size [ms]"]*1e-3
        self.area_size = self.parameters["Simulation Parameters"]["Area Size [m]"]
        self.generation_probability = self.parameters["Simulation Parameters"]["Generation probability [Task/s]"] * self.time_step
        self.uav_speed_ratio = self.processors * self.f_uav / self.f_user
        self.bandwidth = self.total_bandwidth / (self.N_users + 1)
        self.noise_power = self.noise_power_density * self.bandwidth
        self.center = [self.area_size/2, self.area_size / 2]
        self.sim_steps = 1 + (self.total_time // self.time_step)

        # Inference
        self.inference_mode_param = self.inference_settings["mode"]
        self.input_filename = self.inference_settings["input_json_filename"]
        self.image_folder_path = self.inference_settings["image_folder_path"]
        self.output_filename = self.inference_settings["output_json_filename"]


        self.output_json_entry = None
        self.image_folder_entry = None
        self.precalc_entry = None

        self.progress_window = None

        self.results_frame = None

        # Notebook (tab interface) hidden initially
        self.notebook = ttk.Notebook(root)

    def start_simulation(self):
        # Hide initial buttons
        self.start_btn.pack_forget()
        self.load_btn.pack_forget()

        # Build tabbed parameter interface
        self.param_entries = {}
        for category, params in self.parameters.items():
            # Create a scrollable canvas
            container = tk.Frame(self.notebook, bg="white")
            canvas = tk.Canvas(container, bg="white", highlightthickness=0)
            scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="white")

            scrollable_frame.bind(
                "<Configure>",
                lambda e, c=canvas: c.configure(scrollregion=c.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            self.notebook.add(container, text=category)
            self.param_entries[category] = {}

            # Add parameters to the scrollable frame
            for i, (label_text, default_val) in enumerate(params.items()):
                label = tk.Label(scrollable_frame, text=label_text, font=("Helvetica", 12), bg='white')
                label.grid(row=i, column=0, sticky='w', pady=5, padx=10)

                entry = tk.Entry(scrollable_frame, font=("Helvetica", 12))
                entry.insert(0, str(default_val))
                entry.grid(row=i, column=1, pady=5, padx=10)

                self.param_entries[category][label_text] = entry

        # === Inference Settings Tab ===
        inference_frame = tk.Frame(self.notebook, bg="white")
        self.notebook.add(inference_frame, text="Inference")

        # Label for the AI model used
        tk.Label(inference_frame, text="AI Model Used: YOLOv3", font=("Helvetica", 12, "bold"), bg="white") \
            .grid(row=0, column=0, columnspan=2, sticky='w', pady=10, padx=10)

        # Radio buttons
        tk.Label(inference_frame, text="Inference Mode:", font=("Helvetica", 12, "bold"), bg="white") \
            .grid(row=1, column=0, sticky='w', padx=10)

        tk.Radiobutton(inference_frame, text="Precalculated Inferences", variable=self.inference_mode,
                       value="precalculated", font=("Helvetica", 12), bg='white',
                       command=self.toggle_inference_inputs) \
            .grid(row=2, column=0, sticky='w', padx=20, pady=2)

        tk.Radiobutton(inference_frame, text="Calculate Inferences", variable=self.inference_mode,
                       value="calculate", font=("Helvetica", 12), bg='white',
                       command=self.toggle_inference_inputs) \
            .grid(row=3, column=0, sticky='w', padx=20, pady=2)

        # Entry for a precalculated JSON file path
        tk.Label(inference_frame, text="Precalculated File (.json):", font=("Helvetica", 10), bg='white') \
            .grid(row=2, column=1, sticky='w', padx=10)
        self.precalc_entry = tk.Entry(inference_frame, font=("Helvetica", 10), width=40)
        self.precalc_entry.grid(row=2, column=2, padx=10)
        self.precalc_entry.configure(state='disabled')  # Initially disabled

        # Entry for the image folder path
        tk.Label(inference_frame, text="Image Folder Path:", font=("Helvetica", 10), bg='white') \
            .grid(row=3, column=1, sticky='w', padx=10)
        self.image_folder_entry = tk.Entry(inference_frame, font=("Helvetica", 10), width=40)
        self.image_folder_entry.grid(row=3, column=2, padx=10)
        self.image_folder_entry.configure(state='normal')  # Default is calculate

        # Entry for output JSON filename
        tk.Label(inference_frame, text="Save Inference Results As (.json):", font=("Helvetica", 10), bg='white') \
            .grid(row=4, column=1, sticky='w', padx=10)
        self.output_json_entry = tk.Entry(inference_frame, font=("Helvetica", 10), width=40)
        self.output_json_entry.grid(row=4, column=2, padx=10)
        self.output_json_entry.configure(state='normal')  # Initially enabled for default mode

        self.notebook.pack(pady=10, expand=True, fill='both')
        self.button_frame.pack(pady=10)

        self.save_config_btn.pack(side='left', padx=10)
        self.load_config_json_btn.pack(side='left', padx=10)
        self.confirm_btn.pack(side='left', padx=10)
        self.run_btn.pack(side='left', padx=10)

    def confirm_parameters(self):
        updated_params = {}

        for category, entry_dict in self.param_entries.items():
            updated_params[category] = {}
            for param_name, entry_widget in entry_dict.items():
                value_str = entry_widget.get()
                try:
                    # Try to convert to float or int
                    value = float(value_str)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    value = value_str
                updated_params[category][param_name] = value

        self.updated_params = True
        self.parameters = updated_params

        # Save inference settings
        self.inference_settings["mode"] = self.inference_mode.get()
        self.inference_settings[
            "input_json_filename"] = self.precalc_entry.get() + ".json" if self.inference_mode.get() == "precalculated" else ""
        self.inference_settings[
            "image_folder_path"] = self.image_folder_entry.get() if self.inference_mode.get() == "calculate" else ""
        self.inference_settings["output_json_filename"] = (
            self.output_json_entry.get() + ".json" if self.inference_mode.get() == "calculate" else ""
        )

        # Extract and validate critical parameters
        step_size = updated_params["Simulation Parameters"]["Step Size [ms]"]
        task_prob = updated_params["Simulation Parameters"]["Generation probability [Task/s]"]

        if step_size > 1:
            messagebox.showwarning("Invalid Step Size", "❗ Step Size [ms] must not exceed 1.")
            self.updated_params = False
            return

        if task_prob > 3:
            messagebox.showwarning("Invalid Task Generation Rate",
                                   "❗ Generation probability [Task/s] must not exceed 3.")
            self.updated_params = False
            return

        if not self.inference_settings["image_folder_path"] and self.inference_settings["mode"] == "calculate":
            messagebox.showwarning("Missing image folder path",
                                   "❗ The image folder path must not be empty.")
            self.updated_params = False
            return

        if self.inference_settings["input_json_filename"] == ".json" and self.inference_settings["mode"] == "precalculated":
            messagebox.showwarning("Missing path for loading JSON",
                                   "❗ Input JSON Filename must not be empty.")
            self.updated_params = False
            return

        # UAV Parameters
        self.uav_speed = self.parameters["UAV Parameters"]["Speed [m/s]"]
        self.radius = self.parameters["UAV Parameters"]["Radius [m]"]
        self.uav_altitude = self.parameters["UAV Parameters"]["Altitude [m]"]
        self.uav_mass = self.parameters["UAV Parameters"]["Mass [kg]"]
        self.uav_available_energy = self.parameters["UAV Parameters"]["Battery Capacity [kJ]"] * 1e3
        self.f_uav = self.parameters["UAV Parameters"]["Computational Frequency [GHz]"] * 1e9
        self.uav_processing_storage = self.parameters["UAV Parameters"]["Processing Storage [Gb]"] * 1e9
        self.uav_queue_storage = self.parameters["UAV Parameters"]["Queue Storage [Tb]"] * 1e12
        self.processors = self.parameters["UAV Parameters"]["Processing Units [Unit]"]
        self.power_uav = self.parameters["UAV Parameters"]["Power Usage [W]"] * self.processors

        # Local Parameters
        self.user_available_energy = self.parameters["Local Parameters"]["Battery Capacity [kJ]"] * 1e3
        self.f_user = self.parameters["Local Parameters"]["Computational Frequency [GHz]"] * 1e9
        self.power_user = self.parameters["Local Parameters"]["Power Usage [W]"]
        self.local_processing_storage = self.parameters["Local Parameters"]["Processing Storage [Mb]"] * 1e6
        self.local_queue_storage = self.parameters["Local Parameters"]["Local Queue Storage [Gb]"] * 1e9
        self.offloader_queue_storage = self.parameters["Local Parameters"]["Offloader Queue Storage [Gb]"] * 1e9

        # Task Parameters
        self.latency_deadline = self.parameters["Task Parameters"]["Deadline [s]"]
        self.latency_sensitive_deadline = self.parameters["Task Parameters"]["Latency Sensitive Deadline [s]"]
        self.latency_threshold = self.parameters["Task Parameters"]["Latency Threshold [s]"]
        self.energy_threshold = self.parameters["Task Parameters"]["Energy Threshold [J]"]

        # Channel Parameters
        self.total_bandwidth = self.parameters["Channel Parameters"]["Total Bandwidth [MHz]"] * 1e6
        self.noise_power_density = 10 ** (
                    (self.parameters["Channel Parameters"]["Noise Power Density [10log(mW/Hz)]"] / 10) - 3)
        self.carrier_freq = self.parameters["Channel Parameters"]["Carrier Frequency [GHz]"] * 1e9
        self.K_factor_constant = 10 ** (self.parameters["Channel Parameters"]["Rician K-Factor [dB]"] / 10)
        self.tx_power = self.parameters["Channel Parameters"]["Transmission Power [W]"]

        # Simulation Parameters
        self.N_users = self.parameters["Simulation Parameters"]["Number of Users"]
        self.total_time = self.parameters["Simulation Parameters"]["Total Time [s]"]
        self.time_step = self.parameters["Simulation Parameters"]["Step Size [ms]"] * 1e-3
        self.area_size = self.parameters["Simulation Parameters"]["Area Size [m]"]
        self.generation_probability = self.parameters["Simulation Parameters"][
                                 "Generation probability [Task/s]"] * self.time_step
        self.uav_speed_ratio = self.processors * self.f_uav / self.f_user
        self.bandwidth = self.total_bandwidth / (self.N_users + 1)
        self.noise_power = self.noise_power_density * self.bandwidth
        self.center = [self.area_size/2, self.area_size / 2]
        self.sim_steps = 1 + (self.total_time // self.time_step)

        # Inference
        self.inference_mode_param = self.inference_settings["mode"]
        self.input_filename = self.inference_settings["input_json_filename"]
        self.image_folder_path = self.inference_settings["image_folder_path"]
        self.output_filename = self.inference_settings["output_json_filename"]

    def save_config_to_json(self):
        if not self.updated_params:
            messagebox.showinfo("No Data", "Please confirm parameters before saving.")
            return
        self.updated_params = False
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, "w") as f:
                json.dump(self.parameters, f, indent=4)
            messagebox.showinfo("Success", f"Parameters saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def load_config_from_json(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, "r") as f:
                loaded_params = json.load(f)

            # Update fields in GUI
            for category, param_dict in loaded_params.items():
                for param_name, value in param_dict.items():
                    if category in self.param_entries and param_name in self.param_entries[category]:
                        self.param_entries[category][param_name].delete(0, tk.END)
                        self.param_entries[category][param_name].insert(0, str(value))

            self.confirm_parameters()
            messagebox.showinfo("Loaded", f"Parameters loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def run_simulation(self):
        if not self.updated_params:
            messagebox.showwarning("Parameters Not Confirmed",
                                   "Please confirm parameters before running the simulation.")
            return

        if not os.path.exists(self.inference_settings["input_json_filename"]) and self.inference_settings["mode"] == "precalculated":
            messagebox.showwarning("File doesn't exist",
                                   "Please enter a valid file path.")
            self.updated_params = False
            return

        elif os.path.exists(self.inference_settings["input_json_filename"]) and self.inference_settings["mode"] == "precalculated":
            with open(self.inference_settings["input_json_filename"], "r") as f:
                data = json.load(f)
                self.input_list = data["images"]
                self.input_sizes = data["input_sizes"]
                self.output_sizes = data["output_sizes"]
                self.processing_times = data["processing_times"]


        if not os.path.isdir(self.inference_settings["image_folder_path"]) and self.inference_settings["mode"] == "calculate":
            messagebox.showwarning("Directory doesn't exist",
                                   "Please enter a valid directory path.")
            self.updated_params = False
            return

        elif os.path.isdir(self.inference_settings["image_folder_path"]) and self.inference_settings["mode"] == "calculate":
            messagebox.showinfo("Inherence Setup", "Setting up the inherence data.")
            self.input_list = [f for f in os.listdir(self.inference_settings["image_folder_path"]) if f.lower().endswith('.jpg')]
            # Create a popup window with progress bar
            self.progress_window = tk.Toplevel(self.root)
            self.progress_window.title("Running Inference")
            self.progress_window.geometry("400x100")
            self.progress_window.grab_set()  # Make it modal
            self.progress_window.resizable(False, False)

            progress_label = tk.Label(self.progress_window, text="Running inference...", font=("Helvetica", 12))
            progress_label.pack(pady=(10, 5))

            self.progressbar = ttk.Progressbar(self.progress_window, mode="determinate", length=300)
            self.progressbar["maximum"] = len(self.input_list)
            self.progressbar.pack(pady=5)

            self.current_step = 0

            self.inference_loop()
            if self.inference_settings["output_json_filename"] != ".json":
                if os.path.exists(self.inference_settings["output_json_filename"]) and self.inference_settings[
                    "mode"] == "calculate":
                    overwrite = messagebox.askyesno(
                        title="File already exists",
                        message=f"Do you want to overwrite {self.inference_settings["output_json_filename"]}?\nOnly clicking 'Yes' will proceed."
                    )
                    if not overwrite:
                        self.updated_params = False
                        return  # User said no, abort the operation
                else:
                    data = {
                        "images": self.input_list,
                        "input_sizes": self.input_izes,
                        "output_sizes": self.output_sizes,
                        "processing_times": self.processing_times
                    }

                    with open(self.inference_settings["output_json_filename"], "w") as f:
                        json.dump(data, f, indent=4)

        messagebox.showinfo("Run", "🚀 Simulation started with the confirmed parameters.")
        self.initialize_uav_users()

        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Running Simulation")
        self.progress_window.geometry("400x100")
        self.progress_window.grab_set()  # Make it modal
        self.progress_window.resizable(False, False)

        progress_label = tk.Label(self.progress_window, text="Running Simulation...", font=("Helvetica", 12))
        progress_label.pack(pady=(10, 5))

        self.progressbar = ttk.Progressbar(self.progress_window, mode="determinate", length=300)
        self.progressbar["maximum"] = self.sim_steps
        self.progressbar.pack(pady=5)

        self.current_step = 0

        self.simulation_loop()

    def toggle_inference_inputs(self):
        mode = self.inference_mode.get()

        if mode == "precalculated":
            self.precalc_entry.configure(state='normal')
            self.image_folder_entry.delete(0, tk.END)
            self.image_folder_entry.configure(state='disabled')
            self.output_json_entry.delete(0, tk.END)
            self.output_json_entry.configure(state='disabled')
        else:
            self.image_folder_entry.configure(state='normal')
            self.precalc_entry.delete(0, tk.END)
            self.precalc_entry.configure(state='disabled')
            self.output_json_entry.configure(state='normal')

    def inference_loop(self):
        for img in self.input_list:
            # Simulate work here (replace with real logic if needed)
            input_size = os.path.getsize(self.inference_settings["image_folder_path"] + "/" + img) * 8
            self.input_sizes.append(input_size)

            # YOLO inference
            inference = self.model(self.inference_settings["image_folder_path"] + "/" + img)

            # Model output and its size
            model_output = (inference[0].boxes, inference[0].names)
            output_size = sum(sys.getsizeof(obj) for obj in model_output) * 8  # bits
            self.output_sizes.append(output_size)

            # Timing
            speed = inference[0].speed
            preprocess_time = speed['preprocess']
            inference_time = speed['inference']
            postprocess_time = speed['postprocess']
            processing_time = (preprocess_time + inference_time + postprocess_time) / 1000.0  # seconds
            self.processing_times.append(processing_time)
            self.current_step += 1
            self.progressbar["value"] = self.current_step
            self.progressbar.update()

        self.progress_window.destroy()
        messagebox.showinfo("Inference", "✅ Inference finished successfully!")

    def initialize_uav_users(self):
        self.uav = UAV(self.center,
                       self.radius,
                       self.uav_altitude,
                       self.uav_speed,
                       self.uav_mass,
                       self.uav_available_energy,
                       self.power_uav,
                       self.uav_speed_ratio,
                       Queue(self.uav_queue_storage),
                       Processor(self.uav_processing_storage,self.time_step),
                       self.f_user,
                       self.time_step)
        self.uav.compute_trajectory(self.time_step)
        self.uav_positions = np.array(self.uav.trajectory)  # all the possible UAV coordinates
        self.users = [User(self.area_size,
                      i,
                      Queue(self.local_queue_storage),
                      Processor(self.local_processing_storage, self.time_step),
                      Offloader(Queue(self.offloader_queue_storage), self.time_step),
                      self.power_user,
                      self.f_user,
                      self.user_available_energy)
                 for i in range(self.N_users)]

    def simulation_loop(self):
        self.energy_vect = []
        self.snr_values = []
        self.completed_tasks = []
        self.dropped_tasks = []
        self.total_tasks = []
        self.bandwidth_utilization = []

        for step in range(int(self.sim_steps)):  # Progress bar here
            self.progressbar["value"] = step
            self.progressbar.update()
            bandwidth_used = 0
            # User side
            for user in self.users:
                # Generate new task
                if np.random.random() < self.generation_probability:
                    user.generated_tasks += 1
                    # Estimate the channel for offloading decision
                    h_total = compute_channel(
                        self.uav_positions, user.position, step,
                        self.time_step * step, self.time_step, self.carrier_freq,
                        self.K_factor_constant
                    )
                    snr = compute_snr(h_total, self.tx_power, self.noise_power)
                    snr_db = 10 * np.log10(snr)
                    self.snr_values.append(snr_db)  # For visualization

                    # Generate task (pattern recognition)
                    img_idx = random.randint(0, len(self.input_list) - 1)
                    task_deadline = random.choices([self.latency_deadline,
                                                    self.latency_sensitive_deadline],
                                                   weights=[0.8, 0.2])[
                        0]
                    task = Task(user, self.input_sizes[img_idx], self.output_sizes[img_idx],
                                self.processing_times[img_idx],
                                step * self.time_step, task_deadline,
                                self.energy_threshold, self.latency_threshold)
                    self.total_tasks.append(task)  # debugging

                    # Decide what to do with the task (local processing, offloading, or reject)
                    user.decide_on_task(task, self.uav, snr_db, self.bandwidth,
                                        self.tx_power, self.dropped_tasks)

                # Offload tasks to UAV
                user.offloader.update_current_time()
                user.offloader.offload_task(self.uav)

                if user.offloader.busy:
                    bandwidth_used += self.bandwidth

                # Process local tasks
                user.processor.update_current_time()
                user.processor.process_task(user, self.completed_tasks, self.dropped_tasks)

                # For visualization
                user.energy_used_over_time.append(user.initial_energy - user.available_energy)


            # UAV side
            self.uav.processor.update_current_time()
            self.uav.processor.process_task(self.uav, self.completed_tasks, self.dropped_tasks)
            # Move UAV
            self.uav.move_drone()

            # For visualization
            self.energy_vect.append(self.uav.available_energy)
            if self.uav.completed_task is not None and self.uav.completed_task.completed_now:
                time_steps_downlink = int(np.ceil(self.uav.completed_task.downlink_latency / self.time_step))
                curr_index = step
                last_index = step + time_steps_downlink
                if last_index <= len(self.bandwidth_utilization):
                    for i in range(curr_index, last_index):
                        self.bandwidth_utilization[i] += (self.bandwidth / self.total_bandwidth)

            self.bandwidth_utilization.append(bandwidth_used / self.total_bandwidth)  # Normalization

            if not self.uav.available or step == self.sim_steps - 1:  # The UAV runs out of battery
                # Drop tasks from uav queue
                while self.uav.queue.queue:
                    task = self.uav.queue.dequeue_task(False)
                    task.user.dropped_remote += 1
                # Drop the task from offloader queue and process current local tasks
                for user in self.users:
                    while user.offloader.queue.queue:
                        task = user.offloader.queue.dequeue_task(False)
                        task.user.dropped_remote += 1
                    task_remaining = True
                    while task_remaining:
                        user.processor.update_current_time()
                        task_remaining = user.processor.process_task(user, self.completed_tasks, self.dropped_tasks)

                if not self.uav.available:
                    messagebox.showinfo("Battery over", "UAV ran out of battery, returning to base station. Stopping simulation.")
                    self.progressbar["value"] = self.sim_steps
                    break
        self.generated_tasks = []
        self.offloaded_tasks = []
        self.local_tasks = []
        self.completed_local = []
        self.completed_remote = []
        self.dropped_local = []
        self.dropped_remote = []
        self.fairness = []
        self.energy_saved_all = []
        self.latency_local_values = []
        self.latency_remote_values = []
        self.user_energy_all = []

        for user in self.users:
            self.generated_tasks.append(user.generated_tasks)
            self.offloaded_tasks.append(user.offloaded_tasks)
            self.local_tasks.append(user.generated_tasks - user.offloaded_tasks)
            self.dropped_local.append(user.dropped_local)
            self.dropped_remote.append(user.dropped_remote)
            self.completed_local.append(user.completed_local)
            self.completed_remote.append(user.completed_remote)
            self.fairness.append(user.completed_remote / user.offloaded_tasks if user.offloaded_tasks > 0 else 0)
            self.energy_saved_all.append(user.energy_saved)
            self.latency_local_values.extend(user.latency_local)
            self.latency_remote_values.extend(user.latency_remote)
            self.user_energy_all.extend(user.computational_energy)

        self.uav_energy_array = np.array(self.uav.computational_energy)
        self.user_energy_array = np.array(self.user_energy_all)
        self.avg_battery_usage = np.mean([user.energy_used_over_time for user in self.users], axis=0)

        x_coords_users = [user.position[0] for user in self.users]
        y_coords_users = [user.position[1] for user in self.users]

        x_coords_circle = [point[0] for point in self.uav.trajectory]
        y_coords_circle = [point[1] for point in self.uav.trajectory]

        self.results = {
            "snr_values": self.snr_values,
            "generated_tasks": self.generated_tasks,
            "offloaded_tasks": self.offloaded_tasks,
            "local_tasks": self.local_tasks,
            "completed_local": self.completed_local,
            "completed_remote": self.completed_remote,
            "dropped_local": self.dropped_local,
            "dropped_remote": self.dropped_remote,
            "fairness": self.fairness,
            "energy_saved_all": self.energy_saved_all,
            "latency_local_values": self.latency_local_values,
            "latency_remote_values": self.latency_remote_values,
            "bandwidth_utilization": self.bandwidth_utilization,
            "avg_battery_usage": self.avg_battery_usage,
            "uav_energy_array": self.uav_energy_array,
            "user_energy_array": self.user_energy_array,
            "energy_vect": self.energy_vect,
            "uav_computational_energy": self.uav.computational_energy,
            "users_positions": [x_coords_users, y_coords_users],
            "drone_positions": [x_coords_circle, y_coords_circle],
            "initial_uav_position": [self.uav.position[0], self.uav.position[1]],
            "area_size": self.area_size,
            "tot_time": self.total_time,
            "time_step": self.time_step
        }
        self.progress_window.destroy()
        self.show_results_page()

        messagebox.showinfo("Run", "✅ Simulation finished successfully!")

    def show_results_page(self):
        # Hide the notebook and buttons
        self.notebook.pack_forget()
        self.button_frame.pack_forget()

        # Create a new frame for results
        self.results_frame = tk.Frame(self.root, bg="white")
        self.results_frame.pack(expand=True, fill='both')

        # Add "Show Results" and "Save Results" buttons
        show_btn = tk.Button(self.results_frame, text="Show Results", font=("Helvetica", 14), command=self.show_results)
        show_btn.pack(pady=(50, 10))

        save_btn = tk.Button(self.results_frame, text="Save Results", font=("Helvetica", 14), command=self.save_results)
        save_btn.pack(pady=10)

        # Repack the close button at the bottom right
        self.close_btn.lift()  # Ensure it's on top
        self.close_btn.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

    def show_results(self):
        if not hasattr(self, "results") or not self.results:
            messagebox.showwarning("No Results", "❗ No results to display.")
            return

        # Destroy old results frame contents
        self.results_frame.destroy()

        # Create a new main frame with sidebar and content
        self.results_frame = tk.Frame(self.root, bg="white")
        self.results_frame.pack(fill="both", expand=True)

        # LEFT: Sidebar for category selection
        sidebar = tk.Frame(self.results_frame, width=200, bg="lightgray")
        sidebar.pack(side="left", fill="y")

        # RIGHT: Main display area
        self.result_display = ttk.Notebook(self.results_frame)
        self.result_display.pack(side="right", fill="both", expand=True)

        # Result categories and corresponding handler function names
        self.result_categories = [
            "System Overview",
            "Drone path",
            "SNR Values",
            "Pie Charts",
            "Latency Distribution",
            "Bandwidth Utilization",
            "Battery Usage",
            "Energy per Task",
            "Jain Index on Energy",
            "Task Completion per User",
            "Offloaded Task Completion",
            "Local Task Completion",
            "Offloading Fairness",
            "UAV Available Energy"
        ]

        self.result_buttons = {}  # Store for styling

        # Collapsible list using buttons
        for category in self.result_categories:
            btn = tk.Button(sidebar, text=category, font=("Helvetica", 10), anchor="w",
                            bg="white", relief="flat", command=lambda c=category: self.load_result_tab(c))
            btn.pack(fill="x", padx=5, pady=2)
            self.result_buttons[category] = btn

        # Automatically load the first category
        self.load_result_tab(self.result_categories[0])
        # Add Reset Simulation button
        reset_btn = tk.Button(
            self.results_frame,
            text="🔄 Reset Simulation",
            font=("Helvetica", 10),
            bg="lightgray",
            command=self.reset_to_start
        )
        reset_btn.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)

        # Repack the Close button in the bottom-right
        self.close_btn.lift()
        self.close_btn.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

    def save_results(self):
        if not hasattr(self, 'results') or not self.results:
            messagebox.showwarning("No Results", "❗ No results to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return  # User cancelled

        try:
            # Convert non-serializable values (e.g., numpy arrays) to lists
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value

            with open(file_path, "w") as f:
                json.dump(serializable_results, f, indent=4)
            messagebox.showinfo("Saved", f"✅ Results saved to:\n{file_path}, now saving the related config file...")
            self.save_config_to_json()
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to save results:\n{e}")

    def load_results(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert lists back to np.array where needed (optional)
            for key, val in data.items():
                if isinstance(val, list) and key.endswith("_array") or key == "energy_vect":
                    data[key] = np.array(val)

            self.results = data
            messagebox.showinfo("Loaded", f"✅ Results loaded from:\n{file_path}, loading the related config file...")
            self.start_simulation()
            self.image_folder_entry.insert(0, "dud")
            self.show_results_page()
            self.results_frame.destroy()
            self.load_config_from_json()
            self.start_btn.pack_forget()
            self.load_btn.pack_forget()
            self.show_results_page()
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to load results:\n{e}")

    def load_result_tab(self, category):
        # Clear current tabs
        for tab_id in self.result_display.tabs():
            self.result_display.forget(tab_id)

        # Create a new frame for the selected category
        frame = tk.Frame(self.result_display, bg="white")
        self.result_display.add(frame, text=category)

        # Call corresponding display method
        method_name = f"display_{category.lower().replace(' ', '_')}"
        if hasattr(self, method_name):
            getattr(self, method_name)(frame)  # Pass the frame as parent
        else:
            label = tk.Label(frame, text=f"[{category}] content function not implemented.",
                             font=("Helvetica", 12), bg="white", fg="gray")
            label.pack(pady=20)

    def display_system_overview(self, parent):
        self.add_download_button(parent)

        # Use scrollable canvas
        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load from confirmed parameters if available
        data = self.parameters if self.updated_params else self.parameters

        # Display each category
        for category, param_dict in data.items():
            category_label = tk.Label(scrollable_frame, text=category, font=("Helvetica", 14, "bold"), bg="white")
            category_label.pack(anchor="w", pady=(10, 0), padx=10)

            for name, value in param_dict.items():
                row = tk.Frame(scrollable_frame, bg="white")
                row.pack(fill="x", padx=20, pady=2)

                param_label = tk.Label(row, text=name + ":", font=("Helvetica", 11), bg="white", width=35, anchor="w")
                param_label.pack(side="left")

                value_label = tk.Label(row, text=str(value), font=("Helvetica", 11, "bold"), bg="white", fg="darkblue")
                value_label.pack(side="left")

    def display_drone_path(self, parent):
        # Extract user coordinates for plotting
        x_coords_users = self.results["users_positions"][0]
        y_coords_users = self.results["users_positions"][1]
        # Extract UAV trajectory for plotting
        x_coords_circle = self.results["drone_positions"][0]
        y_coords_circle = self.results["drone_positions"][1]

        uav_initial_position_x = self.results["initial_uav_position"][0]
        uav_initial_position_y = self.results["initial_uav_position"][1]

        self.add_download_button(parent)
        # Create a matplotlib figure (instead of plt.figure)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        # Plot all users
        ax.scatter(x_coords_users, y_coords_users, color='blue', marker='o', label='Users')
        # Plot UAV trajectory
        ax.plot(x_coords_circle, y_coords_circle, 'r:', label='UAV Trajectory')
        # Plot initial UAV position
        ax.plot(uav_initial_position_x, uav_initial_position_y, 'ro', markersize=10, label='UAV Position')

        ax.set_title('User Positions and UAV Trajectory')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_xlim(0, self.results["area_size"])
        ax.set_ylim(0, self.results["area_size"])
        ax.grid(True)
        ax.legend()
        self.active_figure = fig

        # Embed the figure in the GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="none", expand=False)

    def display_snr_values(self, parent):
        snr_values = self.results.get("snr_values", [])
        if not snr_values:
            tk.Label(parent, text="No SNR values available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return

        threshold = self.threshold_snr
        below_threshold = sum(1 for value in snr_values if value < threshold)
        unusable_channel = (below_threshold / len(snr_values)) * 100

        # Message label
        msg = f"{unusable_channel:.3f}% of the time poor channel conditions prevented transmission"
        msg_label = tk.Label(parent, text=msg, font=("Helvetica", 12), bg="white", fg="darkred", wraplength=600,
                             justify="left")
        msg_label.pack(pady=(10, 5))

        # Fixed-size frame to hold the plot
        plot_frame = tk.Frame(parent, width=700, height=400, bg="white")
        plot_frame.pack(pady=10)
        plot_frame.pack_propagate(False)

        self.add_download_button(parent)

        # Create the histogram
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        ax.hist(snr_values, bins=30, color='g', edgecolor='black', alpha=0.7)
        ax.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold} dB)')
        ax.set_title("Histogram of SNR Values")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Number of Occurrences")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        self.active_figure = fig

        # Embed the figure
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def display_pie_charts(self, parent):
        # Extract required data from self.results
        offloaded_tasks = self.results.get("offloaded_tasks", [])
        local_tasks = self.results.get("local_tasks", [])
        completed_remote = self.results.get("completed_remote", [])
        dropped_remote = self.results.get("dropped_remote", [])
        completed_local = self.results.get("completed_local", [])
        dropped_local = self.results.get("dropped_local", [])

        if not (offloaded_tasks and local_tasks):
            tk.Label(parent, text="No task data available for pie charts.", font=("Helvetica", 12), bg="white").pack(
                pady=20)
            return

        # Aggregate totals
        total_offloaded = sum(offloaded_tasks)
        total_local = sum(local_tasks)

        completed_offload_total = sum(completed_remote)
        dropped_offload_total = sum(dropped_remote)

        completed_local_total = sum(completed_local)
        dropped_local_total = sum(dropped_local)

        # Custom label formatter (value + percentage)
        def make_autopct(values):
            def autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f'{val} ({pct:.1f}%)'

            return autopct

        self.add_download_button(parent)

        # Create the plots
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=100)

        # 1. Task Distribution
        axes[0].pie(
            [total_offloaded, total_local],
            labels=['Offloaded', 'Local'],
            autopct=make_autopct([total_offloaded, total_local]),
            colors=['#4A90E2', '#F5A623'],
            startangle=120,
            textprops={'fontsize': 12}
        )
        axes[0].set_title('Tasks Distribution', fontsize=14)

        # 2. Offloaded Tasks Outcome
        axes[1].pie(
            [completed_offload_total, dropped_offload_total],
            labels=['Completed', 'Dropped'],
            autopct=make_autopct([completed_offload_total, dropped_offload_total]),
            colors=['#2ECC71', '#E74C3C'],
            startangle=90,
            textprops={'fontsize': 12}
        )
        axes[1].set_title('Offloaded Tasks', fontsize=14)

        # 3. Local Tasks Outcome
        axes[2].pie(
            [completed_local_total, dropped_local_total],
            labels=['Completed', 'Dropped'],
            autopct=make_autopct([completed_local_total, dropped_local_total]),
            colors=['#2ECC71', '#E74C3C'],
            startangle=90,
            textprops={'fontsize': 12}
        )
        axes[2].set_title('Local Tasks', fontsize=14)

        fig.tight_layout()

        self.active_figure = fig

        # Embed figure into Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_latency_distribution(self, parent):
        # Get latency data
        latency_local_values = np.array(self.results.get("latency_local_values", []))
        latency_remote_values = np.array(self.results.get("latency_remote_values", []))

        if latency_local_values.size == 0 and latency_remote_values.size == 0:
            tk.Label(parent, text="No latency data available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return

        # Helper to format stats
        def get_latency_stats(name, data):
            return (
                f"{name} Latency Statistics:\n"
                f"Min: {data.min():.4f} s\n"
                f"Max: {data.max():.4f} s\n"
                f"Average: {data.mean():.4f} s\n"
                f"25th Percentile: {np.percentile(data, 25):.4f} s\n"
                f"Median: {np.percentile(data, 50):.4f} s\n"
                f"75th Percentile: {np.percentile(data, 75):.4f} s"
            )

        # === Side-by-side stats ===
        stats_frame = tk.Frame(parent, bg="white")
        stats_frame.pack(pady=(10, 5), padx=10, anchor="n")

        if latency_local_values.size:
            local_stats = get_latency_stats("Local", latency_local_values)
            tk.Label(stats_frame, text=local_stats, font=("Helvetica", 10), bg="white", justify="left", anchor="nw") \
                .grid(row=0, column=0, sticky="nw", padx=(0, 30))

        if latency_remote_values.size:
            remote_stats = get_latency_stats("Remote", latency_remote_values)
            tk.Label(stats_frame, text=remote_stats, font=("Helvetica", 10), bg="white", justify="left", anchor="nw") \
                .grid(row=0, column=1, sticky="nw")
        self.add_download_button(parent)
        # === Plot the distributions ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=100)

        if latency_local_values.size:
            sns.histplot(latency_local_values, kde=True, bins=100, ax=axes[0], color='orange')
            axes[0].set_title('Latency Distribution for Local Tasks')
            axes[0].set_xlabel('Latency (s)')
            axes[0].set_ylabel('Density')

        if latency_remote_values.size:
            sns.histplot(latency_remote_values, kde=True, bins=100, ax=axes[1], color='blue')
            axes[1].set_title('Latency Distribution for Offloaded Tasks')
            axes[1].set_xlabel('Latency (s)')

        fig.tight_layout()
        self.active_figure = fig

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_bandwidth_utilization(self, parent):
        bandwidth_utilization = np.array(self.results.get("bandwidth_utilization", []))

        if bandwidth_utilization.size == 0:
            tk.Label(parent, text="No bandwidth utilization data available.", font=("Helvetica", 12), bg="white").pack(
                pady=20)
            return

        # Prepare data
        xtime = np.arange(len(bandwidth_utilization))*self.results["time_step"]
        bandwidth_percent = bandwidth_utilization * 100

        self.add_download_button(parent)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

        # Step line
        ax.step(xtime, bandwidth_percent, where='post', color='royalblue', linewidth=2, label='Utilization')

        # Fill area under the step curve
        ax.fill_between(xtime, bandwidth_percent, step='post', alpha=0.3, color='royalblue')

        # Highlight zones
        ax.axhspan(0, 30, color='red', alpha=0.1, label='Low Utilization')
        ax.axhspan(90, 100, color='green', alpha=0.1, label='High Utilization')

        # Labels and styling
        ax.set_title("Bandwidth Utilization Percentage Over Time", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Bandwidth Utilization (%)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10, loc='upper right')
        fig.tight_layout()
        self.active_figure = fig

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_battery_usage(self, parent):
        avg_battery_usage = np.array(self.results.get("avg_battery_usage", []))

        if avg_battery_usage.size == 0:
            tk.Label(parent, text="No battery usage data available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return
        xtime = np.arange(avg_battery_usage.size)*self.results["time_step"]
        self.add_download_button(parent)
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        ax.plot(xtime, avg_battery_usage, color='darkorange', linewidth=2)

        ax.set_title("Average Battery Usage per User", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Energy Used (J)", fontsize=12)

        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        self.active_figure = fig

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_energy_per_task(self, parent):
        uav_energy_array = np.array(self.results.get("uav_computational_energy", []))
        user_energy_array = np.array(self.results.get("user_energy_array", []))
        completed_remote = self.results.get("completed_remote", [])
        completed_local = self.results.get("completed_local", [])

        if uav_energy_array.size == 0 or user_energy_array.size == 0:
            tk.Label(parent, text="Missing energy data for UAV or users.", font=("Helvetica", 12), bg="white").pack(
                pady=20)
            return

        if sum(completed_remote) == 0 or sum(completed_local) == 0:
            tk.Label(parent, text="No completed tasks found for UAV or users.", font=("Helvetica", 12),
                     bg="white").pack(pady=20)
            return

        # Compute averages
        avg_uav_energy = np.sum(uav_energy_array) / sum(completed_remote)
        avg_user_energy = np.sum(user_energy_array) / sum(completed_local)

        # Use DataFrame and hue
        df = pd.DataFrame({
            "Source": ["UAV", "Users"],
            "Energy": [avg_uav_energy, avg_user_energy],
            "Color": ["UAV", "Users"]  # use same values for hue
        })
        self.add_download_button(parent)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        sns.barplot(data=df, x="Source", y="Energy", hue="Color", palette={"UAV": "blue", "Users": "orange"},
                    legend=False, ax=ax)

        ax.set_ylabel('Avg Energy per Task (J/task)', fontsize=12)
        ax.set_title('Average Energy per Task: UAV vs Users', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()
        self.active_figure = fig

        # Embed in GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_jain_index_on_energy(self, parent):
        energy_saved_all = self.results.get("energy_saved_all", [])
        generated_tasks = self.results.get("generated_tasks", [])

        if not energy_saved_all or not generated_tasks:
            tk.Label(parent, text="Missing data for energy saved or task counts.", font=("Helvetica", 12),
                     bg="white").pack(pady=20)
            return

        # Step 1: Flatten if needed
        raw_energy_saved = [
            sum(es) if isinstance(es, list) else es
            for es in energy_saved_all
        ]

        # Step 2: Normalize by generated tasks
        normalized_energy_saved = [
            es / gt if gt > 0 else 0
            for es, gt in zip(raw_energy_saved, generated_tasks)
        ]

        # Step 3: Jain Index (raw)
        numerator_raw = sum(raw_energy_saved) ** 2
        denominator_raw = len(raw_energy_saved) * sum(x ** 2 for x in raw_energy_saved)
        jain_index_raw = numerator_raw / denominator_raw if denominator_raw != 0 else 0

        # Step 4: Jain Index (normalized)
        numerator_norm = sum(normalized_energy_saved) ** 2
        denominator_norm = len(normalized_energy_saved) * sum(x ** 2 for x in normalized_energy_saved)
        jain_index_norm = numerator_norm / denominator_norm if denominator_norm != 0 else 0

        self.add_download_button(parent)
        # Step 5: Plot both
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

        axs[0].bar(range(len(raw_energy_saved)), raw_energy_saved, color='orange')
        axs[0].axhline(y=np.mean(raw_energy_saved), color='r', linestyle='--', linewidth=2, label='y = 5')
        axs[0].set_title(f'Total Energy Saved per User\nJain Index = {jain_index_raw:.4f}', fontsize=12)
        axs[0].set_xlabel('User ID')
        axs[0].set_ylabel('Joules')
        axs[0].grid(True)

        axs[1].bar(range(len(normalized_energy_saved)), normalized_energy_saved, color='skyblue')
        axs[1].set_title(f'Energy Saved per Task per User\nJain Index = {jain_index_norm:.4f}', fontsize=12)
        axs[1].set_xlabel('User ID')
        axs[1].set_ylabel('Joules / Task generated')
        axs[1].grid(True)

        fig.tight_layout()
        self.active_figure = fig

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_task_completion_per_user(self, parent):
        generated_tasks = self.results.get("generated_tasks", [])
        completed_local = self.results.get("completed_local", [])
        completed_remote = self.results.get("completed_remote", [])
        dropped_local = self.results.get("dropped_local", [])
        dropped_remote = self.results.get("dropped_remote", [])

        if not generated_tasks:
            tk.Label(parent, text="No task generation data available.", font=("Helvetica", 12), bg="white").pack(
                pady=20)
            return

        # Completed and dropped task totals
        completed_tasks = [l + r for l, r in zip(completed_local, completed_remote)]
        dropped_tasks = [l + r for l, r in zip(dropped_local, dropped_remote)]

        # Calculate percentages (safe division)
        percentage_completed = [
            (c / gg) * 100 if gg > 0 else 0
            for c, gg in zip(completed_tasks, generated_tasks)
        ]
        percentage_dropped = [
            (d / gg) * 100 if gg > 0 else 0
            for d, gg in zip(dropped_tasks, generated_tasks)
        ]

        x = np.arange(len(generated_tasks))
        bar_width = 0.4

        self.add_download_button(parent)
        # Create the plots
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

        # Plot 1: Number of generated tasks
        ax[0].bar(x, generated_tasks, color='blue')
        ax[0].set_xlabel('User ID')
        ax[0].set_ylabel('Number of Generated Tasks')
        ax[0].set_title('Number of Generated Tasks per User')
        ax[0].set_xticks(x)

        # Plot 2: Completion vs drop percentage
        ax[1].bar(x - bar_width / 2, percentage_completed, width=bar_width, color='green', label='Completed (%)')
        ax[1].bar(x + bar_width / 2, percentage_dropped, width=bar_width, color='red', label='Dropped (%)')
        ax[1].set_xlabel('User ID')
        ax[1].set_ylabel('Percentage of Generated Tasks (%)')
        ax[1].set_title('Task Completion and Drop Rate per User')
        ax[1].set_xticks(x)
        ax[1].legend()

        fig.tight_layout()
        self.active_figure = fig

        # Embed in GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_offloaded_task_completion(self, parent):
        generated_tasks = self.results.get("generated_tasks", [])
        offloaded_tasks = self.results.get("offloaded_tasks", [])
        completed_remote = self.results.get("completed_remote", [])
        dropped_remote = self.results.get("dropped_remote", [])

        if not generated_tasks or not offloaded_tasks:
            tk.Label(parent, text="No offloaded task data available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return

        # Plot 1: % Offloaded compared to generated
        percentage_offloaded = [
            (o / gg) * 100 if gg > 0 else 0
            for o, gg in zip(offloaded_tasks, generated_tasks)
        ]

        # Plot 2: % Completed and Dropped of Offloaded tasks
        percentage_completed_remote = [
            (c / o) * 100 if o > 0 else 0
            for c, o in zip(completed_remote, offloaded_tasks)
        ]
        percentage_dropped_remote = [
            (d / o) * 100 if o > 0 else 0
            for d, o in zip(dropped_remote, offloaded_tasks)
        ]

        x = np.arange(len(offloaded_tasks))
        bar_width = 0.4

        self.add_download_button(parent)
        # Create subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        # Subplot 1: % Offloaded vs Generated
        ax[0].bar(x, percentage_offloaded, color='blue')
        ax[0].set_xlabel('User ID')
        ax[0].set_ylabel('% of Tasks Offloaded')
        ax[0].set_xticks(x)
        ax[0].set_title('Percentage of Tasks Offloaded per User')

        # Subplot 2: Completion vs Drop of Offloaded Tasks
        ax[1].bar(x - bar_width / 2, percentage_completed_remote, width=bar_width, color='green', label='Completed (%)')
        ax[1].bar(x + bar_width / 2, percentage_dropped_remote, width=bar_width, color='red', label='Dropped (%)')
        ax[1].set_xlabel('User ID')
        ax[1].set_ylabel('% of Offloaded Tasks')
        ax[1].set_xticks(x)
        ax[1].set_title('Completion & Drop Rate of Offloaded Tasks')
        ax[1].legend()

        fig.tight_layout()
        self.active_figure = fig

        # Embed in GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_local_task_completion(self, parent):
        generated_tasks = self.results.get("generated_tasks", [])
        local_tasks = self.results.get("local_tasks", [])
        completed_local = self.results.get("completed_local", [])
        dropped_local = self.results.get("dropped_local", [])

        if not generated_tasks or not local_tasks:
            tk.Label(parent, text="No local task data available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return

        # Plot 1: % Locally executed compared to generated
        percentage_local = [
            (local / generated * 100) if generated > 0 else 0
            for local, generated in zip(local_tasks, generated_tasks)
        ]

        # Plot 2: % Completed and Dropped of Local tasks
        percentage_completed_local = [
            (completed / local * 100) if local > 0 else 0
            for completed, local in zip(completed_local, local_tasks)
        ]
        percentage_dropped_local = [
            (dropped / local * 100) if local > 0 else 0
            for dropped, local in zip(dropped_local, local_tasks)
        ]

        x = np.arange(len(local_tasks))
        bar_width = 0.4

        self.add_download_button(parent)
        # Create the plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        # Subplot 1: % Local execution vs Generated
        ax[0].bar(x, percentage_local, color='orange')
        ax[0].set_xlabel('User ID')
        ax[0].set_ylabel('% of Tasks Executed Locally')
        ax[0].set_xticks(x)
        ax[0].set_title('Percentage of Tasks Executed Locally per User')

        # Subplot 2: Completion vs Drop of Local Tasks
        ax[1].bar(x - bar_width / 2, percentage_completed_local, width=bar_width, color='green', label='Completed (%)')
        ax[1].bar(x + bar_width / 2, percentage_dropped_local, width=bar_width, color='red', label='Dropped (%)')
        ax[1].set_xlabel('User ID')
        ax[1].set_ylabel('% of Local Tasks')
        ax[1].set_xticks(x)
        ax[1].set_title('Completion & Drop Rate of Local Tasks')
        ax[1].legend()

        fig.tight_layout()
        self.active_figure = fig

        # Embed into GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_offloading_fairness(self, parent):
        fairness = self.results.get("fairness", [])

        if not fairness:
            tk.Label(parent, text="No fairness data available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return

        x = np.arange(len(fairness))  # User IDs

        self.add_download_button(parent)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.bar(x, fairness, color='dodgerblue')

        ax.set_xlabel('User ID')
        ax.set_ylabel('Fairness Ratio\n(Completed Offloaded Tasks / Offloaded Tasks)')
        ax.set_title('Fairness of Task Offloading')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_ylim(0, 1.05)  # Fairness is between 0 and 1
        ax.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        self.active_figure = fig

        # Embed in GUI
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def display_uav_available_energy(self, parent):
        energy_vect = self.results.get("energy_vect", [])

        if energy_vect is None or len(energy_vect) == 0:
            tk.Label(parent, text="No UAV energy data available.", font=("Helvetica", 12), bg="white").pack(pady=20)
            return

        self.add_download_button(parent)

        # Safe normalization
        max_energy = max(energy_vect)
        if max_energy == 0:
            normalized_energy = energy_vect  # or [0]*len(energy_vect)
        else:
            normalized_energy = 100 * np.array(energy_vect) / max_energy

        xtime = np.arange(len(normalized_energy))*self.results["time_step"]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(xtime, normalized_energy, marker='o', linestyle='-', color='blue')

        ax.set_title("UAV Available Battery (%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Remaining Energy (%)")
        ax.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        self.active_figure = fig

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def add_download_button(self, parent):
        download_btn = tk.Button(
            parent,
            text="Download Plot",
            font=("Helvetica", 10),
            command=self.download_current_plot
        )
        download_btn.pack(anchor="ne", padx=10, pady=(20, 0))

    def download_current_plot(self):
        if self.active_figure is None:
            messagebox.showwarning("No Plot", "There is no plot to download.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png")])
        if file_path:
            try:
                self.active_figure.savefig(file_path, dpi=300)
                messagebox.showinfo("Saved", f"✅ Plot saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"❌ Failed to save plot:\n{e}")

    def reset_to_start(self):
        # Ask for confirmation (optional)
        if not messagebox.askyesno("Reset Simulation", "Are you sure you want to reset and return to the start?"):
            return

        # Destroy the results frame
        if hasattr(self, 'results_frame') and self.results_frame:
            self.results_frame.destroy()

        # Reset any active figure
        self.active_figure = None

        # Clear result data
        self.results = {}

        # Hide notebook and remove all its tabs
        if hasattr(self, 'notebook'):
            self.notebook.pack_forget()
            for tab_id in self.notebook.tabs():
                self.notebook.forget(tab_id)

        # Hide button frame (save/load/confirm/run)
        if hasattr(self, 'button_frame'):
            self.button_frame.pack_forget()

        # Show initial start/load buttons
        self.start_btn.pack(pady=(10, 5))
        self.load_btn.pack(pady=(5, 30))


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()
