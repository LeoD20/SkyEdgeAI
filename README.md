# SkyEdgeAI: UAV-Assisted Edge Computing Simulator

## 📡 Overview

This project simulates a UAV-assisted wireless edge computing system where a drone equipped with computing capabilities serves as a mobile edge server. Users generate computational tasks (based on object detection using YOLOv3) and dynamically decide whether to offload them to the UAV or process them locally. The goal is to evaluate system performance across metrics like latency, energy efficiency, fairness, and resource utilization under realistic wireless conditions.

The simulator includes a full GUI for running, configuring, and visualizing simulations — or alternatively, users can work with the provided Jupyter notebook for more direct control.

## 🎯 Project Goal

To evaluate the performance of UAV-assisted edge computing through simulation, considering:

- **Latency**: End to end processing time locally vs. at the edge.
- **Energy Efficiency**: User vs. UAV consumption.
- **Offloading & Resource Utilization**: Task completion and bandwidth use.
- **Fairness**: Equitable access to edge computing resources.

## 🗂️ File Descriptions

| File / Folder | Description |
|---------------|-------------|
| `EAId_GUI.py` | Main graphical interface for configuring and running simulations, visualizing performance, and managing results. |
| `simulation.ipynb` | Notebook-based interface to run the same simulation logic with plots, allowing deeper control and interactivity without the GUI. |
| `channel_model.py` | Implements wireless channel modeling (Path Loss, Rician fading, Doppler, shadowing, SNR, spectral efficiency). |
| `environment.py` | Defines the UAV and the user class with all their features. It also models the UAV mobility. |
| `task_managment.py` | Manages task creation, offloading decisions, queueing, processing, and energy-latency trade-offs. |
| `config_parameters.json` | Example simulation configuration containing all adjustable parameters from the GUI. |
| `inference_results.json` | Precomputed YOLOv3 inference statistics for faster simulation setup. |
| `simulation_results.json` | Output of a simulation run — includes data such as energy consumption, latency, SNR, fairness, task completion, etc. |
| `requirements.txt` | Python dependencies needed to run the project. |
| `yolo_images/` | Sample image set used during inference simulations. |

## 📦 Installation

Make sure Python 3.8+ is installed. Then install or required dependencies to run the code of the repository:

```bash
git clone https://github.com/AlvaroSolana/SkyEdgeAI.git
cd SkyEdgeAI
pip install -r requirements.txt
```


## 🚀 Usage

### 🖥️ GUI Mode (Recommended)

```bash
python EAId_GUI.py
```

- Set parameters for UAV, users, task generation, and channel conditions.
- Run new simulations or load previous results.
- Visualize key performance indicators like energy usage, latency, SNR, fairness, etc.
- Export configuration or results to JSON.

### 📓 Notebook Mode

Open `simulation.ipynb` in Jupyter:

```bash
jupyter lab
```

This mode runs the same simulation flow but through code cells, allowing greater experimentation and visibility. Ideal for debugging or exploring new setups quickly.


## 📄 License

This project is open-source under the MIT License. Feel free to use, modify, and share.

## 📬 Contact

For questions or contributions, contact the repository owner.
