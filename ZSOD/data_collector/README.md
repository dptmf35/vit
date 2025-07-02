# ZSOD: Zero-Shot Object Detection Toolkit for ROS2

![Project Banner](https://user-images.githubusercontent.com/12345/67890.png) <!-- Placeholder for a cool banner -->

**ZSOD** is a comprehensive, ROS2-based toolkit designed for the rapid collection, annotation, and review of high-quality datasets for Zero-Shot Object Detection (ZSOD). It leverages the power of text and visual prompting with state-of-the-art YOLOE models to enable the creation of custom datasets with minimal effort.

This toolkit is ideal for robotics applications where new objects need to be recognized on-the-fly without extensive retraining.

---

## 核心功能 (Core Features)

-   **📝 Text-Prompt-Based Data Collection**: Automatically collect and label image data by providing a list of target object classes as text prompts.
-   **🎨 Interactive Visual Prompting**: "Show" the model what to detect by drawing bounding boxes on a live video stream. The system then tracks and detects these objects in subsequent frames.
-   **🚀 Dynamic Mode Switching**: Seamlessly switch between a `TEST` mode (for detection and visualization only) and a `COLLECTION` mode (for saving data) via keyboard commands or a ROS2 service call.
-   **🤖 Automated YOLO Formatting**: All collected data is automatically saved in the standard YOLO format (`.txt` labels, `dataset.yaml` config), ready for immediate training with the Ultralytics framework.
-   **🔍 Dataset Review & Annotation Tools**: Includes scripts to review collected images and manually edit or refine annotations, ensuring dataset quality.
-   **⚙️ Highly Configurable**: Easily adjust parameters like confidence thresholds, IoU, collection frequency, and ROS topics via command-line arguments.

---

## 🏛️ Project Architecture

The ZSOD toolkit operates as a series of interconnected ROS2 nodes that process image data and produce a structured dataset.

```
+--------------------------------+
|      ROS2 Environment          |
|                                |
|  [ROS2 Image Topic]            |  <-- e.g., /stereo_image_color, /camera/image_raw
|  (e.g., RealSense Camera)      |
+-----------------|--------------+
                  |
                  v
+--------------------------------+
|         ZSOD Toolkit           |
|                                |
|  +--------------------------+  |
|  |  Data Collector          |  |
|  | (Text/Visual Prompts)    |  |
|  +--------------------------+  |
|  +--------------------------+  |
|  |  Dataset Reviewer        |  |
|  +--------------------------+  |
|  +--------------------------+  |
|  | Annotation Editor        |  |
|  +--------------------------+  |
|                                |
+-----------------|--------------+
                  |
                  v
+--------------------------------+
|      Generated YOLO Dataset    |
|                                |
|  - /images/*.jpg               |
|  - /labels/*.txt               |
|  - /visualizations/*.jpg       |
|  - dataset.yaml                |
+--------------------------------+
```

---

## 🧩 Project Components

The `data_collector` directory contains the core logic for the toolkit.

| File/Script                        | Description                                                                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`yolo_dataset_collector.py`**    | **Core Engine**: A ROS2 node that subscribes to an image topic, runs the YOLOE model with text prompts, and saves valid detections as a YOLO dataset. |
| **`run_dataset_collector.py`**     | **Configurable Launcher**: A user-friendly script with command-line arguments to configure and launch the `yolo_dataset_collector`.                |
| **`visual_prompt_detector.py`**    | **Interactive Detector**: A ROS2 node that allows the user to draw bounding boxes on the first frame to create visual prompts for the YOLOE model. |
| **`run_visual_prompt.py`**         | **Launcher** for the interactive visual prompt detector.                                                                                         |
| **`dataset_reviewer.py`**          | **Review Tool**: A utility to visually inspect the collected images and their corresponding annotations.                                           |
| **`interactive_annotation_editor.py`** | **Annotation Editor**: An advanced tool to manually draw, delete, or modify bounding box annotations on the collected images.                  |
| **`toggle_mode.py`**               | **Utility Script**: A simple client to switch the collector's mode using the ROS2 service.                                                        |
| **`*.pt` files**                   | **Model Weights**: Pre-trained model files (e.g., `yoloe-11m-seg.pt`) used by the detectors.                                                       |
| **`*.sh` files**                   | **Shell Scripts**: Convenience scripts for launching the various components with predefined settings.                                              |

---

## 🛠️ Technology Stack & Libraries

-   **Programming Language**: Python 3
-   **Frameworks**: ROS2 (Robot Operating System 2), PyTorch
-   **Core Libraries**:
    -   `ultralytics`: For the YOLOE models and training infrastructure.
    -   `opencv-python`: For all image processing, drawing, and visualization tasks.
    -   `numpy`: For numerical operations and array manipulation.
    -   `rclpy`: The Python client library for ROS2.
    -   `cv_bridge`: To convert between ROS Image messages and OpenCV images.

---

## 🚀 Getting Started

### Prerequisites

1.  **ROS2 Installation**: A working installation of ROS2 (e.g., Humble, Iron).
2.  **Python Environment**: Python 3.8+ with `pip`.
3.  **Camera Driver**: A ROS2 compatible camera driver publishing `sensor_msgs/msg/Image` topics.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ZSOD
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or install manually:
    # pip install ultralytics opencv-python numpy
    ```

3.  **Source your ROS2 environment:**
    ```bash
    source /opt/ros/humble/setup.bash
    ```

---

## Workflow & Usage

### 1. Text-Prompt-Based Data Collection

This is the primary method for collecting a large dataset based on a predefined list of classes.

1.  **Launch your ROS2 camera node.** Make sure the image topic is active.
2.  **Run the collector.** Use the launcher script for easy configuration.

    ```bash
    # Basic execution with default settings
    python3 data_collector/run_dataset_collector.py

    # Advanced execution with custom parameters
    python3 data_collector/run_dataset_collector.py \
        --image_topic /camera/color/image_raw \
        --conf_threshold 0.7 \
        --collection_interval 1.5 \
        --dataset_path ~/my_robot_dataset
    ```

3.  The collector will now save images and labels to the specified dataset path whenever it detects the target objects.

### 2. Interactive Visual-Prompt-Based Detection

Use this mode to detect objects by showing them to the model.

1.  **Launch your ROS2 camera node.**
2.  **Run the visual prompt script:**
    ```bash
    python3 data_collector/run_visual_prompt.py
    ```
3.  **Follow the on-screen instructions:**
    -   A window will appear showing the live camera feed.
    -   Draw a bounding box around an object of interest.
    -   Select the corresponding class from the list in the terminal.
    -   Repeat for all objects you want to prompt.
    -   Press `c` to confirm the prompts and start live detection.
    -   Press `r` to reset and `q` to quit.

### 3. Dataset Review and Editing

After collecting data, it's crucial to review it for quality.

```bash
# Launch the reviewer to inspect the dataset
python3 data_collector/run_dataset_reviewer.py --dataset_path ~/my_robot_dataset

# Launch the editor to fix annotations
python3 data_collector/interactive_annotation_editor.py --dataset_path ~/my_robot_dataset
```

---

## ⚙️ Configuration Parameters

The `run_dataset_collector.py` script accepts the following arguments:

| Parameter             | Default Value         | Description                                                              |
| --------------------- | --------------------- | ------------------------------------------------------------------------ |
| `--conf_threshold`    | `0.6`                 | Confidence threshold for an object detection to be considered valid.       |
| `--iou_threshold`     | `0.4`                 | Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS). |
| `--collection_interval` | `2.0`                 | Minimum time in seconds between saving two consecutive images.           |
| `--min_detections`    | `1`                   | The minimum number of objects that must be detected to save the sample.    |
| `--max_detections`    | `50`                  | The maximum number of detections to save per image.                      |
| `--dataset_path`      | `~/yolo_dataset`      | The root directory where the collected dataset will be saved.            |
| `--image_topic`       | `/stereo_image_color` | The ROS2 topic to subscribe to for input images.                         |
| `--model_path`        | `yoloe-11m-seg.pt`    | Path to the YOLOE model weights file.                                    |
| `--test_mode`         | `False`               | If enabled, runs detection and visualization without saving any data.    |

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

-   The [Ultralytics](https://ultralytics.com/) team for their amazing YOLO models and framework.
-   The [ROS2](https://ros.org/) community for building the future of robotics.
