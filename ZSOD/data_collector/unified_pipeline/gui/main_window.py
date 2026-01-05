#!/usr/bin/env python3

import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
import threading
from typing import List

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from config.pipeline_config import (
    ConfigManager, PipelineConfig, CollectionConfig,
    TrainingConfig, EvaluationConfig, DeploymentConfig
)
from modules.data_collection import DataCollectionModule
from modules.annotation_review import AnnotationReviewModule
from modules.training import TrainingModule
from modules.evaluation import EvaluationModule
from modules.deployment import DeploymentModule


class UnifiedPipelineGUI:
    """Main GUI for the unified ML pipeline"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unified ML Pipeline - Data Collection to Deployment")
        self.root.geometry("1400x900")

        # Configuration
        self.config_manager = ConfigManager()
        self.config = None

        # Modules
        self.collection_module = None
        self.annotation_module = None
        self.training_module = None
        self.evaluation_module = None
        self.deployment_module = None

        # Setup GUI
        self.setup_gui()

        # Load or create default config
        self.load_config()

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_config_tab()
        self.create_collection_tab()
        self.create_annotation_tab()
        self.create_training_tab()
        self.create_evaluation_tab()
        self.create_deployment_tab()

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_config_tab(self):
        """Create configuration tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⚙️ Configuration")

        # Main frame with scrollbar
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Target Classes
        classes_frame = ttk.LabelFrame(scrollable_frame, text="Target Classes", padding=10)
        classes_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(classes_frame, text="Enter classes (comma-separated):").pack(anchor=tk.W)
        self.classes_text = scrolledtext.ScrolledText(classes_frame, height=5, width=80)
        self.classes_text.pack(fill=tk.X, pady=5)

        # Collection Config
        collection_frame = ttk.LabelFrame(scrollable_frame, text="Collection Settings", padding=10)
        collection_frame.pack(fill=tk.X, padx=10, pady=10)

        self.conf_threshold_var = tk.DoubleVar(value=0.6)
        self.collection_interval_var = tk.DoubleVar(value=2.0)
        self.dataset_path_var = tk.StringVar(value="~/yolo_dataset")
        self.use_yolo11_var = tk.BooleanVar(value=False)

        ttk.Label(collection_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Scale(collection_frame, from_=0.1, to=1.0, variable=self.conf_threshold_var, orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(collection_frame, textvariable=self.conf_threshold_var).grid(row=0, column=2, padx=10)

        ttk.Label(collection_frame, text="Collection Interval (seconds):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Scale(collection_frame, from_=0.5, to=10.0, variable=self.collection_interval_var, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(collection_frame, textvariable=self.collection_interval_var).grid(row=1, column=2, padx=10)

        ttk.Label(collection_frame, text="Dataset Path:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(collection_frame, textvariable=self.dataset_path_var, width=50).grid(row=2, column=1, columnspan=2, sticky=tk.W)

        ttk.Checkbutton(collection_frame, text="Use YOLO11 (trained model)", variable=self.use_yolo11_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Training Config
        training_frame = ttk.LabelFrame(scrollable_frame, text="Training Settings", padding=10)
        training_frame.pack(fill=tk.X, padx=10, pady=10)

        self.model_size_var = tk.StringVar(value="yolo11s.pt")
        self.epochs_var = tk.IntVar(value=100)

        ttk.Label(training_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(training_frame, textvariable=self.model_size_var,
                     values=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
                     state="readonly", width=20).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(training_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(training_frame, from_=10, to=500, textvariable=self.epochs_var, width=20).grid(row=1, column=1, sticky=tk.W)

        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="💾 Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔧 Reset to Default", command=self.reset_config).pack(side=tk.LEFT, padx=5)

    def create_collection_tab(self):
        """Create data collection tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📸 Data Collection")

        # Control frame
        control_frame = ttk.LabelFrame(tab, text="Collection Control", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.collection_status_label = ttk.Label(control_frame, text="Status: Idle", font=("Arial", 12, "bold"))
        self.collection_status_label.pack(anchor=tk.W, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="▶️ Start Collection", command=self.start_collection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔍 Test Mode", command=self.start_test_collection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⏹️ Stop Collection", command=self.stop_collection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 Dataset Info", command=self.show_dataset_info).pack(side=tk.LEFT, padx=5)

        # Statistics frame
        stats_frame = ttk.LabelFrame(tab, text="Collection Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        self.collection_stats_label = ttk.Label(stats_frame, text="No data collected yet", font=("Arial", 10))
        self.collection_stats_label.pack(anchor=tk.W)

        # Log frame
        log_frame = ttk.LabelFrame(tab, text="Collection Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.collection_log = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.collection_log.pack(fill=tk.BOTH, expand=True)

    def create_annotation_tab(self):
        """Create annotation review tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="✏️ Annotation Review")

        # Control frame
        control_frame = ttk.LabelFrame(tab, text="Review Control", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="🔍 Launch Reviewer", command=self.launch_reviewer).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 Dataset Statistics", command=self.show_annotation_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="✅ Validate Labels", command=self.validate_labels).pack(side=tk.LEFT, padx=5)

        # Statistics frame
        stats_frame = ttk.LabelFrame(tab, text="Dataset Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.annotation_stats_text = scrolledtext.ScrolledText(stats_frame, height=20, state=tk.DISABLED)
        self.annotation_stats_text.pack(fill=tk.BOTH, expand=True)

    def create_training_tab(self):
        """Create training tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎓 Training")

        # Control frame
        control_frame = ttk.LabelFrame(tab, text="Training Control", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.training_status_label = ttk.Label(control_frame, text="Status: Not trained", font=("Arial", 12, "bold"))
        self.training_status_label.pack(anchor=tk.W, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="▶️ Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 Dataset Analysis", command=self.analyze_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📈 Training Results", command=self.show_training_results).pack(side=tk.LEFT, padx=5)

        # Progress frame
        progress_frame = ttk.LabelFrame(tab, text="Training Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)

        self.training_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.training_progress.pack(fill=tk.X, pady=5)

        # Log frame
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.training_log = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.training_log.pack(fill=tk.BOTH, expand=True)

    def create_evaluation_tab(self):
        """Create evaluation tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Evaluation")

        # Control frame
        control_frame = ttk.LabelFrame(tab, text="Evaluation Control", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="📊 Evaluate on Dataset", command=self.evaluate_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🖼️ Test on Image", command=self.test_on_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📁 Test on Directory", command=self.test_on_directory).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(tab, text="Evaluation Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.evaluation_results_text = scrolledtext.ScrolledText(results_frame, height=20, state=tk.DISABLED)
        self.evaluation_results_text.pack(fill=tk.BOTH, expand=True)

    def create_deployment_tab(self):
        """Create deployment tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🚀 Deployment")

        # Control frame
        control_frame = ttk.LabelFrame(tab, text="Deployment Control", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.deployment_status_label = ttk.Label(control_frame, text="Status: Not deployed", font=("Arial", 12, "bold"))
        self.deployment_status_label.pack(anchor=tk.W, pady=5)

        # Configuration
        config_frame = ttk.Frame(control_frame)
        config_frame.pack(fill=tk.X, pady=10)

        ttk.Label(config_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.deploy_model_path_var = tk.StringVar(value="train_model/training_output/train/weights/best.pt")
        ttk.Entry(config_frame, textvariable=self.deploy_model_path_var, width=50).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(config_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5)

        ttk.Label(config_frame, text="Camera Topic:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.camera_topic_var = tk.StringVar(value="/stereo_image_color")
        ttk.Entry(config_frame, textvariable=self.camera_topic_var, width=50).grid(row=1, column=1, sticky=tk.W)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="▶️ Start Deployment", command=self.start_deployment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⏹️ Stop Deployment", command=self.stop_deployment).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="ℹ️ Deployment Info", command=self.show_deployment_info).pack(side=tk.LEFT, padx=5)

        # Info frame
        info_frame = ttk.LabelFrame(tab, text="Deployment Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info_text = """
ROS2 Topics:
• Annotated Image: /custom_yolo/annotated_image
• Detections (JSON): /custom_yolo/detections
• Bounding Boxes: /custom_yolo/bounding_boxes

Usage:
1. Make sure the trained model exists
2. Verify camera topic is publishing
3. Click 'Start Deployment' to launch ROS2 node
4. View results on the annotated image topic
        """
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)

    # Configuration methods
    def save_config(self):
        """Save current configuration"""
        try:
            # Get classes
            classes_text = self.classes_text.get("1.0", tk.END).strip()
            classes = [c.strip() for c in classes_text.split(",") if c.strip()]

            if not classes:
                messagebox.showerror("Error", "Please enter at least one target class")
                return

            # Create config
            self.config = PipelineConfig(
                collection=CollectionConfig(
                    target_classes=classes,
                    conf_threshold=self.conf_threshold_var.get(),
                    collection_interval=self.collection_interval_var.get(),
                    dataset_path=self.dataset_path_var.get(),
                    use_yolo11=self.use_yolo11_var.get()
                ),
                training=TrainingConfig(
                    model_size=self.model_size_var.get(),
                    epochs=self.epochs_var.get()
                ),
                evaluation=EvaluationConfig(),
                deployment=DeploymentConfig(
                    model_path=self.deploy_model_path_var.get(),
                    camera_topic=self.camera_topic_var.get()
                )
            )

            self.config_manager.save_config(self.config)
            messagebox.showinfo("Success", "Configuration saved successfully!")
            self.update_status("Configuration saved")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        """Load configuration"""
        self.config = self.config_manager.load_config()

        if not self.config:
            # Create default
            default_classes = self.config_manager.get_default_classes()
            self.config = self.config_manager.create_default_config(default_classes)

        # Update UI
        self.classes_text.delete("1.0", tk.END)
        self.classes_text.insert("1.0", ", ".join(self.config.collection.target_classes))

        self.conf_threshold_var.set(self.config.collection.conf_threshold)
        self.collection_interval_var.set(self.config.collection.collection_interval)
        self.dataset_path_var.set(self.config.collection.dataset_path)
        self.use_yolo11_var.set(self.config.collection.use_yolo11)

        self.model_size_var.set(self.config.training.model_size)
        self.epochs_var.set(self.config.training.epochs)

        self.deploy_model_path_var.set(self.config.deployment.model_path)
        self.camera_topic_var.set(self.config.deployment.camera_topic)

        self.update_status("Configuration loaded")

    def reset_config(self):
        """Reset to default configuration"""
        if messagebox.askyesno("Confirm", "Reset to default configuration?"):
            default_classes = self.config_manager.get_default_classes()
            self.config = self.config_manager.create_default_config(default_classes)
            self.load_config()

    # Collection methods
    def start_collection(self):
        """Start data collection"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        self.collection_module = DataCollectionModule(
            self.config.collection,
            lambda msg, level: self.update_collection_log(msg, level)
        )

        if self.collection_module.start_collection(test_mode=False):
            self.collection_status_label.config(text="Status: Collecting...")
            self.update_status("Data collection started")

    def start_test_collection(self):
        """Start test mode collection"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        self.collection_module = DataCollectionModule(
            self.config.collection,
            lambda msg, level: self.update_collection_log(msg, level)
        )

        if self.collection_module.start_collection(test_mode=True):
            self.collection_status_label.config(text="Status: Testing...")
            self.update_status("Test mode started")

    def stop_collection(self):
        """Stop data collection"""
        if self.collection_module:
            self.collection_module.stop_collection()
            self.collection_status_label.config(text="Status: Stopped")
            self.update_status("Collection stopped")

            # Update statistics
            stats = self.collection_module.get_statistics()
            self.collection_stats_label.config(
                text=f"Total collected: {stats['total_collected']}"
            )

    def show_dataset_info(self):
        """Show dataset information"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        if not self.collection_module:
            self.collection_module = DataCollectionModule(self.config.collection, None)

        info = self.collection_module.get_dataset_info()

        info_text = f"""
Dataset Path: {info['path']}
Exists: {info['exists']}
Images: {info['images_count']}
Labels: {info['labels_count']}
        """

        messagebox.showinfo("Dataset Information", info_text)

    # Annotation methods
    def launch_reviewer(self):
        """Launch annotation reviewer"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        self.annotation_module = AnnotationReviewModule(
            self.config.collection.dataset_path,
            lambda msg, level: self.update_status(msg)
        )

        threading.Thread(target=self.annotation_module.launch_reviewer, daemon=True).start()

    def show_annotation_stats(self):
        """Show annotation statistics"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        if not self.annotation_module:
            self.annotation_module = AnnotationReviewModule(self.config.collection.dataset_path, None)

        stats = self.annotation_module.get_dataset_statistics()

        self.annotation_stats_text.config(state=tk.NORMAL)
        self.annotation_stats_text.delete("1.0", tk.END)

        text = f"""
Dataset Statistics:
-------------------
Total Images: {stats['total_images']}
Images with Labels: {stats['images_with_labels']}
Images without Labels: {stats['images_without_labels']}
Total Annotations: {stats['total_labels']}

Class Distribution:
-------------------
"""
        for class_name, count in sorted(stats['class_distribution'].items()):
            text += f"{class_name}: {count}\n"

        self.annotation_stats_text.insert("1.0", text)
        self.annotation_stats_text.config(state=tk.DISABLED)

    def validate_labels(self):
        """Validate label files"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        if not self.annotation_module:
            self.annotation_module = AnnotationReviewModule(self.config.collection.dataset_path, None)

        validation = self.annotation_module.validate_labels()

        result_text = f"""
Validation Results:
-------------------
Valid Labels: {validation['valid_labels']}
Invalid Labels: {validation['invalid_labels']}

"""
        if validation['errors']:
            result_text += "Errors Found:\n"
            for error in validation['errors'][:20]:  # Show first 20
                result_text += f"  {error['file']}:{error['line']} - {error['error']}\n"

        self.annotation_stats_text.config(state=tk.NORMAL)
        self.annotation_stats_text.delete("1.0", tk.END)
        self.annotation_stats_text.insert("1.0", result_text)
        self.annotation_stats_text.config(state=tk.DISABLED)

    # Training methods
    def start_training(self):
        """Start model training"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        if not messagebox.askyesno("Confirm", "Start training? This may take a while."):
            return

        self.training_module = TrainingModule(
            self.config.training,
            self.config.collection.dataset_path,
            lambda msg, level: self.update_training_log(msg, level)
        )

        self.training_progress.start()
        self.training_status_label.config(text="Status: Training...")

        # Run in thread
        threading.Thread(target=self._train_model, daemon=True).start()

    def _train_model(self):
        """Training thread"""
        success = self.training_module.start_training(background=False)

        self.root.after(0, self.training_progress.stop)

        if success:
            self.root.after(0, lambda: self.training_status_label.config(text="Status: Training completed!"))
            self.root.after(0, lambda: self.update_status("Training completed successfully"))
        else:
            self.root.after(0, lambda: self.training_status_label.config(text="Status: Training failed"))
            self.root.after(0, lambda: self.update_status("Training failed"))

    def analyze_dataset(self):
        """Analyze dataset before training"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        if not self.training_module:
            self.training_module = TrainingModule(
                self.config.training,
                self.config.collection.dataset_path,
                None
            )

        analysis = self.training_module.analyze_dataset()

        text = f"""
Dataset Analysis:
-----------------
Total Images: {analysis['total_images']}
Total Annotations: {analysis['total_annotations']}
Avg Annotations/Image: {analysis['avg_annotations_per_image']:.1f}

Class Distribution:
-------------------
"""
        for class_name, count in sorted(analysis['class_distribution'].items()):
            text += f"{class_name}: {count}\n"

        if analysis['recommendations']:
            text += "\nRecommendations:\n----------------\n"
            for rec in analysis['recommendations']:
                text += f"• {rec}\n"

        self.training_log.config(state=tk.NORMAL)
        self.training_log.delete("1.0", tk.END)
        self.training_log.insert("1.0", text)
        self.training_log.config(state=tk.DISABLED)

    def show_training_results(self):
        """Show training results"""
        if not self.training_module:
            messagebox.showinfo("Info", "No training has been performed yet")
            return

        results = self.training_module.get_training_results()

        if not results['training_completed']:
            messagebox.showinfo("Info", "Training not completed yet")
            return

        text = f"""
Training Results:
-----------------
Model Path: {results['model_path']}
Training Completed: {results['training_completed']}

Metrics:
--------
"""
        if results['metrics']:
            for key, value in results['metrics'].items():
                text += f"{key}: {value}\n"

        self.training_log.config(state=tk.NORMAL)
        self.training_log.delete("1.0", tk.END)
        self.training_log.insert("1.0", text)
        self.training_log.config(state=tk.DISABLED)

    # Evaluation methods
    def evaluate_model(self):
        """Evaluate trained model on dataset"""
        if not self.config:
            messagebox.showerror("Error", "Please save configuration first")
            return

        # Check if training is complete
        if not self.training_module:
            self.training_module = TrainingModule(self.config.training, self.config.collection.dataset_path, None)

        model_path = self.training_module.get_best_model_path()
        if not model_path:
            messagebox.showerror("Error", "No trained model found. Please train a model first.")
            return

        self.evaluation_module = EvaluationModule(
            self.config.evaluation,
            str(model_path),
            lambda msg, level: self.update_status(msg)
        )

        # Get dataset yaml path
        dataset_yaml = Path(self.config.training.output_dir).expanduser() / "dataset.yaml"

        if not dataset_yaml.exists():
            messagebox.showerror("Error", f"Dataset YAML not found: {dataset_yaml}")
            return

        self.update_status("Running evaluation...")

        # Run in thread
        threading.Thread(target=self._evaluate_model, args=(str(dataset_yaml),), daemon=True).start()

    def _evaluate_model(self, dataset_yaml_path):
        """Evaluation thread"""
        metrics = self.evaluation_module.evaluate_on_dataset(dataset_yaml_path)

        text = f"""
Evaluation Results:
-------------------
mAP50: {metrics.get('mAP50', 0):.3f}
mAP50-95: {metrics.get('mAP50-95', 0):.3f}
Precision: {metrics.get('precision', 0):.3f}
Recall: {metrics.get('recall', 0):.3f}

Per-Class Metrics:
------------------
"""
        for class_name, class_metrics in metrics.get('class_metrics', {}).items():
            text += f"{class_name}: AP={class_metrics.get('AP', 0):.3f}\n"

        self.root.after(0, lambda: self._update_evaluation_text(text))

    def _update_evaluation_text(self, text):
        """Update evaluation results text"""
        self.evaluation_results_text.config(state=tk.NORMAL)
        self.evaluation_results_text.delete("1.0", tk.END)
        self.evaluation_results_text.insert("1.0", text)
        self.evaluation_results_text.config(state=tk.DISABLED)

    def test_on_image(self):
        """Test model on single image"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not filename:
            return

        if not self.evaluation_module:
            model_path = self.training_module.get_best_model_path() if self.training_module else None
            if not model_path:
                messagebox.showerror("Error", "No trained model available")
                return

            self.evaluation_module = EvaluationModule(
                self.config.evaluation,
                str(model_path),
                None
            )

        result = self.evaluation_module.test_on_image(filename, save_result=True)

        if result:
            text = f"""
Test Results for: {Path(filename).name}
-----------------
Detections: {len(result['detections'])}

"""
            for i, det in enumerate(result['detections'], 1):
                text += f"{i}. {det['class_name']}: {det['confidence']:.3f}\n"

            if result['output_path']:
                text += f"\nResult saved to: {result['output_path']}"

            self.evaluation_results_text.config(state=tk.NORMAL)
            self.evaluation_results_text.delete("1.0", tk.END)
            self.evaluation_results_text.insert("1.0", text)
            self.evaluation_results_text.config(state=tk.DISABLED)

    def test_on_directory(self):
        """Test model on directory of images"""
        directory = filedialog.askdirectory(title="Select Directory")

        if not directory:
            return

        if not self.evaluation_module:
            model_path = self.training_module.get_best_model_path() if self.training_module else None
            if not model_path:
                messagebox.showerror("Error", "No trained model available")
                return

            self.evaluation_module = EvaluationModule(
                self.config.evaluation,
                str(model_path),
                None
            )

        summary = self.evaluation_module.test_on_directory(directory)

        text = f"""
Test Results for Directory: {directory}
-----------------
Total Images: {summary.get('total_images', 0)}
Images with Detections: {summary.get('images_with_detections', 0)}
Total Detections: {summary.get('total_detections', 0)}
Avg Detections/Image: {summary.get('avg_detections_per_image', 0):.1f}
        """

        self.evaluation_results_text.config(state=tk.NORMAL)
        self.evaluation_results_text.delete("1.0", tk.END)
        self.evaluation_results_text.insert("1.0", text)
        self.evaluation_results_text.config(state=tk.DISABLED)

    # Deployment methods
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch models", "*.pt")]
        )

        if filename:
            self.deploy_model_path_var.set(filename)

    def start_deployment(self):
        """Start ROS2 deployment"""
        model_path = Path(self.deploy_model_path_var.get())

        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found: {model_path}")
            return

        # Update config
        self.config.deployment.model_path = str(model_path)
        self.config.deployment.camera_topic = self.camera_topic_var.get()

        self.deployment_module = DeploymentModule(
            self.config.deployment,
            lambda msg, level: self.update_status(msg)
        )

        if self.deployment_module.start_deployment():
            self.deployment_status_label.config(text="Status: Deployed ✅")
            self.update_status("Deployment started")

    def stop_deployment(self):
        """Stop ROS2 deployment"""
        if self.deployment_module:
            self.deployment_module.stop_deployment()
            self.deployment_status_label.config(text="Status: Stopped")
            self.update_status("Deployment stopped")

    def show_deployment_info(self):
        """Show deployment information"""
        if not self.deployment_module:
            messagebox.showinfo("Info", "Deployment not started yet")
            return

        status = self.deployment_module.get_status()

        info_text = f"""
Deployment Status:
------------------
Running: {status['is_running']}
Model: {status['model_path']}
Camera Topic: {status['camera_topic']}
Output Topic: {status['output_topic']}
Detection Topic: {status['detection_topic']}
        """

        messagebox.showinfo("Deployment Status", info_text)

    # Utility methods
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)

    def update_collection_log(self, message, level):
        """Update collection log"""
        self.collection_log.config(state=tk.NORMAL)
        self.collection_log.insert(tk.END, f"[{level.upper()}] {message}\n")
        self.collection_log.see(tk.END)
        self.collection_log.config(state=tk.DISABLED)

    def update_training_log(self, message, level):
        """Update training log"""
        self.training_log.config(state=tk.NORMAL)
        self.training_log.insert(tk.END, f"[{level.upper()}] {message}\n")
        self.training_log.see(tk.END)
        self.training_log.config(state=tk.DISABLED)

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    app = UnifiedPipelineGUI()
    app.run()


if __name__ == '__main__':
    main()
