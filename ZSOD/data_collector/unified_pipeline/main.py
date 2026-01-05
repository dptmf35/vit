#!/usr/bin/env python3

"""
Unified ML Pipeline - Main Entry Point
From Data Collection to Deployment

This application provides a complete pipeline for:
1. Data Collection from ROS2 camera topics
2. Annotation Review and Editing
3. Model Training
4. Model Evaluation
5. ROS2 Deployment

Author: Auto-generated Pipeline System
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from gui.main_window import main

if __name__ == '__main__':
    print("=" * 60)
    print("Unified ML Pipeline - Data Collection to Deployment")
    print("=" * 60)
    print("\nStarting GUI application...")
    print("\nFeatures:")
    print("  • 📸 Data Collection from ROS2 topics")
    print("  • ✏️  Annotation Review & Editing")
    print("  • 🎓 Model Training (YOLO11)")
    print("  • 📊 Model Evaluation")
    print("  • 🚀 ROS2 Deployment")
    print("\n" + "=" * 60 + "\n")

    main()
