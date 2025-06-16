# 🔄 Dynamic Mode Switching Guide

YOLO Dataset Collector now supports **real-time switching** between Test Mode and Collection Mode without restarting the application!

## 🎯 Features

### Mode Types
- **🔍 Test Mode**: Detection only, results published to `/yolo_detection_rviz` (no data saving)
- **💾 Collection Mode**: Full dataset collection with YOLO format annotations

### Switching Methods
1. **Keyboard Controls** (Runtime)
2. **ROS2 Service Calls** (Remote)
3. **Python Toggle Script** (Convenience)

---

## 🚀 Quick Start

### 1. Launch Interactive Mode
```bash
./start_collection.sh
# Select option 5) Interactive Mode
```

### 2. Runtime Controls

#### Keyboard Controls
- `t` - Switch to **Test Mode**
- `c` - Switch to **Collection Mode** 
- `s` - Show current **Status**
- `q` - **Quit** application

#### Service Controls (New Terminal)
```bash
# Switch to test mode
python3 toggle_mode.py test

# Switch to collection mode  
python3 toggle_mode.py collect

# Check current status
python3 toggle_mode.py status
```

#### ROS2 Service Direct Call
```bash
# Enable collection mode (data: true)
ros2 service call /toggle_collection_mode std_srvs/srv/SetBool "{data: true}"

# Enable test mode (data: false)
ros2 service call /toggle_collection_mode std_srvs/srv/SetBool "{data: false}"
```

### 3. Monitor Mode Status
```bash
# Real-time mode monitoring
ros2 topic echo /collector_mode_status
```

---

## 📋 Usage Examples

### Example 1: Start in Test Mode, Switch to Collection
```bash
# Terminal 1: Start collector
./start_collection.sh  # Choose option 4 (Test Mode)

# Terminal 2: Switch to collection when ready
python3 toggle_mode.py collect
```

### Example 2: Interactive Development Workflow
```bash
# Terminal 1: Start interactive mode
./start_collection.sh  # Choose option 5 (Interactive)

# Use keyboard controls:
# 't' - Test your detection settings
# 'c' - Start collecting when settings are good
# 's' - Check collection statistics
# 't' - Pause collection temporarily
# 'c' - Resume collection
```

### Example 3: Remote Monitoring & Control
```bash
# Terminal 1: Start collector
python3 run_dataset_collector.py

# Terminal 2: Monitor status
ros2 topic echo /collector_mode_status

# Terminal 3: Remote control
python3 toggle_mode.py status
python3 toggle_mode.py test     # Pause collection
python3 toggle_mode.py collect  # Resume collection
```

---

## 🔧 Technical Details

### ROS2 Topics & Services

#### Published Topics
- `/yolo_detection_rviz` - Detection visualization (always published)
- `/collector_mode_status` - Current mode status (`TEST_MODE` or `COLLECTION_MODE`)

#### Services
- `/toggle_collection_mode` - Switch modes (`std_srvs/srv/SetBool`)

#### Service Interface
```
# Request
bool data    # true = collection mode, false = test mode

# Response  
bool success # operation success
string message # status message
```

### Mode Switching Logic
- **Test → Collection**: Automatically sets up dataset directories if needed
- **Collection → Test**: Preserves collected data, stops new collection
- **Status Persistence**: Mode state maintained until manually changed
- **Thread Safety**: All mode switches are thread-safe

### File System Behavior
- **Test Mode**: No file I/O operations
- **Collection Mode**: Creates directory structure on first switch
- **Data Safety**: Existing collections are never overwritten

---

## 🎮 Integration Examples

### Python Script Integration
```python
import rclpy
from std_srvs.srv import SetBool

# Switch to collection mode
def enable_collection():
    # ... ROS2 service call code ...
    pass

# Your application logic
def my_data_collection_workflow():
    enable_collection()  # Start collecting
    time.sleep(60)       # Collect for 1 minute
    disable_collection() # Switch to test mode
```

### Shell Script Automation
```bash
#!/bin/bash
echo "Starting automated collection..."

# Start in test mode
python3 toggle_mode.py test

# Test detection for 10 seconds
sleep 10

# Switch to collection mode
python3 toggle_mode.py collect

# Collect for 5 minutes
sleep 300

# Switch back to test mode
python3 toggle_mode.py test

echo "Collection complete!"
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Service not available**
   ```bash
   # Check if collector is running
   ros2 service list | grep toggle_collection_mode
   ```

2. **Keyboard input not responding**
   ```bash
   # Make sure terminal has focus
   # Press keys in the terminal running the collector
   ```

3. **Mode status not updating**
   ```bash
   # Check topic echo
   ros2 topic echo /collector_mode_status --once
   ```

### Debug Commands
```bash
# Check all collector-related topics
ros2 topic list | grep collector

# Check service availability
ros2 service list | grep toggle

# Test service manually
ros2 service call /toggle_collection_mode std_srvs/srv/SetBool "{data: false}"
```

---

## 🎉 Benefits

- **Flexibility**: Switch modes without restarting
- **Efficiency**: Test detection settings before collection
- **Safety**: Prevent accidental data overwriting
- **Monitoring**: Real-time status visibility
- **Automation**: Scriptable mode switching
- **Multi-terminal**: Control from anywhere 