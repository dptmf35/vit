#!/usr/bin/env python3

import os
import glob
from pathlib import Path
import shutil
import argparse

def update_class_ids(labels_dir, old_classes, new_classes, backup=True):
    """
    Update class IDs in YOLO label files when a new class is inserted
    
    Args:
        labels_dir: Directory containing label files
        old_classes: List of old class names
        new_classes: List of new class names
        backup: Whether to backup original files
    """
    
    labels_dir = Path(labels_dir)
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
    
    # Create mapping from old class IDs to new class IDs
    id_mapping = {}
    
    for old_id, old_class in enumerate(old_classes):
        if old_class in new_classes:
            new_id = new_classes.index(old_class)
            id_mapping[old_id] = new_id
        else:
            print(f"⚠️  Class '{old_class}' not found in new class list")
            return False
    
    print("📊 Class ID Mapping:")
    for old_id, new_id in id_mapping.items():
        old_name = old_classes[old_id]
        new_name = new_classes[new_id]
        if old_id != new_id:
            print(f"  {old_id} ({old_name}) → {new_id} ({new_name}) ✅")
        else:
            print(f"  {old_id} ({old_name}) → {new_id} ({new_name}) (no change)")
    
    # Get all label files
    label_files = list(labels_dir.glob("*.txt"))
    
    if not label_files:
        print("⚠️  No label files found")
        return True
    
    print(f"\n🔄 Processing {len(label_files)} label files...")
    
    updated_count = 0
    error_count = 0
    
    for label_file in label_files:
        try:
            # Backup original file if requested
            if backup:
                backup_file = label_file.with_suffix('.txt.backup')
                shutil.copy2(label_file, backup_file)
            
            # Read original content
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Update class IDs
            updated_lines = []
            file_updated = False
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    updated_lines.append(line + '\n')
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"⚠️  {label_file.name}:{line_num} - Invalid format: {line}")
                    updated_lines.append(line + '\n')
                    continue
                
                try:
                    old_class_id = int(parts[0])
                    
                    if old_class_id in id_mapping:
                        new_class_id = id_mapping[old_class_id]
                        
                        if old_class_id != new_class_id:
                            file_updated = True
                            parts[0] = str(new_class_id)
                        
                        updated_line = ' '.join(parts) + '\n'
                        updated_lines.append(updated_line)
                    else:
                        print(f"⚠️  {label_file.name}:{line_num} - Unknown class ID: {old_class_id}")
                        updated_lines.append(line + '\n')
                
                except ValueError:
                    print(f"⚠️  {label_file.name}:{line_num} - Invalid class ID: {parts[0]}")
                    updated_lines.append(line + '\n')
            
            # Write updated content
            with open(label_file, 'w') as f:
                f.writelines(updated_lines)
            
            if file_updated:
                updated_count += 1
                print(f"✅ Updated: {label_file.name}")
            
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {label_file.name}: {e}")
    
    print(f"\n📈 Summary:")
    print(f"  Total files: {len(label_files)}")
    print(f"  Updated files: {updated_count}")
    print(f"  Errors: {error_count}")
    print(f"  Backup files created: {len(label_files) if backup else 0}")
    
    return error_count == 0

def main():
    parser = argparse.ArgumentParser(description='Update YOLO class IDs when new classes are added')
    parser.add_argument('--labels-dir', type=str, default='~/yolo_dataset/labels',
                       help='Directory containing label files')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')
    
    args = parser.parse_args()
    
    # Old class list (21 classes)
    old_classes = [
        'bed', 'cabinet', 'carpet', 'chair', 'closet', 'curtain', 'desk', 'door', 
        'fridge', 'gas stove', 'hanger', 'lamp', 'microwave', 'nightstand', 
        'plant', 'shelf', 'sofa', 'table', 'tv', 'window', 'vanity'
    ]
    
    # New class list (22 classes with 'countertop' added)
    new_classes = [
        'bed', 'cabinet', 'carpet', 'chair', 'closet', 'countertop', 'curtain', 
        'desk', 'door', 'fridge', 'gas stove', 'hanger', 'lamp', 'microwave', 
        'nightstand', 'plant', 'shelf', 'sofa', 'table', 'tv', 'window', 'vanity'
    ]
    
    print("🔄 YOLO Class ID Updater")
    print("=" * 40)
    print(f"Labels directory: {args.labels_dir}")
    print(f"Old classes: {len(old_classes)}")
    print(f"New classes: {len(new_classes)}")
    print(f"Added class: 'countertop' (ID: {new_classes.index('countertop')})")
    print("=" * 40)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("\nClasses that will be affected:")
        for old_id, old_class in enumerate(old_classes):
            new_id = new_classes.index(old_class)
            if old_id != new_id:
                print(f"  {old_id} ({old_class}) → {new_id} ({old_class})")
        return
    
    # Confirm before proceeding
    response = input("\n⚠️  This will modify label files. Continue? [y/N]: ").strip().lower()
    if response != 'y':
        print("❌ Cancelled.")
        return
    
    # Expand user path
    labels_dir = Path(args.labels_dir).expanduser()
    
    # Update class IDs
    success = update_class_ids(
        labels_dir=labels_dir,
        old_classes=old_classes,
        new_classes=new_classes,
        backup=not args.no_backup
    )
    
    if success:
        print("\n✅ Class ID update completed successfully!")
        print("\n📝 Next steps:")
        print("1. Verify the changes with dataset reviewer")
        print("2. Add new 'countertop' annotations if needed")
        print("3. Re-train your model with the updated dataset")
    else:
        print("\n❌ Some errors occurred during update.")
        print("Check the output above and fix any issues.")

if __name__ == '__main__':
    main() 