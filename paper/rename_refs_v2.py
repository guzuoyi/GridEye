"""Clean rename: fix all PDF files to match citation order.
Map old numbers to new numbers, keeping descriptive suffix.
"""
import os, shutil, glob, time

REF_DIR = "D:/Users/guzuoyi/Documents/WPSDrive/382961466/WPS云盘/顾作一资料/在职研究生相关/研一下/物联网安全/参考文献"

# Mapping old_num -> new_num with papers and bib key
# old=pdf filename number, new=citation order number
rename_map = [
    (1,  1,  "platoon_survey"),
    (2,  2,  "petit_cyberattacks"),
    (5,  3,  "uav_game"),
    (6,  4,  "secure_control_ncs"),
    (7,  5,  "resource_aware_platoon"),
    (8,  6,  "adaptive_cps"),
    (4,  7,  "main_paper"),
    (10, 8,  "fault_estimation"),
    (3,  9,  "usv_surrounding"),
    (11, 10, "network_gain"),
    (9,  11, "fuzzy_resilient"),
]

# Step 1: Rename all to temp (avoid conflicts)
print("Step 1: 暂存所有文件...")
temps = []
for old_n, new_n, suffix in rename_map:
    pattern = os.path.join(REF_DIR, f"ref{old_n:02d}_*.pdf")
    files = glob.glob(pattern)
    # Also check for temp files from previous attempt
    if not files:
        pattern2 = os.path.join(REF_DIR, f"__temp_{old_n:02d}_*")
        files = glob.glob(pattern2)
    if not files:
        print(f"  未找到 ref{old_n:02d}_* 或 __temp_{old_n:02d}_*")
        continue
    old_path = files[0]
    temp_name = f"__{old_n:02d}to{new_n:02d}.pdf"
    temp_path = os.path.join(REF_DIR, temp_name)
    for attempt in range(10):
        try:
            shutil.move(old_path, temp_path)
            temps.append((temp_path, new_n, suffix))
            print(f"  {os.path.basename(old_path):<40} → {temp_name}")
            break
        except PermissionError:
            if attempt == 0:
                print(f"  等待文件释放: {os.path.basename(old_path)}")
            time.sleep(1)
    else:
        print(f"  ❌ 无法移动: {os.path.basename(old_path)} (请关闭打开的PDF文件)")

# Step 2: Rename temp to final
print("\nStep 2: 重命名为最终名称...")
for temp_path, new_n, suffix in temps:
    new_name = f"ref{new_n:02d}_{suffix}.pdf"
    new_path = os.path.join(REF_DIR, new_name)
    for attempt in range(10):
        try:
            shutil.move(temp_path, new_path)
            sz = os.path.getsize(new_path) // 1024
            print(f"  {os.path.basename(temp_path):<16} → {new_name:<40} ({sz}KB)")
            break
        except PermissionError:
            time.sleep(1)
    else:
        print(f"  ❌ 无法重命名: {os.path.basename(temp_path)}")

# Final listing
print(f"\n{'='*60}")
print("最终目录:")
print(f"{'='*60}")
for f in sorted(os.listdir(REF_DIR)):
    path = os.path.join(REF_DIR, f)
    if os.path.isfile(path) and f.endswith('.pdf'):
        sz = os.path.getsize(path) // 1024
        print(f"  {f:<50} {sz:>5}KB")
    elif os.path.isfile(path) and f.startswith('__'):
        # Clean up any remaining temp files
        os.remove(path)
        print(f"  🗑️ 删除临时文件: {f}")
