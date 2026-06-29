"""Rename PDF files to match compiled reference numbering.
Mapping: old_ref_number → new_ref_number (based on citation order in main.tex)
"""
import os, shutil

REF_DIR = "D:/Users/guzuoyi/Documents/WPSDrive/382961466/WPS云盘/顾作一资料/在职研究生相关/研一下/物联网安全/参考文献"

# Mapping: (old_bib_key, old_file_num, new_file_num, paper_desc)
mapping = [
    ("jia2016",       1,  1,  "队列综述"),
    ("petit2015",     2,  2,  "车辆攻击"),
    ("geng2025",      5,  3,  "UAV博弈"),
    ("sargolzaei2021",6,  4,  "NCS安全控制"),
    ("yang2022",      7,  5,  "资源感知编队"),
    ("ren2020",       8,  6,  "自适应CPS"),
    ("xia2026",       4,  7,  "原文"),
    ("xu2022",       10,  8,  "故障估计"),
    ("wang2024",      3,  9,  "USV控制"),
    ("marelli2023",  11, 10,  "网络增益"),
    ("pan2024",       9, 11,  "模糊弹性控制"),
]

print("重命名计划:")
print(f"{'旧文件名':<45} → {'新文件名':<45}  论文")
print("-" * 100)
for _, old_n, new_n, desc in mapping:
    old = f"ref{old_n:02d}_*.pdf"
    new = f"ref{new_n:02d}_*.pdf"
    # Find actual filename
    import glob
    old_files = glob.glob(os.path.join(REF_DIR, f"ref{old_n:02d}_*.pdf"))
    if old_files:
        old_name = os.path.basename(old_files[0])
        # Extract the suffix after the number
        parts = old_name.split('_', 1)
        suffix = parts[1] if len(parts) > 1 else "paper.pdf"
        new_name = f"ref{new_n:02d}_{suffix}"
        print(f"  {old_name:<45} → {new_name:<45}  {desc}")

print("自动执行重命名...")

# Step 1: Rename to temp names (avoid conflicts)
temp_names = []
for _, old_n, new_n, desc in mapping:
    old_files = glob.glob(os.path.join(REF_DIR, f"ref{old_n:02d}_*.pdf"))
    if old_files:
        old_path = old_files[0]
        temp_path = os.path.join(REF_DIR, f"__temp_{old_n:02d}_{new_n:02d}.pdf")
        import time
        for attempt in range(5):
            try:
                shutil.move(old_path, temp_path)
                break
            except PermissionError:
                time.sleep(1)
        else:
            print(f"  ❌ 无法移动: {old_path} (文件被占用)")
            continue
        temp_names.append((temp_path, old_n, new_n, desc))
        print(f"  暂存: {os.path.basename(old_path)} → {os.path.basename(temp_path)}")

# Step 2: Rename to final names
for temp_path, old_n, new_n, desc in temp_names:
    old_files = glob.glob(os.path.join(REF_DIR, f"ref{old_n:02d}_*.pdf"))
    old_name = os.path.basename(old_files[0]) if old_files else f"ref{old_n:02d}.pdf"
    parts = old_name.split('_', 1)
    suffix = parts[1] if len(parts) > 1 else "paper.pdf"
    new_name = f"ref{new_n:02d}_{suffix}"
    new_path = os.path.join(REF_DIR, new_name)
    for attempt in range(5):
        try:
            shutil.move(temp_path, new_path)
            break
        except PermissionError:
            time.sleep(1)
    else:
        print(f"  ❌ 无法移动: {temp_path} (文件被占用)")
        continue
    print(f"  完成: {os.path.basename(temp_path)} → {new_name}")

print("\n最终目录:")
for f in sorted(os.listdir(REF_DIR)):
    if f.endswith('.pdf'):
        sz = os.path.getsize(os.path.join(REF_DIR, f)) // 1024
        print(f"  {f:<50} {sz:>5}KB")
