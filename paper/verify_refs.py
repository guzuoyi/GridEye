"""Verify each PDF matches its expected reference content."""
import fitz, os

REF_DIR = "D:/Users/guzuoyi/Documents/WPSDrive/382961466/WPS云盘/顾作一资料/在职研究生相关/研一下/物联网安全/参考文献"

# Expected content for each file
expected = {
    "ref01_platoon_survey.pdf": ("platoon", "Jia"),
    "ref02_petit_cyberattacks.pdf": ("cyberattack", "Petit"),
    "ref03_usv_surrounding.pdf": ("surrounding", "Wang"),
    "ref04_main_paper.pdf": ("Hierarchical Game", "Xia"),
    "ref05_uav_game.pdf": ("Quadrotor", "Geng"),
    "ref06_secure_control_ncs.pdf": ("Secure Control", "Sargolzaei"),
    "ref07_resource_aware_platoon.pdf": ("resource-aware", "Yang"),
    "ref08_adaptive_cps.pdf": ("Adaptive Control", "Ren"),
    "ref09_fuzzy_resilient.pdf": ("Fuzzy Resilient", "Pan"),
    "ref10_fault_estimation.pdf": ("Fault Estimation", "Xu"),
    "ref11_network_gain.pdf": ("Network Gain", "Marelli"),
}

all_ok = True
for fname, (keyword, author) in expected.items():
    path = os.path.join(REF_DIR, fname)
    if not os.path.exists(path):
        print(f"❌ {fname}: 文件不存在")
        all_ok = False
        continue
    
    size = os.path.getsize(path)
    try:
        doc = fitz.open(path)
        pages = len(doc)
        # Get first page text
        text = ""
        for p in range(min(pages, 2)):
            text += doc[p].get_text()
        doc.close()
        
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        if keyword_lower in text_lower or author.lower() in text_lower:
            # Show first meaningful line
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 30]
            first_line = lines[0][:100] if lines else "(empty)"
            print(f"✅ {fname:<40} {size//1024:>5}KB  {pages:>2}页 | {first_line}")
        else:
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 30]
            first_line = lines[0][:100] if lines else "(empty)"
            print(f"⚠️ {fname:<40} {size//1024:>5}KB  {pages:>2}页 | 未找到'{keyword}' - 实际: {first_line}")
            all_ok = False
    except Exception as e:
        print(f"❌ {fname}: 读取失败 - {e}")
        all_ok = False

print(f"\n{'='*60}")
print(f"结论: {'✅ 全部一致' if all_ok else '⚠️ 存在不匹配'}")
