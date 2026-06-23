import sys
path = r'E:\gu_app\traffic\src\cache\cache.py'
with open(path, encoding='utf-8') as f:
    c = f.read()
# 1) state_ts → state_timestamps
c = c.replace('state_ts', 'state_timestamps')
# 2) Add risk_cleared + need_cleanup to QwenResult
old = '    reasoning: str = ""\n\n\n@dataclass'
new = '    reasoning: str = ""\n    risk_cleared: bool = False\n    need_cleanup: bool = False\n\n\n@dataclass'
c = c.replace(old, new)
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Fixed:', c.count('state_timestamps'), 'state_timestamps refs, risk_cleared:', 'risk_cleared' in c)
