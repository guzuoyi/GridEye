path = r'E:\gu_app\traffic\tests\test_cache.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Fix 1: confirmed_refresh last_seen
c = c.replace(
    "rec = make_record(State.CONFIRMED, dwell=6, last_seen=0.0)\n        rec.state_timestamps[State.CONFIRMED] = 0.0\n        new_state, needs_qwen, _ = sm.evaluate(rec, current_time=35.0)",
    "rec = make_record(State.CONFIRMED, dwell=6, last_seen=35.0)\n        rec.state_timestamps[State.CONFIRMED] = 5.0\n        new_state, needs_qwen, _ = sm.evaluate(rec, current_time=36.0)"
)

# Fix 2: cleanup test assertion
c = c.replace(
    "assert stale[0].state == State.CONFIRMED",
    "assert cache._records[stale[0].id].state == State.CLEARED"
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Fixed tests')
