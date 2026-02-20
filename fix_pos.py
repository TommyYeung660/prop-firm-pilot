import re

with open('tests/test_position_sizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'volume = sizer.calculate_volume("USDUSD", equity, sl_pips)',
    'volume = sizer.calculate_volume("USDJPY", equity, sl_pips)'
)

content = content.replace(
    'volume = sizer.max_volume_for_risk("EURUSD", 10.0, 50.0)',
    'volume = sizer.max_volume_for_risk("EURUSD", 1.0, 50.0)'
)

with open('tests/test_position_sizer.py', 'w', encoding='utf-8') as f:
    f.write(content)
