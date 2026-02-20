import re

with open('tests/test_prop_firm_guard_e8_one.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'assert result.rule_name == "MAX_DRAWDOWN"',
    'assert result.rule_name == "MAX_DRAWDOWN_DYNAMIC"'
)

content = content.replace(
    'trade = _trade(risk=10.0) # 4950 - 10 = 4940 < 4945 safe floor!',
    'trade = _trade(risk=20.0) # Projected loss: 250 + 20 = 270 > 265.2 (Safe Limit)'
)

with open('tests/test_prop_firm_guard_e8_one.py', 'w', encoding='utf-8') as f:
    f.write(content)
