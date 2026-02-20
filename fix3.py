import re

with open('tests/test_prop_firm_guard.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'assert config.execution.max_positions == 2',
    'assert config.execution.max_positions == 1'
)

with open('tests/test_prop_firm_guard.py', 'w', encoding='utf-8') as f:
    f.write(content)
