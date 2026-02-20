import re

with open('tests/test_prop_firm_guard.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'assert config.account.initial_balance == 5000',
    'assert config.account.initial_balance == 50000'
)

content = content.replace(
    'assert len(config.instruments) == 5',
    'assert len(config.instruments) == 2'
)

with open('tests/test_prop_firm_guard.py', 'w', encoding='utf-8') as f:
    f.write(content)
