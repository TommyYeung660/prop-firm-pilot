import re

with open('tests/test_matchtrader_client.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '"offer": {"system": {"uuid": "first_uuid"}}}',
    '"offer": {"system": {"uuid": "first_uuid"}}'
)

with open('tests/test_matchtrader_client.py', 'w', encoding='utf-8') as f:
    f.write(content)
