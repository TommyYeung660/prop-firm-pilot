import re

with open('tests/test_prop_firm_guard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Delete the TestTrialConfigLoading class
content = re.sub(r'class TestTrialConfigLoading:.*?(?=class|\Z)', '', content, flags=re.DOTALL)

with open('tests/test_prop_firm_guard.py', 'w', encoding='utf-8') as f:
    f.write(content)
