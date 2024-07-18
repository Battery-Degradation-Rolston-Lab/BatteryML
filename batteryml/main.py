# main.py (continued)
from batteryml.builders import MODELS

# Print all registered classes
for class_name in MODELS.class_mapping:
    print(class_name)
