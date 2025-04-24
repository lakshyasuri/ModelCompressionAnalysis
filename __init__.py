import sys
import os

project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
print(f"PYTHONPATH set: {project_root}")