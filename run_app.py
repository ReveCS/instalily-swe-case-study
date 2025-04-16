import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the app
from backend.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 