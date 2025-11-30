# Navigate to your project directory
cd path/to/your/project

# Create virtual environment
python3 -m venv shiny_env

# Activate the environment
source shiny_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
shiny run app.py

# When done, deactivate
deactivate
