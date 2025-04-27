# Musical Intuition in Humans and Machines

This project explores the ability of machine learning models to predict melodic resolutions in the context of authentic cadences using music theory principles and experimental data.

### Authors
- Nicole Lam (nicole.lam@yale.edu)
- Breanna Nguyen (breanna.nguyen@yale.edu)

### Project Structure
- `data/`: Contains processed and raw musical data in MIDI format
- `src/`: Core project code for data processing, modeling, and evaluation
- `notebooks/`: Jupyter notebooks for data exploration
- `tests/`: Unit tests for core components
- `main.py`: Entry point for running training and evaluation

### Goals
- Predict the final note in a melodic sequence using linear and ridge regression
- Evaluate model performance against human baseline
- Explore how musical context (key, dominant chord) improves prediction

### Setup
```bash
pip install -r requirements.txt

### References
- Behavioral data from: [https://osf.io/wgz9t/]
