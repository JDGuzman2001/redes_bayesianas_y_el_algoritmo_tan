# Tree Augmented Naïve Bayes (TAN) Implementation

## Overview
This repository contains an implementation of a Bayesian Network using the Tree Augmented Naïve Bayes (TAN) algorithm. The goal is to demonstrate how TAN can be used to predict outcomes based on probabilistic dependencies between variables.

The example provided models a scenario where we predict whether a student will pass an exam (`Pass`) based on the following predictors:
- **StudyTime:** Amount of time spent studying (High, Medium, Low).
- **Stress:** Stress level of the student (High, Low).
- **Sleep:** Amount of sleep the student gets (Sufficient, Insufficient).

## Features
- Learn the structure of the Bayesian Network using the TAN algorithm.
- Calculate Conditional Probability Distributions (CPDs) for all variables.
- Perform inference to predict probabilities for the target variable (`Pass`).

## Technologies Used
- **Python**
- **pgmpy**: A Python library for probabilistic graphical models.
- **pandas**: For data manipulation and preparation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JDGuzman2001/redes_bayesianas_y_el_algoritmo_tan.git
   cd redes_bayesianas_y_el_algoritmo_tan
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes:
   - pgmpy
   - pandas

## Usage

1. Run the main Python script:
   ```bash
   python main.py
   ```

2. The script will:
   - Learn the TAN structure.
   - Display the learned network edges.
   - Output the probabilities of passing (`Pass`) and the CPDs for all variables.

3. Modify the `data` variable in `main.py` to use a custom dataset.

## Example Output

### Learned TAN Structure:
```
[('Pass', 'StudyTime'), ('StudyTime', 'Stress'), ('Stress', 'Sleep')]
```

### Probability of Passing (Pass):
```
+----------+-------------+
| Pass     |   phi(Pass) |
+==========+=============+
| Pass(No) |      0.0000 |
+----------+-------------+
| Pass(Sí) |      1.0000 |
+----------+-------------+
```

### CPDs for Variables:
For example, the CPD for `StudyTime`:
```
+------------------+----------+--------------------+
| Pass             | Pass(No) | Pass(Sí)           |
+------------------+----------+--------------------+
| StudyTime(Alto)  | 0.0      | 0.6666666666666666 |
+------------------+----------+--------------------+
| StudyTime(Bajo)  | 0.5      | 0.0                |
+------------------+----------+--------------------+
| StudyTime(Medio) | 0.5      | 0.3333333333333333 |
+------------------+----------+--------------------+
```





