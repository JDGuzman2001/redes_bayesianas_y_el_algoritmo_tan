import pandas as pd
from pgmpy.estimators import TreeSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# **Dataset**
data = pd.DataFrame({
    "StudyTime": ["Alto", "Bajo", "Medio", "Alto", "Medio"],
    "Sleep": ["Suficiente", "Insuficiente", "Suficiente", "Insuficiente", "Insuficiente"],
    "Stress": ["Bajo", "Alto", "Bajo", "Alto", "Bajo"],
    "Pass": ["Sí", "No", "Sí", "Sí", "No"]
})

# Convertir a variables categóricas
for column in data.columns:
    data[column] = data[column].astype("category")

# **Identificación de la estructura del árbol**
# Usamos el algoritmo Chow-Liu para construir un árbol basado en dependencias mutuas
estimator = TreeSearch(data, root_node="Pass")  # La clase 'Pass' será la raíz
tan_model = estimator.estimate()  # Cambiado a estimate()

# **Entrenamiento del modelo Bayesian Network**
bn_model = BayesianNetwork(tan_model.edges())
bn_model.fit(data, estimator=MaximumLikelihoodEstimator)

# **Visualización de las estructuras aprendidas**
print("Estructura aprendida (TAN):", tan_model.edges())

inference = VariableElimination(bn_model)

# Ejemplo de predicción: Probabilidad de aprobar dado que se estudió "Alto" y hubo "Estrés bajo"
query = inference.query(variables=["Pass"], evidence={"StudyTime": "Alto", "Stress": "Bajo"})
print("\nProbabilidad de aprobar (Pass):")
print(query)

# **Probabilidades condicionales**
for node in bn_model.nodes():
    print(f"\nTabla CPD para {node}:")
    print(bn_model.get_cpds(node))
