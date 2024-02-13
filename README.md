# Cirrosis: Predicción de Status del Paciente

![title](/notebooks/images/cirrhosis.png)

## Objetivos

- **Mejorar la precisión del diagnóstico:** Identificar con precisión el estado del paciente, lo que puede ayudar a los médicos a tomar mejores decisiones sobre el tratamiento y la gestión de la enfermedad.
- **Identificar factores predictivos clave:** Revelar qué características o factores clínicos tienen mayor influencia en la progresión de la cirrosis y en los resultados de los pacientes.

## Contexto

La cirrosis es una enfermedad hepática crónica que puede tener consecuencias graves para la salud de los pacientes. Es importante desarrollar modelos de aprendizaje automático que puedan predecir los resultados de la cirrosis y ayudar a los médicos en el manejo clínico de la enfermedad.

## Requisitos

Python 3.11

**Dependencias**:

- numpy
- pandas
- matplotlib
- ydata-profiling
- matplotlib
- seaborn
- scikit-learn
- xgboost
- metaflow
- mlflow
- bentoml

## Instalacion

```bash
git clone https://github.com/victormhp/cirrhosis-prediction.git
```

```bash
cd cirrhosis-predicdtion
```

Las dependencias se encuentran listadas en el archivo `requirements.txt`

```bash
pip install -r requierements.txt
```

## Despliegue

Construir bundle de bentoml y asignar una id.

```bash
cd ./service
bentoml build
```

Crear imagen de Docker

```bash
bentoml containerize <bento-id>
```

**Note**: La id es que se le asigno al bundle.

Ejectuar imagen de Docker

```bash
docker run –rm -p 3000:3000 <bundle-id>
```

Si desea probar el funcionamiento del servicio de bentoml puedes correr el notebook `service_test.ipynb`.
