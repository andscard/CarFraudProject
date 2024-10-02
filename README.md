# Fraudes en Seguros de Vehículos

Una compañía de seguros para vehículos tiene un registro histórico de sus siniestros identificados como Fraudes y busca mejorar su eficiencia de costos en el proceso de detección de siniestro y fraudes.

​Este proyecto tiene como objetivo implementar un modelo de ML que permita identificar patrones similares en los clientes y predecir cuántos de los nuevos reportes serían considerados fraude y poder automatizar el proceso.

## Metodología

Se utilizó un conjunto de datos de reclamaciones de seguros de automóviles personales que incluye la etiqueta "Fraud Found" que indica si se trata de Fraude o No Fraude. Este conjunto de datos abarca un total de 15,420 reclamaciones realizadas entre enero de 1994 y diciembre de 1996.

**Features de interés**

-   Género y Edad
-   Valor deducible
-   Categría y precio del vehículo
-   Tipo de póliza
-   Número anterior de solicitudes

**​Análisis exploratorio**

![Screenshot](/results/imagen1.png)
​​![Screenshot](/results/imagen2.png)
![Screenshot](/results/imagen3.png)
![Screenshot](/results/imagen4.png)

**Feature Engineering**

Se categorizó el campo "Vehicle Price (Precio Vehículo)" en rangos de precios definidos como : Medium-End, Deluxe, Affordable, High-End

| Vehicle Price | Count |
| ------------- | ----- |
| Medium-End    | 11612 |
| Deluxe        | 2164  |
| Affordable    | 1096  |
| High-End      | 548   |

![Screenshot](/results/imagen5.png)

Se categorizó el campo "Make (Fabricante)" en segmentos de mercado como : Medium-End, Budget, y Luxury

| Make       | Count |
| ---------- | ----- |
| Medium-End | 8589  |
| Budget     | 6326  |
| Luxury     | 505   |

![Screenshot](/results/imagen6.png)

Se categorizó el campo "Age of Policy Holder (Edad del titular de la póliza)" en segmentos de grupos etnarios como : Adults, Young Adults, Seniors y Adolescents.

| Age of Policy Holder | Count |
| -------------------- | ----- |
| Adults               | 13856 |
| Young Adults         | 721   |
| Seniors              | 508   |
| Adolescents          | 335   |

**Data Splitting**

Se decide hacer el stratify con tres estratos para asegurar distribución similar en los dataframes

```
estrato = ["AccidentArea", "Fault", "FraudFound"]

train_data, tmp_data = train_test_split(new_data, test_size=0.3, stratify=new_data[estrato], random_state=42)
val_data, test_data = train_test_split(tmp_data, test_size=0.50, random_state=42)
```

![Screenshot](/results/imagen7.png)

**Encoding**

-   **One Hot Encoding**: Vehicle Cagtegory

```
OneHotEncoder(cols=["VehicleCategory"], use_cat_names=True)
```

-   **Ordinal Encoding**: Maken VehiclePrice, PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder, NumberOfCars,
    BasePolicy

```
ordinal_encoder = OrdinalEncoder(cols=["Make","VehiclePrice","PastNumberOfClaims","AgeOfVehicle","AgeOfPolicyHolder","NumberOfCars","BasePolicy"],
                                 mapping=[{"col": "Make", "mapping": {"Budget": 1, "Medium-End": 2, "Luxury": 3, "unknown": 0}},
                                  {"col": "VehiclePrice", "mapping": {"Affordable": 1, "Medium-End": 2, "High-End": 3,"Deluxe":4, "unknown": 0}},
                                  {"col": "PastNumberOfClaims", "mapping": {"none": 0, "1": 1, "2 to 4": 2, "more than 4": 3, "unknown": 0}},
                                  {"col":"AgeOfVehicle","mapping":{"new": 0, "2 years": 1,"3 years": 2,"4 years": 3,"5 years": 4,"6 years": 5,"7 years": 6,"more than 7": 7}},
                                  {"col":"AgeOfPolicyHolder","mapping":{"Adolescents": 0, "Young Adults": 1,"Adults": 2,"Seniors": 3}},
                                  {"col":"NumberOfCars","mapping":{"1 vehicle": 0, "2 vehicles": 1,"3 to 4": 2,"5 to 8": 3,"more than 8": 4}},
                                  {"col":"BasePolicy","mapping":{"Collision": 0, "Liability": 1,"All Perils": 2}},
                                  {"col":"FraudFound","mapping":{"No": 0, "Yes": 1}},
                                  {"col":"AccidentArea","mapping":{"Urban": 0, "Rural": 1}},
                                  {"col":"Sex","mapping":{"Male": 0, "Female": 1}},
                                  {"col":"Fault","mapping":{"Policy Holder": 0, "Third Party": 1}},
                                  {"col":"PoliceReportFiled","mapping":{"No": 0, "Yes": 1}},
                                  {"col":"WitnessPresent","mapping":{"No": 0, "Yes": 1}},
                                  {"col":"AgentType","mapping":{"External": 0, "Internal": 1}}])
```

**Training Features**

```
training_columns = [
    "Make", "AccidentArea", "Sex",	"Age",	"Fault", "VehicleCategory_Sedan",	"VehicleCategory_Sport",	"VehicleCategory_Utility",
"VehiclePrice", "Deductible",	"DriverRating", "PastNumberOfClaims",	"AgeOfVehicle",
"AgeOfPolicyHolder",	"PoliceReportFiled",	"WitnessPresent",	"AgentType", "NumberOfCars",	"Year",	"BasePolicy"
]
```

**Hyperparameter Tuning**

Se usaron dos modelos para el análisis: Random Forest y XGBOOST

-   RANDOM FOREST

```
rfparam_grid = {
    'n_estimators':[250,100,150],
    'max_depth':[10,8,9],
    'min_samples_split':[50,25,40],
    'min_samples_leaf':[25,15,10],
    'class_weight':['balanced']
}
```

-   RXGBOOST

```
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'gamma': [0, 0.1],
    'scale_pos_weight': [5, 10]
}
```

## Resultados

Random Forest tiene mejor detección de la clase "Fraud: 1" (alto recall), pero muchos falsos positivos (baja precision). Por otro lado XGBoost muestra un equilibrio entre recall y precision en ambas clases y logra mejor accuracy en validación. Dado que se considera mas crítico identificar todos los reclamos aprobados para que estos sean luego resueltos, nuestra métrica de interés es 'recall'.

![Screenshot](/results/imagen10.png)

-   **Matriz de Confusión**

XGBoost tiene un mejor desempeño en la clasificación de la clase "Not Fraud: 0", con una precisión en los TN de 65% frente al 55.6% de Random Forest.
Random Forest, aunque tiene una tasa de TP ligeramente mejor 5.0% vs 4.6%, tiene una tasa significativamente mayor de falsos positivos.

![Screenshot](/results/imagen11.png)

-   **SHAP - BAR PLOT**

Las variables con mayor impacto en el modelo son Fault, seguida por las categorías del vehículo (Sedan, Sport) y BasePolicy. El eje X representa la magnitud promedio de impacto en la predicción.

![Screenshot](/results/imagen12.png)

-   **SHAP - EXPLANATION VALUES**

Valores altos de Fault y VehicleCategory_Sedan tienden a tener un mayor impacto positivo en la predicción del modelo.

![Screenshot](/results/imagen13.png)

-   **SHAP - PLOT WATERFALL**

El valor base es 0.5 (predicción inicial del modelo sin características).
Fault y BasePolicy son características que incrementan la predicción , mientras Age, PastNumberOfClaims, etc, las que disminuyen.
La predicción final es 0.657.

![Screenshot](/results/imagen14.png)

## Implementación en el negocio

​**KPIs**

-   Ahorro total en fraudes detectados (Beneficio del modelo)
-   Costo total de falsos positivos (Pérdida por revisiones innecesarias)
-   Costo total de fraudes no detectados (Pérdida)

​![Screenshot](/results/imagen16.png)

Para una estimación promedio de los KPIs mencionados, se consideraron los siguientes valores:

-   Deducible promedio: $400
-   Costo administrativo cuando no es fraude: $120
-   Costo administrativo cuando se hace una revisión exhaustiva: $800
-   Costo del fraude si no se detecta: $5400

El beneficio inmediato de implementar el modelo : $51,400

​**Resultados**

-   Ahorro Revisón Modelo IA

    $Ahorro Proceso OP$ = $(TN)* Costo Revisión$ =$\$178,320$

-   Costo total de fraudes no detectados (Fraudes Pagados)

    $Costo FN=FN * (Costo Fraude−Deducible) - Ahorro Modelo IA$

-   Costo total de falsos positivos (Costo De Revisión)

    $Costo Total FP$ = $FP * (Costo Administrativo Normal)$ =$\$81,720$

-   Ahorro total en fraudes detectados

    $Ahorro Por Fraude$ = $TN * (Costo Fraude)-$ $\$96,000$

​![Screenshot](/results/imagen15.png)

## Limitaciones

## Conclusiones y recomendaciones

-   Las variables significativas en el modelo son Fault, Vehicle Category y Base Policy.
-   Implementando el modelo, la empresa tiene ahorros de $96.600 por optimización en los procesos y $425.120 por la detección de fraudes. Siendo en total un ahorro del $521.720.

## Future Work

**Nuevos Features**:

-   Meses desde el accidente hasta el dia que se reporta a la aseguradora.
-   Combinar variables categóricas y crear más features.

**Mejora del modelo**:

-   Mover los falsos positivos a verdaderos negativos para aumentar la eficiencia del modelo.
-   Reducir los falsos negativos a verdaderos positivos para reducir los costos del modelo.

**Probar Stacking**:

-   Combinar dos o más modelos para probar un mejor desempeño del modelo, que identifique mejor los patrones para detectar fraudes.
