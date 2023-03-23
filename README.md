# IA: Lab 7 KMeans

## Autores
- Luis Pedro Gonzalez Aldana
- Jessica Ortiz
- Rebecca Smith
- Andrea Lam
## Files
Los archivos importantes son:
- main.py
- Task1_1.py
## Task 1
### Exploracion de datos

#### Encoding y seleccion de v  ariables
- **TransactionID**: ID de transacción (no útil para análisis). *Se elimina*
- **CustomerID**: Como identificador no representa un valor de interés. *Se elimina*
- **CustomerDOB**: Fecha de cumpleaños del cliente. *Se elimina*
- **CustGender**: Genero del cliente, valor categorico de posible interés. Debe ser categorizado como dummie. *Se elimina* 
- **CustLocation**: Localización. Puede ser de interés pero su codificación dado los 860046 datos únicos representan un consumo de RAM no costeable. *Se elimina*
- **CustAccountBalance**: Balance de la cuenta al momento de la transaccion. *Se utiliza*
- **TransacionDate**: Fecha de la transaccion.*Se utiliza pero debe ser procesado o casteado a un numero procesable por el modelo.* 
- **TransactionTime**: Tiempo que tomo la transacion (timestamp). *Se utiliza*
- **TransactionAmount**: Cantidad transaccionada. *Se utiliza*
#### Balanceo
Considerando que es un modelo no supervizado, no existe variable que se desee balancear. 
### Escalamiento de variables
Dada la independencia de las dimensiones de las variables (solo se hace la construccion de clusters), no es necesario el escalamiento de variables. 
### Metrica de desempeño
**WSS**
(Within-Cluster-Sum of Squared). Suma de las distancias euclidiandas entre una observacion y su centroide. En este caso se usa para evaluar un cluster con k centroides. Se utiliza también como parte de la elección una k, aplicable para el método del codo. 

## Task 1.1
- Split de datos: Se omite porque no se puede testear el no supervizado. Como mucho se puede hacer prueba de algun punto para ver a que cluster pertenece. De momento no interesa eso. **Se usan todos los datos para el fit**.
- Investigacion de PCA en directorio _docs_.
- Metrica de desempeño, codo con wss.

