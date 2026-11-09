# Guion de la sustentación final

**Alerta temprana de deserción** · Machine Learning II · Miguel Camargo y Camilo Hernández

Duración prevista: **8 min 40 s** (límite: 10 min). Diapositivas: [`diapositivas.pdf`](diapositivas.pdf) ·
app: <https://desercion-tec.streamlit.app/>

| # | Diapositiva | Quién | Tiempo |
|---|---|---|---|
| 1 | Portada | Miguel | 15 s |
| 2 | Del artículo a nuestra pregunta | Miguel | 45 s |
| 3 | Estrategia de datos | Miguel | 50 s |
| 4 | Un flujo sin fuga de información | Miguel | 50 s |
| 5 | El boosting empata; elegimos LightGBM | Miguel | 50 s |
| 6 | Tres preguntas incómodas, respondidas con datos | Camilo | 45 s |
| 7 | El 20 % de mayor riesgo concentra la mitad de los desertores | Camilo | 55 s |
| 8 | Qué mira el modelo | Camilo | 40 s |
| 9 | Los errores no cuestan lo mismo | Camilo | 45 s |
| 10 | La aplicación, en vivo (demo) | Miguel | 70 s |
| 11 | Lo que no resuelve y lo que sigue | Camilo | 40 s |
| 12 | Cierre y preguntas | Camilo | 15 s |
| | **Total** | Miguel 4:40 · Camilo 4:00 | **8:40** |

**Antes de empezar:** abrir la app en otra pestaña unos minutos antes. Streamlit Cloud duerme
las apps sin uso y tarda en despertar.

---

## 1. Portada — Miguel (≈15 s)

Buenos días. Somos Miguel Camargo y Camilo Hernández. Nuestro proyecto responde una pregunta
concreta: al terminar el primer semestre, ¿podemos saber qué estudiantes no van a seguir al
segundo año, a tiempo para acompañarlos? Lo trabajamos con datos reales del Tecnológico de
Monterrey, y al final les mostramos la aplicación funcionando.

## 2. Del artículo a nuestra pregunta — Miguel (≈45 s)

Partimos del artículo de Villar y Andrade, que predice la deserción en una universidad de
Portugal combinando lo que se sabe del estudiante al ingresar con sus notas de primer y segundo
semestre. Comparan siete algoritmos y gana el boosting, con un F1 de hasta 0,88.

Nosotros adaptamos ese método al Tecnológico de Monterrey, con 24.251 estudiantes. Hicimos un
cambio importante: predecimos un semestre antes, solo con las notas del primero, para que la
universidad tenga más tiempo de actuar. Comparamos nueve modelos, cuidando que no se filtre
información, y terminamos con el modelo desplegado en una aplicación.

## 3. Estrategia de datos — Miguel (≈50 s)

El dataset del Tec tiene más de 143 mil registros, pero las notas del primer semestre solo
existen para las cohortes 2019 y 2020. Por eso el estudio se concentra en 24.251 estudiantes de
pregrado de esas dos cohortes.

La variable objetivo vale uno si el estudiante ya no está matriculado al cerrar su primer año, y
solo pasa en el 7,7 % de los casos. Ese desbalance marca todo lo que sigue.

Antes de modelar separamos un 20 % como prueba sellada, agrupado por estudiante, y no lo tocamos
hasta el final.

También dejamos variables por fuera: las socioeconómicas, porque falta casi la mitad y no
aportan; las que delatan el desenlace, porque serían trampa; y a quienes ya se habían ido a mitad
de semestre.

## 4. Un flujo sin fuga de información — Miguel (≈50 s)

Todo el preprocesamiento vive dentro de un único Pipeline, que se ajusta solo con los datos de
entrenamiento de cada pliegue: imputar, escalar y codificar nunca ven los datos de validación.
SMOTE no lo impusimos: lo pusimos como una opción más en la búsqueda, y la validación lo rechazó
en todas las decisiones.

Usamos validación cruzada anidada, 5 por 3: la interna elige hiperparámetros y el umbral, la
externa mide.

Comparamos con PR-AUC, porque con 7,7 % de desertores la exactitud engaña, y fijamos el umbral
con F2, que pesa el doble el recall.

Y un hallazgo crítico: el artículo aplica SMOTE antes de separar los datos. Si hacemos lo mismo,
nuestro F1 salta de 0,12 a 0,95. Parte del desempeño publicado viene del protocolo, no del modelo.

## 5. El boosting empata; elegimos LightGBM — Miguel (≈50 s)

Comparamos nueve modelos: dos líneas base, la regresión logística del curso anterior, Random
Forest, SVM y cuatro de boosting. Los cuatro de boosting quedaron arriba y empatados.

Para elegir entre ellos miramos cuatro frentes. En desempeño, LightGBM no se distingue de los
demás de boosting, con la prueba t corregida de Nadeau y Bengio, pero sí supera a la regresión
logística. Es igual de estable entre rondas. Cuesta mucho menos: medio segundo por entrenamiento
frente a 16 de CatBoost. Y como es un modelo de árboles, permite calcular SHAP exacto para cada
estudiante. Con un desempeño empatado, el costo y la interpretabilidad deciden.

*Relevo:* Le paso la palabra a Camilo para los resultados.

## 6. Tres preguntas incómodas — Camilo (≈45 s)

Antes de abrir el conjunto de prueba nos hicimos tres preguntas incómodas sobre nuestro propio
diseño.

Primera: la rúbrica de admisión solo falta en la cohorte 2019, así que su marca de “sin dato”
podía estar delatando la cohorte. La probamos y no cambia nada; igual la quitamos, porque en una
cohorte nueva la rúbrica siempre está.

Segunda: ¿funciona en una cohorte que no vio? Entrenando con 2019 y probando en 2020, el PR-AUC
baja de 0,22 a 0,19. Lo reportamos como limitación.

Tercera: las probabilidades están bien calibradas, así que la aplicación las puede mostrar tal
cual.

Todo esto con datos de desarrollo: la prueba siguió sellada.

## 7. El 20 % de mayor riesgo concentra la mitad de los desertores — Camilo (≈55 s)

Con el modelo congelado, abrimos la prueba una sola vez: 4.843 estudiantes que nunca se usaron
para decidir nada.

El resultado más útil está en la curva: si la universidad alerta al azar al 20 % de sus
estudiantes, encuentra al 20 % de los desertores. Siguiendo el orden del modelo encuentra casi a
la mitad, el 48,5 %.

Con eso proponemos tres niveles. Alto, el 20 % de mayor riesgo: ahí deserta el 18 %, cuatro veces
más que en el nivel bajo, y la acción es contactarlos esa semana. Medio: seguimiento en el mes.
Bajo: acompañamiento habitual.

Y el test confirma lo que anticipó la validación: un PR-AUC de 0,22 frente a 0,247, dentro del
intervalo de confianza, con las probabilidades bien calibradas.

## 8. Qué mira el modelo — Camilo (≈40 s)

¿Qué mira el modelo? Con SHAP medimos cuánto pesa cada variable en las predicciones de la prueba.
Dos factores dominan, con el doble de peso que el tercero: el promedio del primer semestre, donde
a mejor promedio menos riesgo, y el porcentaje de beca, donde a más beca menos riesgo. La edad va
en sentido contrario: los mayores tienen más riesgo.

En palabras de la universidad: el riesgo se concentra en quien tuvo un mal arranque académico y
no tiene apoyo económico.

Ojo: son asociaciones, no causas. SHAP no dice que dar más beca evite la deserción; las becas
exigen promedio, así que también reflejan quién entra con buen desempeño.

## 9. Los errores no cuestan lo mismo — Camilo (≈45 s)

De los 367 desertores de la prueba, el modelo alertó a 235 y se le escaparon 132. A cambio emitió
falsas alarmas: unas 7,5 alertas por cada desertor encontrado.

Estos dos errores no cuestan lo mismo. Una falsa alarma cuesta una conversación de apoyo con
alguien que no la necesitaba. Un desertor no detectado es un estudiante que la universidad pierde
sin haber intentado nada. Por eso elegimos el umbral con F2. Y si los recursos son pocos, atender
solo el nivel alto baja la carga a 5,5 alertas por desertor.

Lo más importante: dónde falla. De los 23 becarios por talento académico que desertan, solo 6
reciben alerta, porque su beca y su promedio los hacen ver seguros. Conclusión: el modelo apoya a
los consejeros, no los reemplaza.

*Relevo:* Le paso la palabra a Miguel para la demostración.

## 10. La aplicación, en vivo — Miguel (≈70 s, demo)

1. **Pestaña “Panel del grupo”** (abre sola, con 40 estudiantes ficticios): arriba, cuántos hay
   en riesgo alto y con alerta; luego la composición por nivel, la distribución del riesgo y la
   lista ordenada para los consejeros. A la derecha, la ficha del estudiante de mayor riesgo:
   13 %, riesgo alto, y por qué. Se puede subir un archivo propio, CSV o Excel, y reemplaza al
   ejemplo.
2. **Pestaña “Un estudiante”**: con los valores típicos sale riesgo bajo, alrededor de 4 %. Bajo
   el promedio a 70, pongo 2 materias reprobadas, beca en 0 y marco “sin dato” en el examen
   presencial, y pulso *Estimar riesgo*: pasa a riesgo alto, cerca de 14 %, mayor que el 90 % de
   los estudiantes.

Lo importante técnicamente: la app no reimplementa nada. Carga el mismo Pipeline que evaluamos en
prueba, con un contrato que rechaza datos fuera de rango, y una predicción de referencia que da
exactamente lo mismo en mi computador, en Docker y en la nube.

*Si falla internet:* explicar sobre la captura de la diapositiva.

*Relevo:* Camilo cierra con las limitaciones.

## 11. Lo que no resuelve y lo que sigue — Camilo (≈40 s)

Somos honestos con lo que el modelo no resuelve. Solo hay dos cohortes con notas del primer
semestre, y la pandemia tocó a ambas. En una cohorte nueva conviene esperar algo menos de
desempeño: un PR-AUC cercano a 0,19. La señal es moderada, así que el modelo sirve para
priorizar, no para decidir sobre un estudiante. Y los becarios que desertan casi no se ven.

Lo que sigue: en MLOps, tratar cada cohorte como un ciclo, vigilando la calibración,
reentrenando con la cohorte más reciente y versionando el contrato, que ya existe. En datos,
sumar señales como la asistencia. Y en Deep Learning, el aporte no sería cambiar el boosting con
estos mismos datos, sino usar las notas semestre a semestre con modelos secuenciales que
aprendan la trayectoria.

## 12. Cierre — Camilo (≈15 s)

En resumen: elegimos LightGBM porque empata con los mejores y es el más barato de mantener; en
datos que nunca vio, el 20 % de mayor riesgo concentra casi la mitad de los desertores; y el
modelo está desplegado, explicado y es reproducible. La idea de fondo es ordenar a tiempo para
acompañar a tiempo. Muchas gracias; quedamos atentos a sus preguntas.
