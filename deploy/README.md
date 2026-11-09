# Despliegue de la app de riesgo de deserción

App Streamlit que recibe los datos de un estudiante (o un archivo con varios), aplica el mismo
Pipeline del entrenamiento y devuelve la probabilidad de deserción, la decisión con el umbral
validado y su explicación SHAP. Sigue la estructura del laboratorio de la semana 14:
**artefactos con contrato → imagen Docker → Compose con Caddy → VM de Google Cloud con HTTPS**.

```
deploy/
├── .streamlit/config.toml   # tema claro, paleta y fuente del tablero
├── app/main.py              # interfaz (un estudiante, carga de archivo, sobre el modelo)
├── app/model_service.py     # cargar, validar contra el contrato, predecir, niveles y SHAP
├── app/theme.py · charts.py # estilo de vidrio pastel y gráficos Altair
├── artifacts/               # generados por `desercion final` (no se editan a mano)
│   ├── model.joblib             # Pipeline completo: imputación, escalado, one-hot y LightGBM
│   ├── model_metadata.json      # contrato: orden y rangos de variables, umbral, versión, métricas
│   └── reference_prediction.json# perfil FICTICIO y su probabilidad (prueba de paridad)
├── tests/                   # contrato, paridad, validación y carga masiva
├── Dockerfile  compose.yaml  Caddyfile  .env.example  .dockerignore
```

## Contrato

- Streamlit escucha solo dentro de Docker, en el puerto `8501`; nunca se publica.
- Caddy publica `80` y `443`, obtiene el certificado HTTPS y reenvía a `app:8501`.
- `.env` define `APP_DOMAIN`, no se versiona y nunca entra a la imagen.
- `model_metadata.json` fija el orden de variables, los rangos y categorías admitidos, el umbral
  (F2) y la versión. Una entrada fuera del contrato se rechaza con un mensaje claro.
- La imagen solo contiene `app/`, `artifacts/` y `requirements.txt`: ningún dato del TEC.
  No se usan llaves de cuentas de servicio.

## 1. Correr en local

```bash
# Sin Docker (desde la raíz del repositorio, con el entorno del proyecto)
pip install -r deploy/app/requirements.txt
pytest deploy/tests
cd deploy && streamlit run app/main.py          # http://localhost:8501

# Con Docker (desde deploy/)
cp .env.example .env                             # APP_DOMAIN=localhost
docker compose config -q && docker compose up -d --build
docker compose ps                                # app debe aparecer "healthy"
# https://localhost (en local Caddy usa un certificado propio; el navegador avisa)
docker compose down
```

Para regenerar los artefactos tras cambiar el modelo: `desercion final --force` en la raíz
(reproduce el mismo modelo con la semilla 42 y vuelve a escribir `artifacts/`).

## 2. Desplegar en Streamlit Community Cloud (opción usada en la entrega)

El repositorio ya trae lo que Community Cloud necesita cuando la app no está en la raíz:

| Archivo | Por qué |
|---|---|
| `deploy/app/requirements.txt` | Community Cloud busca las dependencias primero en la carpeta del archivo principal; así no instala el `pyproject.toml` de la raíz (todo el proyecto). Docker usa este mismo archivo. |
| `.streamlit/config.toml` (raíz) | Community Cloud solo lee la configuración desde la raíz. Es copia de `deploy/.streamlit/config.toml` (una prueba verifica que coincidan). |
| `packages.txt` (raíz) | `libgomp1`, biblioteca del sistema que necesita LightGBM. |

Pasos:

1. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub dueña del repositorio y autorizar el acceso a repositorios privados.
2. **Create app → Deploy a public app from GitHub** (aunque el repo sea privado):
   - Repository: `miguel-camargo611/ML2-desercion-estudiantil` · Branch: `main`
   - Main file path: `deploy/app/main.py`
   - App URL: un subdominio corto, p. ej. `desercion-tec`
3. **Advanced settings → Python 3.13** (la versión con la que se serializó el modelo). No hay secretos.
4. **Deploy** y esperar la instalación (unos minutos; el registro se ve en *Manage app*).
5. **Compartir:** como el repositorio es privado, la app nace privada. Desde **Share**, invitar
   el correo del profesor o, si la opción está disponible, hacerla pública: la app no contiene
   datos de estudiantes, solo el modelo y su contrato.

Cada `push` a `main` redespliega la app automáticamente. Simulación local equivalente (entorno
limpio, lanzado desde la raíz como lo hace Community Cloud):

```bash
python -m venv .cloudenv && .cloudenv/Scripts/pip install -r deploy/app/requirements.txt
.cloudenv/Scripts/streamlit run deploy/app/main.py
```

## 3. Desplegar en Google Cloud (alternativa vista en clase)

Pasos de la guía de clase adaptados a este proyecto. Todo se hace desde el navegador
(Cloud Console y el botón **SSH** de la VM); no hace falta instalar `gcloud`.

| # | Paso | Detalle |
|---|---|---|
| 1 | Proyecto | Cloud Console → *Proyecto nuevo* `ml2-desercion-equipo`. Anotar el *Project ID*. |
| 2 | Facturación y presupuesto | Vincular la facturación. *Billing → Budgets & alerts*: USD 5, alertas al 50/90/100 %. **La alerta avisa, no frena el gasto.** |
| 3 | Cuenta de servicio | *IAM → Service Accounts*: `sa-streamlit-runtime`, **sin roles y sin llave**. |
| 4 | Firewall | *VPC → Firewall*, con el tag `streamlit-web`: `allow-streamlit-web` (0.0.0.0/0, TCP 80 y 443) y `allow-ssh-from-iap` (35.235.240.0/20, TCP 22). **Ninguna regla para 8501.** |
| 5 | VM | `vm-desercion`, `us-central1`, `e2-small`, Ubuntu 24.04 LTS, disco de 20 GB, tag `streamlit-web`, cuenta `sa-streamlit-runtime`, IP externa efímera. Anotar la IP. |
| 6 | Docker en la VM | Botón **SSH** y los comandos oficiales de la guía de clase (repositorio apt de Docker, `docker-compose-plugin`, `usermod -aG docker`). Validar con `docker run --rm hello-world`. |
| 7 | Subir la app | En el computador: comprimir la carpeta `deploy/` (sin `.env`) como `deploy.zip`. En la ventana SSH: ⚙ → *Upload file*. Luego `sudo apt install -y unzip && unzip deploy.zip && cd deploy`. |
| 8 | Dominio | Con IP `34.120.10.20`: `cp .env.example .env` y `sed -i 's/^APP_DOMAIN=.*/APP_DOMAIN=34-120-10-20.sslip.io/' .env` |
| 9 | Desplegar | `docker compose config -q && docker compose up -d --build`, luego `docker compose ps` y `docker compose logs --tail=80 app caddy`. Abrir `https://34-120-10-20.sslip.io`. |
| 10 | Limpieza (después de la sustentación) | Borrar la VM y su disco, verificar que no quede IP reservada, borrar las dos reglas de firewall y revisar *Billing → Reports*. |

Costo de referencia: una `e2-small` cuesta del orden de USD 0,50 al día mientras existe.

## 4. Pruebas de aceptación (Docker)

| ID | Prueba | Criterio | Local (Docker Desktop) | VM |
|---|---|---|---|---|
| A1 | `docker compose config` | Sin error ni `TODO` | ✅ | ☐ |
| A2 | Construcción limpia | `docker compose build --no-cache app` termina | ✅ (`docker build`) | ☐ |
| A3 | Salud | `app` aparece `healthy` | ✅ | ☐ |
| A4 | HTTPS | Certificado válido, sin `-k` | — (local usa CA propia) | ☐ |
| A5 | Redirección | HTTP redirige a HTTPS | ✅ 308 | ☐ |
| A6 | Inferencia | La referencia conserva clase y probabilidad (±1e-6) | ✅ p = 0,040091 | ☐ |
| A7 | Explicación | SHAP carga; se lee como asociación, no causalidad | ✅ | ☐ |
| A8 | Superficie | 8501 no accesible desde fuera | ✅ solo `expose` | ☐ |
| A9 | Reinicio | Tras recrear, vuelve a `healthy` | ✅ | ☐ |
| A10 | Limpieza | Sin VM, disco ni IP al cerrar | — | ☐ |

Comandos de verificación en la VM:

```bash
curl -I http://34-120-10-20.sslip.io          # A5: 308 hacia https
curl -I https://34-120-10-20.sslip.io         # A4: 200 sin -k
docker compose exec -T app python -c "import json,sys; sys.path.insert(0,'app'); \
from model_service import load_bundle,predict; m,md=load_bundle('artifacts'); \
r=json.load(open('artifacts/reference_prediction.json')); print(predict(m,md,r['input']), r['probability'])"   # A6
docker compose down && docker compose up -d && docker compose ps                # A9
```

## 5. Coherencia entre entrenamiento y uso

- **Mismas versiones:** `requirements.txt` fija las versiones con las que se entrenó el modelo
  (`pyproject.toml`), y la prueba de paridad lo verifica en cada entorno.
- **Mismo preprocesamiento:** la app no reimplementa nada; usa el Pipeline serializado. Solo
  calcula la variable construida (promedio del 1.er semestre − promedio de bachillerato) con la
  fórmula del entrenamiento, verificada por una prueba.
- **Mismos valores admitidos:** rangos y categorías salen del conjunto de desarrollo. Una
  categoría nueva o un valor fuera de rango no se predice: se reporta.
- **"Sin dato" solo donde es habitual:** el examen presencial (lo presentó en línea) y la
  rúbrica. La rúbrica se imputa sin marca de faltante porque esa marca solo distinguía la
  cohorte AD19 (`reports/tables/sensibilidad_pliegues.csv`).
