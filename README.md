# 🏋️ GymBot - Analizador Inteligente de Entrenamientos de Gimnasio

Un sistema completo, modular, robusto y **100% gratuito ($0)** diseñado para registrar, sincronizar y analizar entrenamientos de gimnasio a partir de un chat privado de Telegram, utilizando **MTProto en solo lectura pasiva**, persistencia local en **SQLite (WAL)** y el motor de IA **Google Gemini 2.5 Flash** (`google-genai`).

---

## 🎯 Características Principales

1. **Lectura Pasiva 100% Limpia (Zero-Noise)**:
   - Accede a tu canal o chat privado donde registras tus series mediante **Telethon (MTProto)** con tus credenciales de usuario.
   - **Garantía absoluta de solo lectura**: GymBot nunca escribe, no responde, no envía reacciones ni ensucia tu chat de notas.
2. **Doble Interfaz de Consulta**:
   - **Interfaz PC (Terminal CLI)**: Terminal interactiva con formato visual (`rich`), comandos especiales (`:sync`, `:stats`, `:clear`, `:help`, `:exit`) y soporte de argumentos directos (`python cli.py "¿Cuál fue mi mejor banca del mes?"`).
   - **Interfaz Móvil (Bot Secundario)**: Bot independiente creado con BotFather para responder tus consultas directamente al celular con **autenticación estricta por whitelist** (`telegram_user_id`).
3. **Motor de IA Gratuito (Gemini 2.5 Flash)**:
   - Integrado con el SDK oficial `google-genai`.
   - Ventana de contexto de 1 millón de tokens bajo el Free Tier (15 RPM, 1.500 peticiones diarias gratuitas).
   - Prompt de sistema especializado en fisiología del ejercicio, sobrecarga progresiva, estimación teórica de 1RM (Epley/Brzycki) y análisis de fatiga.
4. **Persistencia y Sincronización Eficiente**:
   - Base de datos SQLite local configurada en modo **WAL (Write-Ahead Logging)** para lectura concurrente libre de bloqueos.
   - Sincronización incremental mediante cursor `min_id` para consultar únicamente mensajes nuevos.
   - Conexión MTProto efímera que evita bloqueos de la base de datos de sesión entre la CLI y el servicio en segundo plano.
5. **Listo para Producción en Linux**:
   - Configuración vía variables de entorno (`.env`).
   - Archivo de servicio `gymbot.service` para `systemd`.

---

## 🏗️ Arquitectura del Sistema

```
[Chat Privado de Registros] (Telegram)
           │
           │ (Telethon MTProto - Solo Lectura Pasiva)
           ▼
 [telegram_service/mtproto_reader.py] (Conexión Efímera Incremental min_id)
           │
           ▼
   [storage/database.py] (SQLite en modo WAL: gymbot.db)
           │
           ├──► [ai/gemini_client.py + ai/analyzer.py] (Gemini 2.5 Flash)
           │                     │
           │                     ├──► [cli.py] (Terminal PC Interactiva / Argumentos)
           │                     │
           │                     └──► [telegram_service/bot_service.py] (Bot Móvil con Whitelist)
           │
      [service.py] (Daemon background para systemd: Bot Móvil + Sync Periódico)
```

---

## 📁 Estructura del Proyecto

```
gymbot/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuración tipada con Pydantic y validación de .env
├── storage/
│   ├── __init__.py
│   ├── database.py          # Gestor SQLite en modo WAL, migraciones y consultas
│   └── models.py            # Dataclasses (RawMessage, SyncMetadata, DatabaseStats)
├── telegram_service/
│   ├── __init__.py
│   ├── mtproto_reader.py    # Lector MTProto (Telethon) estricto de solo lectura
│   └── bot_service.py       # Bot de Telegram para consultas móviles con whitelist
├── ai/
│   ├── __init__.py
│   ├── gemini_client.py     # Cliente oficial google-genai con reintentos para Free Tier
│   └── analyzer.py          # Constructor de contexto y directivas de entrenador de fuerza
├── cli.py                   # CLI para terminal de PC (modo interactivo y argumentos)
├── service.py               # Servicio en segundo plano para el bot móvil
├── auth_telethon.py         # Asistente interactivo guiado para el login inicial de Telethon
├── gymbot.service           # Archivo de unidad systemd para Linux
├── requirements.txt         # Dependencias del proyecto
├── .env.example             # Plantilla de variables de entorno
└── README.md                # Documentación del sistema
```

---

## 🚀 Guía de Instalación y Configuración

### 1. Requisitos Previos
- Python 3.10 o superior (compatible y probado en Python 3.14).
- Cuenta de Telegram personal.
- Clave de API de Google Gemini (gratuita).

### 2. Clonar y Crear el Entorno Virtual

```bash
git clone <tu-repositorio> gymbot
cd gymbot

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Obtención de Credenciales

1. **Credenciales de Telegram MTProto (para leer tus registros)**:
   - Dirígete a [my.telegram.org](https://my.telegram.org) e inicia sesión con tu número de teléfono.
   - Entra en **API development tools**.
   - Crea una aplicación (ej. nombre: `GymBotReader`) y copia tu `api_id` y `api_hash`.
2. **Chat de Entrenamientos**:
   - Crea un canal o grupo privado en Telegram donde solo tú escribas tus entrenamientos.
   - Obtén el ID del chat (generalmente un número negativo como `-1001234567890`) o usa su `@username` si es público/canal privado con enlace de invitación.
3. **Bot Secundario para Consultas Móviles**:
   - Abre Telegram y busca al bot oficial [@BotFather](https://t.me/BotFather).
   - Envía `/newbot`, dale un nombre (ej. `MiGymBot`) y un username (ej. `mi_gym_tracker_bot`).
   - BotFather te entregará un `token` HTTP API.
4. **API Key de Google Gemini (Free Tier)**:
   - Entra en [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Genera tu clave de API gratuita.

### 4. Configurar el Archivo `.env`

Copia la plantilla y edítala con tus claves:

```bash
cp .env.example .env
nano .env  # o usa tu editor preferido
```

Ejemplo de contenido:
```env
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef0123456789abcdef0123456789
TELEGRAM_SESSION_NAME=gymbot_user
TELEGRAM_WORKOUT_CHAT_ID=-1001234567890

TELEGRAM_BOT_TOKEN=1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ_12345678
ALLOWED_TELEGRAM_USER_IDS=123456789

GEMINI_API_KEY=AIzaSyD-TuClaveGeminiAqui1234567890
GEMINI_MODEL=gemini-2.5-flash

DATABASE_PATH=gymbot.db
AUTO_SYNC_INTERVAL_MINUTES=60
```

---

## 🔑 Autenticación Inicial de MTProto (Paso Único)

Telethon requiere iniciar sesión una sola vez para generar el archivo `gymbot_user.session`. Ejecuta el asistente guiado:

```bash
python auth_telethon.py
```

El script te solicitará:
1. Tu número de teléfono con prefijo internacional (ej. `+54911xxxxxxxx`).
2. El código de verificación de 5 dígitos que Telegram enviará a tus dispositivos.
3. Tu contraseña de doble factor (2FA/Cloud Password), si la tienes activada.

Al finalizar, el asistente mostrará tu **Telegram User ID**. Asegúrate de colocar este ID en `ALLOWED_TELEGRAM_USER_IDS` dentro de tu `.env` para que el bot móvil reconozca tu usuario.

---

## 💻 Uso de la Interfaz PC (CLI)

### Modo Interactivo Continuo
Simplemente ejecuta:
```bash
python cli.py
```
Se abrirá la consola interactiva con un prompt personalizado:
```
gymbot > :sync
✔ Sincronización exitosa: 12 nuevos mensajes guardados. Total en base de datos: 142.

gymbot > ¿Cuál fue mi mejor press de banca este mes?
[Analizando con Gemini 2.5 Flash...]

gymbot > :stats
[Muestra tabla con fechas y conteos]

gymbot > :exit
```

### Modo de Argumentos Directos
Puedes hacer preguntas puntuales directamente desde tu shell o scripts:
```bash
python cli.py "¿Qué peso hice en sentadillas la última sesión?"
python cli.py "Calcula mi 1RM en peso muerto según mis últimos registros"
python cli.py :sync
python cli.py :stats
```

---

## 📱 Uso de la Interfaz Móvil (Bot Secundario)

Para iniciar el bot móvil en tu terminal o servidor:
```bash
python service.py
```

Abre Telegram en tu celular y busca tu bot secundario:
1. Envía `/start` para verificar la conexión y ver el estado de la base de datos.
2. Envía `/sync` para forzar la sincronización de tus últimos entrenamientos registrados.
3. Envía `/stats` para consultar totales y fechas registradas.
4. Escribe cualquier pregunta en lenguaje natural o **envía un mensaje de voz/audio**:
   - Pulsa el micrófono en Telegram y haz tu pregunta hablando (ej. *"¿Cómo viene mi progresión de press de banca?"*).
   - El bot transcribirá el audio con Gemini a texto, te mostrará la transcripción y responderá con el análisis técnico de tus datos.
   - *"¿Qué ejercicios hice el martes pasado?"*
   - *"Revisa si tengo anotadas molestias en hombros o rodillas."*

> 🔒 **Seguridad**: Si cualquier otro usuario intenta enviar un mensaje a tu bot, este será completamente ignorado y se registrará un aviso en los logs.

---

## 🐧 Despliegue como Servicio systemd en Linux (Modo Usuario - Recomendado)

Para que el bot móvil y la sincronización automática funcionen permanentemente en segundo plano sin requerir privilegios de `root` ni generar conflictos de permisos en tu home (ideal para Fedora, Ubuntu, Debian, etc.):

1. **Copiar el servicio al directorio de usuario**:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp gymbot.service ~/.config/systemd/user/gymbot.service
   ```
2. **Habilitar el servicio persistente (que siga corriendo incluso al cerrar sesión)**:
   ```bash
   loginctl enable-linger $USER
   ```
3. **Recargar e iniciar el servicio**:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now gymbot.service
   ```
4. **Verificar el estado y los logs en tiempo real**:
   ```bash
   systemctl --user status gymbot.service
   journalctl --user -u gymbot.service -f
   ```

---

## 🧪 Pruebas Unitarias y Calidad de Código

Para verificar el linter y formateador (Ruff):
```bash
ruff check .
ruff format --check .
```

Para ejecutar la suite de pruebas automatizadas:
```bash
pytest
# o con el intérprete del entorno virtual:
.venv/bin/python -m unittest discover tests
```

---

## 💡 Consejos para tus Notas de Entrenamiento
GymBot está optimizado para entender anotaciones humanas en lenguaje natural:
- `Banca: 80kg x 8, 85kg x 6, 85kg x 5 @8`
- `Sentadillas 100 4x6 (sentí molestia leve en rodilla izq en la 3ra serie)`
- `Dominadas: lastre 15kg 3x5 + fallo sin lastre`
- `Elevaciones laterales 12kg fallo, dropset a 8kg`

¡Gemini 2.5 Flash interpretará automáticamente las cargas, RPE, series de aproximación vs efectivas y tus sensaciones personales!
