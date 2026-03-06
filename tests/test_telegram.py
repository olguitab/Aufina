import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")

if not token or not chat_id:
    print("Faltan credenciales de Telegram en .env")
else:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(url, data={
        "chat_id": chat_id, 
        "text": "🤖 *Prueba de Sentinel AI*\n\nSi lees esto, las notificaciones están configuradas correctamente.", 
        "parse_mode": "Markdown"
    })
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
