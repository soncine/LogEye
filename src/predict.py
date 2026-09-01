import json
import joblib
from pathlib import Path
from .preprocessing import preparar_login

BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = BASE_DIR / 'models' / 'logeye_v1.joblib'
CAMINHO_CONFIG = BASE_DIR / 'models' / 'logeye_v1_config.json'

modelo = joblib.load(CAMINHO_MODELO)
with open(CAMINHO_CONFIG, 'r') as arquivo:
    config = json.load(arquivo)
threshold = config['threshold']

def analisar_login(dados_login):
    login = preparar_login(dados_login)
    score = modelo.predict_proba(login)[0, 1]
    suspeito = int(score >= threshold)
    return {
        'versao': config['versao'],
        'score': float(score),
        'suspeito': suspeito
    }

