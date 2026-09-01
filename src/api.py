from fastapi import FastAPI
from pydantic import BaseModel
from .predict import analisar_login

app = FastAPI(
    title='LogEye API',
    version='1.0'
)

class LoginEntrada(BaseModel):
    timestamp_login: str
    tentativas_falhas: int
    novo_dispositivo: int
    novo_ip: int
    vpn_proxy: int
    pais_diferente: int
    distancia_km: float
    tempo_desde_ultimo_login: float
    primeiro_login: int


@app.get('/health')#Verificando se esta 'vivo'
def health():
    return {
        'status': 'ok',
        'versao': '1.0'
    }

@app.post('/analisar')
def analisar(login: LoginEntrada):
    dados_login = login.model_dump()
    resultado = analisar_login(dados_login)
    return resultado