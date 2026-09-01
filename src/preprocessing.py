import pandas as pd
import numpy as np

#Preparando para receber dados mais simples e transforma-los no que o LogEye usa, feito para melhorar a experiencia do usuario

def preparar_login(dados_login):
    timestamp = pd.to_datetime(dados_login['timestamp_login'])
    hora = timestamp.hour
    login_madrugada = int(0 <= hora <= 5)

    distancia_log = np.log1p(
    dados_login['distancia_km']
    )
    tempo_log = np.log1p(
    dados_login['tempo_desde_ultimo_login']
    )

    dados_processados = {
    'tentativas_falhas': dados_login['tentativas_falhas'],
    'novo_dispositivo': dados_login['novo_dispositivo'],
    'novo_ip': dados_login['novo_ip'],
    'vpn_proxy': dados_login['vpn_proxy'],
    'pais_diferente': dados_login['pais_diferente'],
    'distancia_log': distancia_log,
    'tempo_log': tempo_log,
    'primeiro_login': dados_login['primeiro_login'],
    'login_madrugada': login_madrugada,
    }
    login_processado = pd.DataFrame([dados_processados])
    return login_processado

