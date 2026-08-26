import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

quantidade_logins = 10000
np.random.seed(42)

#Criando datas sintéticas
data_inicio = pd.Timestamp('2026-01-01 00:00:00')
data_fim = pd.Timestamp('2026-12-31 23:59:59')
intervalo_tempo = data_fim - data_inicio
segundos_intervalo = intervalo_tempo.total_seconds()

segundos_aleatorios = np.random.randint(
    0,
    int(segundos_intervalo),
    size=quantidade_logins
)

timestamps = data_inicio + pd.to_timedelta(segundos_aleatorios, unit='s')
timestamps = timestamps.sort_values()

df = pd.DataFrame({'timestamp_login': timestamps})

#Criando tentativas falhas 
tentativas_falhas = np.random.choice(
    [0, 1, 2, 3, 4, 5, 6, 7], 
    size=quantidade_logins, 
    p=[0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02]
    )

df['tentativas_falhas'] = tentativas_falhas

#print(df['tentativas_falhas'].value_counts(normalize=True).sort_index())

#Criando novo dispositivo 
novo_dispositivo = np.random.binomial(
    n=1,
    p=0.1,
    size=quantidade_logins
)

df['novo_dispositivo'] = novo_dispositivo

#Criando novo IP
# Dispositivos novos tendem a usar IPs novos,
# mas introduzimos variação entre os usuários.
probabilidade_novo_ip = np.where(
    df['novo_dispositivo'] == 1,
    np.random.choice([0.2, 0.8], size=quantidade_logins, p=[0.1, 0.9]),
    np.random.choice([0.2, 0.8], size=quantidade_logins, p=[0.9, 0.1])
    )

novo_ip = np.random.binomial(n=1, p=probabilidade_novo_ip)
df['novo_ip'] = novo_ip

#Criando VPN/Proxy
probabilidade_vpn = np.where(
    df['novo_ip'] == 1,
    0.3,
    0.1
)

vpn_proxy = np.random.binomial(n=1, p=probabilidade_vpn)
df['vpn_proxy'] = vpn_proxy

#Criando país diferente
condicoes = [
    (df['novo_ip'] == 0) & (df['vpn_proxy'] == 0),
    (df['novo_ip'] == 0) & (df['vpn_proxy'] == 1),
    (df['novo_ip'] == 1) & (df['vpn_proxy'] == 0),
    (df['novo_ip'] == 1) & (df['vpn_proxy'] == 1)
]
probabilidades = [0.02, 0.15, 0.20, 0.40]

probabilidade_pais_diferente = np.select(
    condicoes,
    probabilidades
)

pais_diferente = np.random.binomial(
    n=1,
    p=probabilidade_pais_diferente
)
df['pais_diferente'] = pais_diferente

#Criando distancia_km
escala_distancia = np.where(
    df['pais_diferente'] == 1,
    1500,
    100
)

distancia_km = np.random.exponential(scale= escala_distancia)
distancia_km = np.round(distancia_km, 2)

df['distancia_km'] = distancia_km

#Criando ID do usuário
quantidade_usuarios = 250

usuarios = np.random.randint(1, quantidade_usuarios + 1, size=quantidade_logins)
df['usuario_id'] = usuarios

#Criando tempo desde ultimo login e primeiro login (login anterior é apenas para suporte)
login_anterior = (
    df.groupby('usuario_id')['timestamp_login'].shift(1)
)
df['login_anterior'] = login_anterior

tempo = df['timestamp_login'] - df['login_anterior']

df['tempo_desde_ultimo_login'] = np.round(tempo.dt.total_seconds() / 60, 2)

df['primeiro_login'] = df['login_anterior'].isna().astype(int)

mediana_tempo = df['tempo_desde_ultimo_login'].median()
df['tempo_desde_ultimo_login'] = (df['tempo_desde_ultimo_login'].fillna(mediana_tempo))

#Criando informações de horario com timestamp
df['hora_login'] = df['timestamp_login'].dt.hour

df['dia_semana'] = df['timestamp_login'].dt.day_of_week

df['fim_de_semana'] = (df['dia_semana'] >= 5).astype(int)

df['login_madrugada'] = ((df['hora_login'] >= 0) & (df['hora_login'] <= 5)).astype(int)

#Criando o protagonista login suspeito
probabilidade_suspeito = np.full(
    quantidade_logins,
    0.01
)
aumento_tentativas = df['tentativas_falhas'] * 0.015
probabilidade_suspeito += aumento_tentativas

aumento_novo_dispositivo = df['novo_dispositivo'] * 0.05
probabilidade_suspeito += aumento_novo_dispositivo

aumento_novo_ip = df['novo_ip'] * 0.04
probabilidade_suspeito += aumento_novo_ip

aumento_vpn = df['vpn_proxy'] * 0.03
probabilidade_suspeito += aumento_vpn

aumento_pais_diferente = df['pais_diferente'] * 0.08
probabilidade_suspeito += aumento_pais_diferente

df['distancia_log'] = np.log1p(df['distancia_km'])
limite_distancia = np.log1p(100)
excesso_distancia = np.clip(df['distancia_log'] - limite_distancia, 0, None)
aumento_distancia = excesso_distancia * 0.015
probabilidade_suspeito += aumento_distancia

tempo_horas = df['tempo_desde_ultimo_login'] / 60
df['login_recente'] = ((df['tempo_desde_ultimo_login'] <= 360) & (df['primeiro_login'] == 0)).astype(int)
