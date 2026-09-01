# 👁️ LogEye — Detecção de Logins Suspeitos com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)

## 📑 Índice

- [📋 Visão Geral](#-visão-geral)
- [🎯 Objetivos](#-objetivos)
- [🧠 Como o LogEye Funciona](#-como-o-logeye-funciona)
- [📊 Dataset](#-dataset)
- [🔎 Análise Exploratória de Dados](#-análise-exploratória-de-dados)
- [⚙️ Features Utilizadas](#️-features-utilizadas)
- [🤖 Modelos Avaliados](#-modelos-avaliados)
- [📈 Resultados](#-resultados)
- [🎚️ Ajuste de Threshold](#️-ajuste-de-threshold)
- [🏗️ Arquitetura do Projeto](#️-arquitetura-do-projeto)
- [🌐 API REST com FastAPI](#-api-rest-com-fastapi)
- [🚀 Como Executar](#-como-executar)
- [📦 Dependências](#-dependências)
- [⚠️ Limitações](#️-limitações)
- [🔮 Próximas Evoluções](#-próximas-evoluções)
- [🎓 Aprendizados](#-aprendizados)
- [📌 Contexto do Projeto](#-contexto-do-projeto)

---

## 📋 Visão Geral

O **LogEye** é um projeto de Machine Learning voltado para a identificação de **tentativas de login potencialmente suspeitas**.

A aplicação analisa diferentes características relacionadas ao contexto de uma autenticação, como quantidade de tentativas falhas, dispositivo utilizado, endereço IP, uso de VPN ou proxy, localização geográfica, distância da localização habitual e horário do acesso.

A partir desses sinais, um modelo de Machine Learning gera um **score de risco** e classifica o login como normal ou suspeito utilizando um threshold definido durante a etapa de validação.

Além da análise de dados e treinamento do modelo, o projeto disponibiliza a inferência através de uma **API REST construída com FastAPI**, permitindo que outras aplicações enviem dados de login e recebam uma avaliação de risco.

> **Importante:** o LogEye é um projeto educacional desenvolvido sobre dados sintéticos. O resultado produzido pelo modelo representa um indicador de risco e não uma confirmação de fraude, invasão ou ataque.

---

## 🎯 Objetivos

O projeto foi desenvolvido com o objetivo de aplicar, em um único fluxo, conceitos de:

- geração e manipulação de dados;
- análise exploratória de dados;
- feature engineering;
- classificação supervisionada;
- tratamento de classes desbalanceadas;
- avaliação de modelos com métricas adequadas;
- ajuste de threshold;
- persistência de modelos de Machine Learning;
- construção de pipelines;
- desenvolvimento de uma API REST;
- organização de um projeto de dados próximo de uma aplicação real.

O LogEye também busca demonstrar por que métricas como **accuracy isoladamente podem ser enganosas** em problemas nos quais a classe de interesse representa apenas uma pequena parcela dos dados.

---

## 🧠 Como o LogEye Funciona

O fluxo principal da aplicação é:

```text
Dados de um login
        │
        ▼
Pré-processamento
        │
        ├── transformação logarítmica da distância
        ├── transformação logarítmica do tempo
        └── identificação de login na madrugada
        │
        ▼
Pipeline de Machine Learning
        │
        ├── StandardScaler
        └── Logistic Regression
        │
        ▼
Score de risco
        │
        ▼
Threshold configurado
        │
        ▼
Normal ou Suspeito
```

Na utilização através da API:

```text
Aplicação cliente
       │
       │ POST /analisar
       ▼
FastAPI
       │
       ▼
Validação com Pydantic
       │
       ▼
preprocessing.py
       │
       ▼
predict.py
       │
       ▼
logeye_v1.joblib
       │
       ▼
Score + classificação
       │
       ▼
Resposta JSON
```

---

## 📊 Dataset

O dataset utilizado pelo projeto foi **gerado sinteticamente em Python** para permitir o desenvolvimento completo do pipeline sem depender de dados reais ou sensíveis de autenticação.

A versão inicial contém:

- **10.000 tentativas de login**
- **250 usuários sintéticos**
- registros distribuídos ao longo de **2026**
- aproximadamente **7,2% de logins suspeitos**
- aproximadamente **92,8% de logins normais**

A distribuição propositalmente desbalanceada busca simular um cenário no qual eventos suspeitos são menos frequentes que eventos legítimos.

### Variáveis armazenadas no dataset

| Feature | Descrição |
| :--- | :--- |
| `timestamp_login` | Data e horário do login |
| `tentativas_falhas` | Quantidade recente de tentativas de login malsucedidas |
| `novo_dispositivo` | Indica se o dispositivo ainda não era conhecido |
| `novo_ip` | Indica se o IP ainda não era conhecido para o usuário |
| `vpn_proxy` | Indica utilização de VPN ou proxy |
| `pais_diferente` | Indica alteração do país habitual |
| `distancia_km` | Distância entre a localização habitual e a localização atual |
| `usuario_id` | Identificador sintético do usuário |
| `tempo_desde_ultimo_login` | Tempo em minutos desde o último login |
| `primeiro_login` | Indica se é o primeiro registro disponível do usuário |
| `login_suspeito` | Variável alvo utilizada no treinamento |

### Construção do target

O target foi criado de forma **probabilística**, e não através de uma única regra determinística.

Diferentes sinais aumentam a probabilidade de um registro ser classificado como suspeito, mas nenhuma feature individual determina obrigatoriamente o resultado.

Isso permite que existam, por exemplo:

- logins normais realizados em outro país;
- logins suspeitos sem utilização de VPN;
- dispositivos novos legítimos;
- múltiplos fatores de risco ocorrendo simultaneamente.

Essa característica torna o problema mais adequado para uma abordagem multivariada de Machine Learning.

---

## 🔎 Análise Exploratória de Dados

Antes do treinamento, foi realizada uma análise exploratória para compreender as distribuições e relações presentes no dataset.

Entre os principais resultados observados:

### Desbalanceamento do target

```text
Normal     → 92,8%
Suspeito   →  7,2%
```

Esse resultado mostrou desde o início que **accuracy não poderia ser utilizada como principal métrica de avaliação**.

Um modelo que classificasse todos os registros como normais já teria aproximadamente 92,8% de accuracy.

### Tentativas falhas

A proporção de logins suspeitos apresentou crescimento conforme aumentava o número de tentativas falhas.

Exemplos observados no dataset:

```text
0 tentativas → ~5,3% suspeitos
3 tentativas → ~11,3% suspeitos
7 tentativas → ~17,5% suspeitos
```

### Novo dispositivo

```text
Dispositivo conhecido → ~6,4% suspeitos
Novo dispositivo      → ~14,0% suspeitos
```

### Novo IP

```text
IP conhecido → ~4,7% suspeitos
Novo IP      → ~12,8% suspeitos
```

### VPN ou Proxy

```text
Sem VPN/Proxy → ~6,3% suspeitos
Com VPN/Proxy → ~12,1% suspeitos
```

### País diferente

```text
Mesmo país      → ~5,7% suspeitos
País diferente  → ~19,5% suspeitos
```

Foi uma das variáveis com maior associação individual com o target.

### Horário

Também foi analisado o horário do login.

Agrupando os acessos entre **00h e 05h**:

```text
Fora da madrugada → ~6,4% suspeitos
Madrugada         → ~9,4% suspeitos
```

### Matriz de correlação

Entre as maiores relações encontradas entre as próprias features estão aproximadamente:

```text
pais_diferente ↔ distancia_log      0.57
novo_ip ↔ pais_diferente            0.37
novo_dispositivo ↔ novo_ip          0.31
vpn_proxy ↔ pais_diferente          0.28
novo_ip ↔ vpn_proxy                 0.27
```

Nenhuma feature apresentou correlação extremamente alta com `login_suspeito`, reforçando a ideia de que a classificação depende da **combinação de múltiplos sinais**.

---

## ⚙️ Features Utilizadas

Após a análise exploratória e o processo de feature engineering, o modelo final recebe 9 features:

| Feature | Tipo | Descrição |
| :--- | :--- | :--- |
| `tentativas_falhas` | Numérica | Número de tentativas falhas |
| `novo_dispositivo` | Binária | Dispositivo novo |
| `novo_ip` | Binária | IP novo |
| `vpn_proxy` | Binária | Utilização de VPN ou proxy |
| `pais_diferente` | Binária | Login em país diferente |
| `distancia_log` | Numérica | `log1p(distancia_km)` |
| `tempo_log` | Numérica | `log1p(tempo_desde_ultimo_login)` |
| `primeiro_login` | Binária | Primeiro login registrado do usuário |
| `login_madrugada` | Binária | Login realizado entre 00h e 05h |

A coluna `usuario_id` é mantida como identificador no dataset, porém **não é utilizada como feature pelo modelo**.

O timestamp bruto também não é fornecido diretamente ao classificador. Informações temporais relevantes são derivadas durante o pré-processamento.

---

## 🤖 Modelos Avaliados

Durante o desenvolvimento foram comparadas diferentes estratégias.

### Logistic Regression — Baseline

Primeiro modelo utilizado como referência.

Pipeline:

```text
StandardScaler
      ↓
LogisticRegression
```

Com o threshold padrão de `0.5`, o modelo classificou todos os registros do conjunto de teste como normais.

Apesar disso:

```text
ROC-AUC          ≈ 0.754
Average Precision ≈ 0.205
```

Isso demonstrou que o modelo havia aprendido algum poder de ordenação dos riscos, mesmo que o threshold padrão fosse inadequado.

### Logistic Regression com `class_weight="balanced"`

A classe minoritária recebeu maior peso durante o treinamento.

O recall aumentou significativamente, porém houve forte aumento de falsos positivos.

### SMOTENC

Foi utilizado **SMOTENC** em vez do SMOTE convencional porque várias features são binárias/categóricas.

Dessa forma foram evitados exemplos sintéticos inválidos, como:

```text
novo_ip = 0.37
vpn_proxy = 0.68
```

O balanceamento foi aplicado **somente sobre o conjunto de treinamento**, mantendo o conjunto de teste completamente original.

### Random Forest

Também foi avaliada uma `RandomForestClassifier` com:

```python
n_estimators=200
class_weight="balanced"
random_state=42
```

Apesar da capacidade de modelar relações não lineares, a configuração testada apresentou desempenho inferior à Logistic Regression neste dataset.

---

## 📈 Resultados

O conjunto de teste utilizado na comparação contém:

```text
3.000 registros

2.784 normais
216 suspeitos
```

### Comparação para a classe suspeita

| Modelo / Estratégia | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: |
| Logistic Regression — threshold 0.50 | 0.00 | 0.00 | 0.00 |
| Logistic Regression — class weight balanced | 0.16 | 0.68 | 0.26 |
| Logistic Regression + SMOTENC | 0.15 | 0.69 | 0.24 |
| Random Forest balanced | 0.16 | 0.12 | 0.14 |
| **Logistic Regression + threshold ajustado** | **0.25** | **0.32** | **0.28** |

### Capacidade de discriminação

| Modelo | ROC-AUC | Average Precision |
| :--- | ---: | ---: |
| Logistic Regression | **0.7536** | **0.2052** |
| Logistic Regression balanced | 0.7534 | 0.2029 |
| Logistic Regression + SMOTENC | 0.7522 | 0.2000 |
| Random Forest | 0.6792 | 0.1193 |

A Logistic Regression original apresentou a melhor capacidade de ranqueamento entre os experimentos realizados.

---

## 🎚️ Ajuste de Threshold

O threshold padrão da Logistic Regression é aproximadamente:

```text
0.50
```

Porém, nenhuma previsão no teste ultrapassou esse limite na primeira avaliação.

Em vez de escolher arbitrariamente um novo valor utilizando o conjunto de teste, o conjunto de treinamento foi novamente dividido:

```text
5.600 registros → treinamento interno
1.400 registros → validação
3.000 registros → teste final
```

O threshold foi escolhido exclusivamente sobre o conjunto de validação buscando o maior F1-score.

O valor encontrado foi:

```text
threshold ≈ 0.1529
```

Na validação:

```text
Precision ≈ 27,9%
Recall    ≈ 30,7%
F1        ≈ 29,2%
```

Depois de congelar esse valor, ele foi aplicado ao conjunto de teste.

Resultado final:

```text
Precision = 0.25
Recall    = 0.32
F1        = 0.28
Accuracy  = 0.88
```

Matriz de confusão:

```text
[[2573  211]
 [ 147   69]]
```

Ou seja:

```text
TN = 2573
FP = 211
FN = 147
TP = 69
```

O threshold escolhido representa um compromisso entre detectar logins suspeitos e limitar a quantidade de falsos alertas.

Ele **não deve ser interpretado como um valor universal de segurança** e foi otimizado especificamente para o dataset sintético utilizado nesta versão.

---

## 🏗️ Arquitetura do Projeto

A estrutura principal do LogEye é semelhante a:

```text
LogEye/
│
├── data/
│   └── logins.csv
│
├── models/
│   ├── logeye_v1.joblib
│   └── logeye_v1_config.json
│
├── notebooks/
│   └── analise.ipynb
│
├── src/
│   ├── api.py
│   ├── gerar_dataset.py
│   ├── predict.py
│   └── preprocessing.py
│   
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Principais arquivos

#### `src/gerar_dataset.py`

Responsável pela geração do dataset sintético utilizado no projeto.

#### `notebooks/analise.ipynb`

Contém:

- análise exploratória;
- visualizações;
- feature engineering;
- treinamento;
- comparação dos modelos;
- métricas;
- análise Precision-Recall;
- ajuste de threshold.

#### `src/preprocessing.py`

Transforma os dados brutos recebidos pela aplicação nas mesmas 9 features utilizadas durante o treinamento.

Entre as transformações:

```text
distancia_km
→ distancia_log

tempo_desde_ultimo_login
→ tempo_log

timestamp_login
→ login_madrugada
```

#### `src/predict.py`

Responsável por:

- carregar o modelo persistido;
- carregar a configuração da versão;
- aplicar o pré-processamento;
- gerar o score;
- aplicar o threshold;
- retornar a decisão.

#### `src/api.py`

Expõe o LogEye através de uma API REST desenvolvida com FastAPI.

---

## 🌐 API REST com FastAPI

O LogEye possui dois endpoints principais.

### Health Check

```http
GET /health
```

Resposta:

```json
{
  "status": "ok",
  "versao": "1.0"
}
```

Esse endpoint permite verificar se o serviço está disponível.

---

### Analisar Login

```http
POST /analisar
```

Exemplo de requisição:

```json
{
  "timestamp_login": "2026-08-31 03:20:00",
  "tentativas_falhas": 3,
  "novo_dispositivo": 1,
  "novo_ip": 1,
  "vpn_proxy": 0,
  "pais_diferente": 1,
  "distancia_km": 850,
  "tempo_desde_ultimo_login": 45,
  "primeiro_login": 0
}
```

Exemplo de resposta:

```json
{
  "versao": "1.0",
  "score": 0.38313039804893684,
  "suspeito": 1
}
```

Neste exemplo:

```text
score ≈ 0.3831
threshold ≈ 0.1529

score > threshold
→ suspeito = 1
```

O campo `score` deve ser interpretado como **score produzido pelo modelo**, utilizado para apoiar uma decisão de risco.

Uma aplicação integrada ao LogEye poderia posteriormente utilizar essa resposta para decidir ações como:

- permitir normalmente o acesso;
- solicitar autenticação adicional;
- gerar um alerta;
- registrar o evento para investigação;
- solicitar MFA;
- notificar uma equipe responsável.

Essas ações não são executadas automaticamente pelo modelo nesta versão.

---

## 🚀 Como Executar

### Pré-requisitos

É necessário possuir:

- Python 3;
- Git;
- pip.

---

### 1. Clonar o repositório

```bash
git clone <https://github.com/soncine/LogEye>
cd LogEye
```

---

### 2. Criar um ambiente virtual

```bash
python -m venv .venv
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

---

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

---

### 4. Iniciar a API

Na raiz do projeto:

```bash
uvicorn src.api:app --reload
```

O servidor ficará disponível em:

```text
http://127.0.0.1:8000
```

---

### 5. Abrir a documentação interativa

O FastAPI gera automaticamente uma interface Swagger em:

```text
http://127.0.0.1:8000/docs
```

Nela é possível testar diretamente:

```text
GET  /health
POST /analisar
```

---

## 📦 Dependências

As principais bibliotecas utilizadas são:

| Biblioteca | Finalidade |
| :--- | :--- |
| **Pandas** | Manipulação de dados e criação dos DataFrames |
| **NumPy** | Operações matemáticas e geração de dados |
| **Matplotlib** | Visualizações durante a EDA |
| **Seaborn** | Visualizações estatísticas e matriz de correlação |
| **scikit-learn** | Modelos, métricas, pipelines e preprocessing |
| **imbalanced-learn** | SMOTENC para tratamento de classes desbalanceadas |
| **Joblib** | Persistência e carregamento do modelo |
| **FastAPI** | Construção da API REST |
| **Pydantic** | Validação dos dados recebidos pela API |
| **Uvicorn** | Servidor ASGI utilizado para executar a aplicação |

As versões exatas utilizadas estão disponíveis em:

```text
requirements.txt
```

---

## ⚠️ Limitações

A versão 1.0 possui limitações importantes que devem ser consideradas.

### Dataset sintético

Todos os dados foram gerados artificialmente.

As associações encontradas durante a análise exploratória representam as regras probabilísticas definidas durante a construção do dataset e **não devem ser interpretadas como estatísticas reais de cibersegurança**.

### Distribuição temporal simplificada

Os timestamps foram distribuídos de maneira aproximadamente uniforme ao longo do período simulado.

Em sistemas reais, autenticações normalmente apresentam padrões muito mais complexos de horário e frequência.

### Usuários sintéticos

Os usuários também foram distribuídos de maneira relativamente uniforme, sem perfis comportamentais individuais altamente complexos.

### Distância geográfica

`distancia_km` representa a distância entre uma localização considerada habitual e a localização atual.

Ela **não representa a distância entre o login anterior e o login atual**.

Portanto, a versão atual ainda não implementa uma verdadeira detecção de **impossible travel**.

### Score não representa confirmação de ataque

O resultado:

```json
{
  "suspeito": 1
}
```

significa apenas que o score ultrapassou o threshold definido pelo projeto.

Isso não comprova que:

- uma conta foi comprometida;
- ocorreu invasão;
- houve fraude;
- o usuário é malicioso.

O resultado deve ser interpretado como um **sinal para apoiar outras camadas de segurança**.

### Modelo educacional

A versão atual não foi treinada nem validada sobre dados reais de autenticação e não deve ser utilizada diretamente em ambientes de produção.

---

## 🔮 Próximas Evoluções

O projeto foi estruturado para permitir diferentes evoluções após a versão 1.0.

Entre as possibilidades:

- integração com APIs de geolocalização de IP;
- enriquecimento automático de informações de IP;
- identificação de ASN e provedor;
- detecção real de **impossible travel**;
- histórico comportamental individual por usuário;
- análise da localização do login anterior;
- análise de velocidade geográfica entre acessos;
- níveis de risco como `baixo`, `médio` e `alto`;
- regras diferentes de threshold de acordo com o contexto;
- autenticação multifator baseada em risco;
- notificações para usuários ou administradores;
- banco de dados para histórico de autenticações;
- monitoramento de drift do modelo;
- novos algoritmos de Machine Learning;
- XGBoost;
- otimização de hiperparâmetros;
- explicabilidade com SHAP;
- testes automatizados;
- Docker;
- CI/CD;
- deploy da API em ambiente de nuvem;
- dashboards de monitoramento de segurança.

---

## 🎓 Aprendizados

O desenvolvimento do LogEye envolveu diferentes etapas de um projeto completo de Machine Learning.

Entre os principais conceitos aplicados estão:

```text
Geração de dados
        ↓
Análise exploratória
        ↓
Feature engineering
        ↓
Separação treino/teste
        ↓
Pipeline
        ↓
Logistic Regression
        ↓
Métricas para dados desbalanceados
        ↓
ROC-AUC
        ↓
Precision-Recall
        ↓
class_weight
        ↓
SMOTENC
        ↓
Random Forest
        ↓
Threshold tuning
        ↓
Validação separada do teste
        ↓
Persistência do modelo
        ↓
Preprocessing reutilizável
        ↓
API REST
```

Um dos principais aprendizados foi perceber que **o modelo com maior accuracy não necessariamente é o melhor modelo para o problema**.

No baseline, por exemplo, a accuracy chegou a aproximadamente 92,8%, apesar de nenhum login suspeito ser detectado.

A análise de precision, recall, F1, ROC-AUC e Average Precision permitiu avaliar o problema de forma muito mais adequada.

---

## 📌 Contexto do Projeto

O LogEye foi desenvolvido como projeto de estudo no contexto do **Bootcamp Bradesco — GenAI, Dados e Cyber da DIO**, reunindo conceitos de análise de dados, Machine Learning, segurança da informação e desenvolvimento de APIs.

A versão **LogEye v1.0** representa a primeira implementação funcional do projeto, cobrindo desde a geração dos dados até a disponibilização das previsões através de uma API REST.

O projeto possui finalidade educacional e continuará evoluindo após esta primeira versão.

---

## 👁️ LogEye

```text
Observe.
Analise.
Classifique.
```

**Machine Learning aplicado à análise de risco em autenticações.**