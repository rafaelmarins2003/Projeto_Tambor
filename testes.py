#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo Preditivo para Desempenho em Corridas de Tambor

Este script realiza uma análise preditiva aprofundada do desempenho do conjunto
cavalo-cavaleiro em corridas de tambor, considerando o histórico conjunto e individual
de cada um, utilizando todas as variáveis disponíveis no dataset.

Autor: Manus AI
Data: Maio 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import joblib
import os

# Configurações
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# Função para carregar e preparar os dados
def carregar_e_preparar_dados(caminho_arquivo):
    """
    Carrega o dataset e realiza a limpeza e preparação inicial dos dados.

    Args:
        caminho_arquivo: Caminho para o arquivo Excel com os dados

    Returns:
        DataFrame pandas limpo e preparado para análise
    """
    print("Carregando dados...")
    df = pd.read_excel(caminho_arquivo)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")

    # Converter colunas de tempo para numérico
    colunas_tempo = ['S1', 'T1', 'S2', 'T2', 'S3', 'T3', 'Total']
    for col in colunas_tempo:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remover linhas sem informação de cavalo ou competidor
    df = df.dropna(subset=['Horse', 'Competitior'])

    # Remover linhas sem tempo total (nossa variável alvo)
    df = df.dropna(subset=['Total'])

    # Verificar e tratar valores extremos no tempo total
    q1 = df['Total'].quantile(0.01)
    q3 = df['Total'].quantile(0.99)
    df = df[(df['Total'] >= q1) & (df['Total'] <= q3)]

    # Ordenar por evento e competidor para análise temporal
    if 'Event' in df.columns:
        df = df.sort_values(['Event', 'Competitior', 'Horse'])

    print(f"Dataset após limpeza inicial: {df.shape[0]} linhas")
    return df


# Função para criar features históricas
def criar_features_historicas(df):
    """
    Cria features históricas para cavalos, competidores e duplas.

    Args:
        df: DataFrame pandas com os dados limpos

    Returns:
        DataFrame pandas com features históricas adicionadas
    """
    print("Criando features históricas...")

    # Criar identificador único para cada dupla cavalo-competidor
    df['Dupla'] = df['Horse'] + ' - ' + df['Competitior']

    # Estatísticas históricas por cavalo
    stats_cavalo = df.groupby('Horse')['Total'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_cavalo.columns = ['Horse', 'cavalo_count', 'cavalo_mean', 'cavalo_std', 'cavalo_min', 'cavalo_max']

    # Estatísticas históricas por competidor
    stats_competidor = df.groupby('Competitior')['Total'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_competidor.columns = ['Competitior', 'competidor_count', 'competidor_mean', 'competidor_std',
                                'competidor_min', 'competidor_max']

    # Estatísticas históricas por dupla
    stats_dupla = df.groupby('Dupla')['Total'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    stats_dupla.columns = ['Dupla', 'dupla_count', 'dupla_mean', 'dupla_std', 'dupla_min', 'dupla_max']

    # Juntar as estatísticas ao dataframe original
    df = pd.merge(df, stats_cavalo, on='Horse', how='left')
    df = pd.merge(df, stats_competidor, on='Competitior', how='left')
    df = pd.merge(df, stats_dupla, on='Dupla', how='left')

    # Calcular a experiência do cavalo com diferentes competidores
    exp_cavalo = df.groupby('Horse')['Competitior'].nunique().reset_index()
    exp_cavalo.columns = ['Horse', 'cavalo_num_competidores']

    # Calcular a experiência do competidor com diferentes cavalos
    exp_competidor = df.groupby('Competitior')['Horse'].nunique().reset_index()
    exp_competidor.columns = ['Competitior', 'competidor_num_cavalos']

    # Juntar as estatísticas de experiência ao dataframe
    df = pd.merge(df, exp_cavalo, on='Horse', how='left')
    df = pd.merge(df, exp_competidor, on='Competitior', how='left')

    # Calcular a consistência (coeficiente de variação)
    df['cavalo_cv'] = df['cavalo_std'] / df['cavalo_mean']
    df['competidor_cv'] = df['competidor_std'] / df['competidor_mean']
    df['dupla_cv'] = df['dupla_std'] / df['dupla_mean']

    # Calcular a diferença entre o desempenho da dupla e o desempenho individual
    df['diff_dupla_cavalo'] = df['dupla_mean'] - df['cavalo_mean']
    df['diff_dupla_competidor'] = df['dupla_mean'] - df['competidor_mean']

    # Calcular a taxa de melhoria/piora da dupla em relação ao esperado
    df['expected_mean'] = (df['cavalo_mean'] + df['competidor_mean']) / 2
    df['dupla_improvement'] = df['expected_mean'] - df['dupla_mean']

    # Estatísticas por classe de competição
    if 'Class' in df.columns:
        stats_classe = df.groupby('Class')['Total'].agg(['count', 'mean', 'std']).reset_index()
        stats_classe.columns = ['Class', 'classe_count', 'classe_mean', 'classe_std']
        df = pd.merge(df, stats_classe, on='Class', how='left')

    # Estatísticas por evento
    if 'Event' in df.columns:
        stats_evento = df.groupby('Event')['Total'].agg(['count', 'mean', 'std']).reset_index()
        stats_evento.columns = ['Event', 'evento_count', 'evento_mean', 'evento_std']
        df = pd.merge(df, stats_evento, on='Event', how='left')

    # Calcular estatísticas de tempos parciais (se disponíveis)
    parciais = ['S1', 'T1', 'S2', 'T2', 'S3', 'T3']
    for parcial in parciais:
        if parcial in df.columns:
            # Média por cavalo
            temp = df.groupby('Horse')[parcial].mean().reset_index()
            temp.columns = ['Horse', f'cavalo_{parcial}_mean']
            df = pd.merge(df, temp, on='Horse', how='left')

            # Média por competidor
            temp = df.groupby('Competitior')[parcial].mean().reset_index()
            temp.columns = ['Competitior', f'competidor_{parcial}_mean']
            df = pd.merge(df, temp, on='Competitior', how='left')

            # Média por dupla
            temp = df.groupby('Dupla')[parcial].mean().reset_index()
            temp.columns = ['Dupla', f'dupla_{parcial}_mean']
            df = pd.merge(df, temp, on='Dupla', how='left')

    # Calcular taxa de penalidades
    if 'B1' in df.columns and 'B2' in df.columns and 'B3' in df.columns:
        # Converter para indicador de penalidade (1 se tem penalidade, 0 caso contrário)
        df['penalty_B1'] = df['B1'].notna().astype(int)
        df['penalty_B2'] = df['B2'].notna().astype(int)
        df['penalty_B3'] = df['B3'].notna().astype(int)
        df['total_penalties'] = df['penalty_B1'] + df['penalty_B2'] + df['penalty_B3']

        # Taxa de penalidades por cavalo
        temp = df.groupby('Horse')['total_penalties'].mean().reset_index()
        temp.columns = ['Horse', 'cavalo_penalty_rate']
        df = pd.merge(df, temp, on='Horse', how='left')

        # Taxa de penalidades por competidor
        temp = df.groupby('Competitior')['total_penalties'].mean().reset_index()
        temp.columns = ['Competitior', 'competidor_penalty_rate']
        df = pd.merge(df, temp, on='Competitior', how='left')

        # Taxa de penalidades por dupla
        temp = df.groupby('Dupla')['total_penalties'].mean().reset_index()
        temp.columns = ['Dupla', 'dupla_penalty_rate']
        df = pd.merge(df, temp, on='Dupla', how='left')

    # Calcular taxa de desclassificações
    if 'DQ' in df.columns:
        df['is_DQ'] = df['DQ'].notna().astype(int)

        # Taxa de DQ por cavalo
        temp = df.groupby('Horse')['is_DQ'].mean().reset_index()
        temp.columns = ['Horse', 'cavalo_DQ_rate']
        df = pd.merge(df, temp, on='Horse', how='left')

        # Taxa de DQ por competidor
        temp = df.groupby('Competitior')['is_DQ'].mean().reset_index()
        temp.columns = ['Competitior', 'competidor_DQ_rate']
        df = pd.merge(df, temp, on='Competitior', how='left')

        # Taxa de DQ por dupla
        temp = df.groupby('Dupla')['is_DQ'].mean().reset_index()
        temp.columns = ['Dupla', 'dupla_DQ_rate']
        df = pd.merge(df, temp, on='Dupla', how='left')

    print(f"Dataset com features históricas: {df.shape[1]} colunas")
    return df


# Função para preparar os dados para modelagem
def preparar_para_modelagem(df, target='Total'):
    """
    Prepara os dados para modelagem, separando features e target,
    e tratando valores ausentes.

    Args:
        df: DataFrame pandas com features históricas
        target: Nome da coluna alvo para previsão

    Returns:
        X: Features para modelagem
        y: Variável alvo
        feature_names: Nomes das features
    """
    print("Preparando dados para modelagem...")

    # Definir colunas a serem usadas como features
    colunas_numericas = [
        'cavalo_count', 'cavalo_mean', 'cavalo_std', 'cavalo_min', 'cavalo_max',
        'competidor_count', 'competidor_mean', 'competidor_std', 'competidor_min', 'competidor_max',
        'dupla_count', 'dupla_mean', 'dupla_std', 'dupla_min', 'dupla_max',
        'cavalo_num_competidores', 'competidor_num_cavalos',
        'cavalo_cv', 'competidor_cv', 'dupla_cv',
        'diff_dupla_cavalo', 'diff_dupla_competidor', 'dupla_improvement'
    ]

    # Adicionar estatísticas de tempos parciais se disponíveis
    parciais = ['S1', 'T1', 'S2', 'T2', 'S3', 'T3']
    for parcial in parciais:
        col_cavalo = f'cavalo_{parcial}_mean'
        col_competidor = f'competidor_{parcial}_mean'
        col_dupla = f'dupla_{parcial}_mean'

        if col_cavalo in df.columns:
            colunas_numericas.append(col_cavalo)
        if col_competidor in df.columns:
            colunas_numericas.append(col_competidor)
        if col_dupla in df.columns:
            colunas_numericas.append(col_dupla)

    # Adicionar taxas de penalidades se disponíveis
    if 'cavalo_penalty_rate' in df.columns:
        colunas_numericas.extend(['cavalo_penalty_rate', 'competidor_penalty_rate', 'dupla_penalty_rate'])

    # Adicionar taxas de desclassificação se disponíveis
    if 'cavalo_DQ_rate' in df.columns:
        colunas_numericas.extend(['cavalo_DQ_rate', 'competidor_DQ_rate', 'dupla_DQ_rate'])

    # Adicionar estatísticas de classe e evento se disponíveis
    if 'classe_mean' in df.columns:
        colunas_numericas.extend(['classe_count', 'classe_mean', 'classe_std'])

    if 'evento_mean' in df.columns:
        colunas_numericas.extend(['evento_count', 'evento_mean', 'evento_std'])

    # Colunas categóricas (para one-hot encoding)
    colunas_categoricas = ['Class'] if 'Class' in df.columns else []

    # Filtrar apenas duplas com histórico suficiente
    df_filtered = df[df['dupla_count'] >= 3].copy()
    print(f"Registros após filtrar duplas com pelo menos 3 corridas: {df_filtered.shape[0]}")

    # Separar features e target
    X = df_filtered[colunas_numericas + colunas_categoricas].copy()
    y = df_filtered[target].copy()

    # Verificar e tratar valores ausentes nas features
    for col in X.columns:
        missing = X[col].isna().sum()
        if missing > 0:
            print(f"Coluna {col} tem {missing} valores ausentes ({missing / len(X) * 100:.2f}%)")

    # Retornar features, target e nomes das features
    return X, y, colunas_numericas, colunas_categoricas


# Função para treinar e avaliar modelos
def treinar_e_avaliar_modelos(X, y, colunas_numericas, colunas_categoricas):
    """
    Treina e avalia diferentes modelos de regressão.

    Args:
        X: Features para modelagem
        y: Variável alvo
        colunas_numericas: Lista de colunas numéricas
        colunas_categoricas: Lista de colunas categóricas

    Returns:
        Melhor modelo treinado e resultados da avaliação
    """
    print("Treinando e avaliando modelos...")

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar preprocessador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), colunas_numericas),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), colunas_categoricas)
        ]
    )

    # Definir modelos a serem testados
    modelos = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(n_estimators=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=20, random_state=42)
    }

    # Treinar e avaliar cada modelo
    resultados = {}
    melhor_rmse = float('inf')
    melhor_modelo = None
    melhor_pipeline = None

    for nome, modelo in modelos.items():
        # Criar pipeline com preprocessamento e modelo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', modelo)
        ])

        # Treinar modelo
        pipeline.fit(X_train, y_train)

        # Fazer previsões
        y_pred = pipeline.predict(X_test)

        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Armazenar resultados
        resultados[nome] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }

        print(f"{nome}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

        # Verificar se é o melhor modelo até agora
        if rmse < melhor_rmse:
            melhor_rmse = rmse
            melhor_modelo = nome
            melhor_pipeline = pipeline

    print(f"\nMelhor modelo: {melhor_modelo} com RMSE = {melhor_rmse:.4f}")

    # Se o melhor modelo for Random Forest ou Gradient Boosting, analisar importância das features
    if melhor_modelo in ['Random Forest', 'Gradient Boosting']:
        modelo = melhor_pipeline.named_steps['model']

        # Obter nomes das features após preprocessamento
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat' and len(cols) > 0:
                # Para colunas categóricas, obter nomes após one-hot encoding
                encoder = trans.named_steps['onehot']
                cats = encoder.get_feature_names_out(cols)
                feature_names.extend(cats)

        # Obter importância das features
        importancias = modelo.feature_importances_

        # Criar DataFrame com importâncias
        if len(feature_names) == len(importancias):
            df_importancias = pd.DataFrame({
                'Feature': feature_names,
                'Importância': importancias
            }).sort_values('Importância', ascending=False)

            print("\nImportância das features (top 10):")
            print(df_importancias.head(10))

            # Plotar importância das features
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importância', y='Feature', data=df_importancias.head(15))
            plt.title(f'Importância das Features - {melhor_modelo}', fontsize=16)
            plt.tight_layout()
            plt.savefig('importancia_features.png', dpi=300, bbox_inches='tight')

    return melhor_pipeline, resultados


# Função para otimizar hiperparâmetros do melhor modelo
def otimizar_hiperparametros(X, y, melhor_modelo, colunas_numericas, colunas_categoricas):
    """
    Otimiza os hiperparâmetros do melhor modelo usando GridSearchCV.

    Args:
        X: Features para modelagem
        y: Variável alvo
        melhor_modelo: Pipeline com o melhor modelo
        colunas_numericas: Lista de colunas numéricas
        colunas_categoricas: Lista de colunas categóricas

    Returns:
        Modelo otimizado
    """
    print("\nOtimizando hiperparâmetros do melhor modelo...")

    # Identificar o tipo de modelo
    modelo_nome = melhor_modelo.named_steps['model'].__class__.__name__

    # Criar preprocessador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), colunas_numericas),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), colunas_categoricas)
        ]
    )

    # Definir grid de hiperparâmetros com base no tipo de modelo
    param_grid = {}

    if modelo_nome == 'RandomForestRegressor':
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(random_state=42))
        ])
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

    elif modelo_nome == 'GradientBoostingRegressor':
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(random_state=42))
        ])
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }

    elif modelo_nome == 'Ridge':
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(random_state=42))
        ])
        param_grid = {
            'model__alpha': [0.1, 1.0, 10.0, 100.0]
        }

    elif modelo_nome == 'Lasso':
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', Lasso(random_state=42))
        ])
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0]
        }

    else:  # LinearRegression ou outro
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        param_grid = {}

    # Se temos hiperparâmetros para otimizar
    if param_grid:
        # Criar grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        # Treinar grid search
        grid_search.fit(X, y)

        # Mostrar melhores hiperparâmetros
        print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
        print(f"Melhor RMSE (CV): {-grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    else:
        # Se não há hiperparâmetros para otimizar, retornar o modelo original
        pipeline.fit(X, y)
        return pipeline


# Função para fazer previsões para novas duplas
def prever_para_novas_duplas(modelo, df_original, df_features):
    """
    Faz previsões para duplas cavalo-cavaleiro que ainda não correram juntas,
    mas têm histórico individual.

    Args:
        modelo: Modelo treinado
        df_original: DataFrame original
        df_features: DataFrame com features

    Returns:
        DataFrame com previsões para novas duplas potenciais
    """
    print("\nFazendo previsões para novas combinações potenciais...")

    # Identificar cavalos e competidores com histórico suficiente
    cavalos_validos = df_features[df_features['cavalo_count'] >= 5]['Horse'].unique()
    competidores_validos = df_features[df_features['competidor_count'] >= 5]['Competitior'].unique()

    print(f"Cavalos com histórico suficiente: {len(cavalos_validos)}")
    print(f"Competidores com histórico suficiente: {len(competidores_validos)}")

    # Limitar para um número razoável de combinações
    if len(cavalos_validos) > 20:
        cavalos_validos = cavalos_validos[:20]
    if len(competidores_validos) > 20:
        competidores_validos = competidores_validos[:20]

    # Criar todas as combinações possíveis
    combinacoes = []
    for cavalo in cavalos_validos:
        for competidor in competidores_validos:
            # Verificar se esta dupla já existe no dataset
            dupla = f"{cavalo} - {competidor}"
            if dupla not in df_features['Dupla'].values:
                # Obter estatísticas do cavalo
                stats_cavalo = df_features[df_features['Horse'] == cavalo].iloc[0]

                # Obter estatísticas do competidor
                stats_competidor = df_features[df_features['Competitior'] == competidor].iloc[0]

                # Criar registro para a nova dupla
                registro = {
                    'Horse': cavalo,
                    'Competitior': competidor,
                    'Dupla': dupla,
                    'cavalo_count': stats_cavalo['cavalo_count'],
                    'cavalo_mean': stats_cavalo['cavalo_mean'],
                    'cavalo_std': stats_cavalo['cavalo_std'],
                    'cavalo_min': stats_cavalo['cavalo_min'],
                    'cavalo_max': stats_cavalo['cavalo_max'],
                    'competidor_count': stats_competidor['competidor_count'],
                    'competidor_mean': stats_competidor['competidor_mean'],
                    'competidor_std': stats_competidor['competidor_std'],
                    'competidor_min': stats_competidor['competidor_min'],
                    'competidor_max': stats_competidor['competidor_max'],
                    'cavalo_num_competidores': stats_cavalo['cavalo_num_competidores'],
                    'competidor_num_cavalos': stats_competidor['competidor_num_cavalos'],
                    'cavalo_cv': stats_cavalo['cavalo_cv'],
                    'competidor_cv': stats_competidor['competidor_cv'],
                    # Valores estimados para a dupla
                    'dupla_count': 0,
                    'dupla_mean': (stats_cavalo['cavalo_mean'] + stats_competidor['competidor_mean']) / 2,
                    'dupla_std': (stats_cavalo['cavalo_std'] + stats_competidor['competidor_std']) / 2,
                    'dupla_min': max(stats_cavalo['cavalo_min'], stats_competidor['competidor_min']),
                    'dupla_max': min(stats_cavalo['cavalo_max'], stats_competidor['competidor_max']),
                    'dupla_cv': (stats_cavalo['cavalo_cv'] + stats_competidor['competidor_cv']) / 2,
                    'diff_dupla_cavalo': 0,  # Será calculado abaixo
                    'diff_dupla_competidor': 0,  # Será calculado abaixo
                    'expected_mean': (stats_cavalo['cavalo_mean'] + stats_competidor['competidor_mean']) / 2,
                    'dupla_improvement': 0  # Será calculado abaixo
                }

                # Adicionar estatísticas de tempos parciais se disponíveis
                parciais = ['S1', 'T1', 'S2', 'T2', 'S3', 'T3']
                for parcial in parciais:
                    col_cavalo = f'cavalo_{parcial}_mean'
                    col_competidor = f'competidor_{parcial}_mean'
                    col_dupla = f'dupla_{parcial}_mean'

                    if col_cavalo in df_features.columns and col_competidor in df_features.columns:
                        registro[col_cavalo] = stats_cavalo[col_cavalo]
                        registro[col_competidor] = stats_competidor[col_competidor]
                        registro[col_dupla] = (stats_cavalo[col_cavalo] + stats_competidor[col_competidor]) / 2

                # Adicionar taxas de penalidades se disponíveis
                if 'cavalo_penalty_rate' in df_features.columns:
                    registro['cavalo_penalty_rate'] = stats_cavalo['cavalo_penalty_rate']
                    registro['competidor_penalty_rate'] = stats_competidor['competidor_penalty_rate']
                    registro['dupla_penalty_rate'] = (stats_cavalo['cavalo_penalty_rate'] + stats_competidor[
                        'competidor_penalty_rate']) / 2

                # Adicionar taxas de desclassificação se disponíveis
                if 'cavalo_DQ_rate' in df_features.columns:
                    registro['cavalo_DQ_rate'] = stats_cavalo['cavalo_DQ_rate']
                    registro['competidor_DQ_rate'] = stats_competidor['competidor_DQ_rate']
                    registro['dupla_DQ_rate'] = (stats_cavalo['cavalo_DQ_rate'] + stats_competidor[
                        'competidor_DQ_rate']) / 2

                # Adicionar classe mais comum do competidor se disponível
                if 'Class' in df_features.columns:
                    classe_comum = df_original[df_original['Competitior'] == competidor]['Class'].mode().iloc[0]
                    registro['Class'] = classe_comum

                    if 'classe_mean' in df_features.columns:
                        classe_stats = df_features[df_features['Class'] == classe_comum].iloc[0]
                        registro['classe_count'] = classe_stats['classe_count']
                        registro['classe_mean'] = classe_stats['classe_mean']
                        registro['classe_std'] = classe_stats['classe_std']

                combinacoes.append(registro)

    # Criar DataFrame com as combinações
    if combinacoes:
        df_combinacoes = pd.DataFrame(combinacoes)

        # Fazer previsões
        X_pred = df_combinacoes.drop(['Horse', 'Competitior', 'Dupla'], axis=1, errors='ignore')

        # Garantir que temos as mesmas colunas que o modelo espera
        colunas_modelo = modelo.named_steps['preprocessor'].transformers_[0][2]  # Colunas numéricas
        for col in colunas_modelo:
            if col not in X_pred.columns:
                X_pred[col] = 0  # Valor padrão para colunas ausentes

        # Fazer previsões
        previsoes = modelo.predict(X_pred)

        # Adicionar previsões ao DataFrame
        df_combinacoes['Tempo_Previsto'] = previsoes

        # Ordenar por tempo previsto (do menor para o maior)
        df_combinacoes = df_combinacoes.sort_values('Tempo_Previsto')

        # Selecionar colunas relevantes para o resultado
        colunas_resultado = ['Horse', 'Competitior', 'Tempo_Previsto', 'cavalo_mean', 'competidor_mean',
                             'expected_mean']
        if 'Class' in df_combinacoes.columns:
            colunas_resultado.append('Class')

        return df_combinacoes[colunas_resultado].head(20)
    else:
        print("Não foram encontradas novas combinações potenciais.")
        return None


# Função para analisar uma dupla específica
def analisar_dupla_especifica(df, cavalo, competidor):
    """
    Analisa em detalhes o desempenho de uma dupla específica.

    Args:
        df: DataFrame com os dados
        cavalo: Nome do cavalo
        competidor: Nome do competidor

    Returns:
        DataFrame com análise detalhada
    """
    print(f"\nAnalisando dupla específica: {competidor} - {cavalo}")

    # Filtrar dados da dupla
    df_dupla = df[(df['Horse'] == cavalo) & (df['Competitior'] == competidor)].copy()

    if len(df_dupla) == 0:
        print("Dupla não encontrada no dataset.")
        return None

    print(f"Total de corridas da dupla: {len(df_dupla)}")

    # Estatísticas básicas
    stats = df_dupla['Total'].describe()
    print(f"Estatísticas de tempo da dupla:")
    print(stats)

    # Evolução do desempenho ao longo do tempo
    if 'Event' in df_dupla.columns:
        df_dupla = df_dupla.sort_values('Event')

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(df_dupla)), df_dupla['Total'], marker='o')
        plt.axhline(y=df_dupla['Total'].mean(), color='r', linestyle='--',
                    label=f'Média: {df_dupla["Total"].mean():.2f}s')
        plt.title(f'Evolução do Tempo - {competidor} com {cavalo}', fontsize=16)
        plt.xlabel('Corrida (ordem cronológica)', fontsize=14)
        plt.ylabel('Tempo Total (segundos)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('evolucao_dupla.png', dpi=300, bbox_inches='tight')

    # Comparar com o desempenho do cavalo com outros competidores
    df_cavalo = df[df['Horse'] == cavalo].copy()
    outros_competidores = df_cavalo[df_cavalo['Competitior'] != competidor]['Competitior'].unique()

    if len(outros_competidores) > 0:
        print(f"\nComparando com desempenho do cavalo {cavalo} com outros competidores:")

        comparacoes = []
        for outro in outros_competidores:
            tempos = df_cavalo[df_cavalo['Competitior'] == outro]['Total']
            if len(tempos) >= 3:  # Apenas competidores com pelo menos 3 corridas
                comparacoes.append({
                    'Competidor': outro,
                    'Corridas': len(tempos),
                    'Média': tempos.mean(),
                    'Mínimo': tempos.min(),
                    'Diferença': tempos.mean() - df_dupla['Total'].mean()
                })

        if comparacoes:
            df_comparacoes = pd.DataFrame(comparacoes).sort_values('Média')
            print(df_comparacoes)

            # Plotar comparação
            plt.figure(figsize=(12, 6))
            competidores = [competidor] + df_comparacoes['Competidor'].tolist()
            medias = [df_dupla['Total'].mean()] + df_comparacoes['Média'].tolist()

            bars = plt.bar(competidores, medias)
            bars[0].set_color('green')  # Destacar o competidor atual

            plt.title(f'Tempo Médio do Cavalo {cavalo} com Diferentes Competidores', fontsize=16)
            plt.xlabel('Competidor', fontsize=14)
            plt.ylabel('Tempo Médio (segundos)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('comparacao_competidores.png', dpi=300, bbox_inches='tight')

    # Comparar com o desempenho do competidor com outros cavalos
    df_competidor = df[df['Competitior'] == competidor].copy()
    outros_cavalos = df_competidor[df_competidor['Horse'] != cavalo]['Horse'].unique()

    if len(outros_cavalos) > 0:
        print(f"\nComparando com desempenho do competidor {competidor} com outros cavalos:")

        comparacoes = []
        for outro in outros_cavalos:
            tempos = df_competidor[df_competidor['Horse'] == outro]['Total']
            if len(tempos) >= 3:  # Apenas cavalos com pelo menos 3 corridas
                comparacoes.append({
                    'Cavalo': outro,
                    'Corridas': len(tempos),
                    'Média': tempos.mean(),
                    'Mínimo': tempos.min(),
                    'Diferença': tempos.mean() - df_dupla['Total'].mean()
                })

        if comparacoes:
            df_comparacoes = pd.DataFrame(comparacoes).sort_values('Média')
            print(df_comparacoes)

            # Plotar comparação
            plt.figure(figsize=(12, 6))
            cavalos = [cavalo] + df_comparacoes['Cavalo'].tolist()
            medias = [df_dupla['Total'].mean()] + df_comparacoes['Média'].tolist()

            bars = plt.bar(cavalos, medias)
            bars[0].set_color('green')  # Destacar o cavalo atual

            plt.title(f'Tempo Médio do Competidor {competidor} com Diferentes Cavalos', fontsize=16)
            plt.xlabel('Cavalo', fontsize=14)
            plt.ylabel('Tempo Médio (segundos)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('comparacao_cavalos.png', dpi=300, bbox_inches='tight')

    return df_dupla


# Função principal
def main(caminho_arquivo, analisar_dupla=None):
    """
    Função principal que executa todo o pipeline de análise.

    Args:
        caminho_arquivo: Caminho para o arquivo Excel com os dados
        analisar_dupla: Tupla (cavalo, competidor) para análise específica
    """
    # Criar diretório para resultados
    os.makedirs('resultados', exist_ok=True)

    # Carregar e preparar dados
    df = carregar_e_preparar_dados(caminho_arquivo)

    # Criar features históricas
    df_features = criar_features_historicas(df)

    # Preparar para modelagem
    X, y, colunas_numericas, colunas_categoricas = preparar_para_modelagem(df_features)

    # Treinar e avaliar modelos
    melhor_modelo, resultados = treinar_e_avaliar_modelos(X, y, colunas_numericas, colunas_categoricas)

    # Otimizar hiperparâmetros
    modelo_otimizado = otimizar_hiperparametros(X, y, melhor_modelo, colunas_numericas, colunas_categoricas)

    # Salvar modelo
    joblib.dump(modelo_otimizado, 'resultados/modelo_preditivo_tambor.pkl')
    print("Modelo salvo em 'resultados/modelo_preditivo_tambor.pkl'")

    # Fazer previsões para novas duplas
    df_previsoes = prever_para_novas_duplas(modelo_otimizado, df, df_features)
    if df_previsoes is not None:
        df_previsoes.to_csv('resultados/previsoes_novas_duplas.csv', index=False)
        print("Previsões para novas duplas salvas em 'resultados/previsoes_novas_duplas.csv'")

    # Analisar dupla específica se solicitado
    if analisar_dupla:
        cavalo, competidor = analisar_dupla
        df_analise = analisar_dupla_especifica(df, cavalo, competidor)
        if df_analise is not None:
            df_analise.to_csv(f'resultados/analise_{cavalo}_{competidor}.csv', index=False)
            print(f"Análise da dupla salva em 'resultados/analise_{cavalo}_{competidor}.csv'")

    print("\nAnálise preditiva concluída com sucesso!")


# Executar o script se for o arquivo principal
if __name__ == "__main__":
    # Caminho para o arquivo de dados
    caminho_arquivo = 'ML_Equs.xlsx'

    # Exemplo de dupla para análise específica (descomentar para usar)
    # analisar_dupla = ('MARI TA FAME HSC', 'MARIA EDUARDA DUARTE')

    # Executar análise
    # main(caminho_arquivo)

    # Para analisar uma dupla específica, use:
    main(caminho_arquivo, analisar_dupla=('MARI TA FAME HSC', 'MARIA EDUARDA DUARTE'))
