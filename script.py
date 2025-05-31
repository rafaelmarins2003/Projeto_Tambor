import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 0. Leitura e setup
df = pd.read_excel('ML_Equs.xlsx')
df.columns = df.columns.str.lower()

# 1. Seleção de colunas e conversão numérica
cols = ['horse','competitior','s1','t1','s2','t2','s3','t3','fs','total','placing']
df = df[cols].copy()
for col in cols[2:]:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
df = df.dropna(subset=cols[2:]).copy()

# 2. Remoção de outliers
for col in ['s1','t1','s2','t2','s3','t3','fs']:
    z = (df[col] - df[col].mean()) / df[col].std()
    df = df[z.abs() < 3]

# 3. Cálculo de velocidades e features
dist_s1, dist_s2, dist_s3, dist_turn = 18.3, 32, 32, 4.7
df['vel_s1'] = dist_s1 / df['s1']
df['vel_s2'] = dist_s2 / df['s2']
df['vel_s3'] = dist_s3 / df['s3']
df['vel_t1'] = dist_turn / df['t1']
df['vel_t2'] = dist_turn / df['t2']
df['vel_t3'] = dist_turn / df['t3']
eps = 1e-6
df['accel_s1_s2'] = (df['vel_s2'] - df['vel_s1']) / df['t1']
df['accel_s2_s3'] = (df['vel_s3'] - df['vel_s2']) / df['t2']
df['explosion_idx'] = df['vel_s1'] / df['vel_s3']
df['area_vel'] = 0.5*(df['vel_s1']+df['vel_s2'])*(df['t1']+df['t2']) + 0.5*(df['vel_s2']+df['vel_s3'])*(df['t2']+df['t3'])
df['seg_std'] = df[['t1','t2','t3']].std(axis=1)
df['consistency_score'] = 1 / (df['seg_std'] + eps)

# 4. Perfis individuais
feat = ['vel_s1','vel_s2','vel_s3','accel_s1_s2','accel_s2_s3','explosion_idx','area_vel','consistency_score']
horse_prof = df.groupby('horse')[feat].mean().add_prefix('h_').reset_index()
comp_prof = df.groupby('competitior')[feat].mean().add_prefix('c_').reset_index()

# 5. Seleção top N entidades para reduzir combinações
horse_counts = df['horse'].value_counts()
comp_counts = df['competitior'].value_counts()
keep_h = horse_counts.nlargest(50).index
keep_c = comp_counts.nlargest(50).index
horse_prof = horse_prof[horse_prof['horse'].isin(keep_h)]
comp_prof  = comp_prof[comp_prof['competitior'].isin(keep_c)]

# 6. Combinações inéditas
hp = horse_prof[['horse']].copy(); hp['key'] = 1
cp = comp_prof[['competitior']].copy(); cp['key'] = 1
all_pairs = hp.merge(cp, on='key')[['horse','competitior']]

seen = df.groupby(['horse','competitior']).size().reset_index()[['horse','competitior']]
candidates = all_pairs.merge(seen.assign(seen=1), on=['horse','competitior'], how='left')\
                      .query('seen.isna()').drop('seen', axis=1)

# 7. Mescla perfis e calcula distância
cand = candidates.merge(horse_prof, on='horse').merge(comp_prof, on='competitior')
h_cols = [c for c in cand.columns if c.startswith('h_')]
c_cols = [c for c in cand.columns if c.startswith('c_')]
Xh = cand[h_cols].values; Xc = cand[c_cols].values
scaler = StandardScaler().fit(np.vstack([Xh, Xc]))
Xhs = scaler.transform(Xh); Xcs = scaler.transform(Xc)
cand['compat_dist'] = np.linalg.norm(Xhs - Xcs, axis=1)

# 8. Top 10 novas parcerias
top_new = cand.sort_values('compat_dist', ascending=False).head(10)[['horse','competitior','compat_dist']]
print(top_new)