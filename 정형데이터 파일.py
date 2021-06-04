#!/usr/bin/env python
# coding: utf-8

# # 7강 정형데이터 분류 베이스라인 모델 3 추가 피처 엔지니어링
# 
# - 세번째 베이스라인 모델에서 7강에서 설명한 피처 엔지니어링 추가 구현

# ## Function 정의
# 
# - seed_everything() : seed 고정 함수
# - generate_label() : label 생성 함수
# - print_score() : 평가지표 출력 함수
# - feature_preprocessing() : 데이터 전처리 함수(label encoding, 결측치 처리)
# - feature_engineering2() : 피처 엔지니어링 함수
# - make_lgb_oof_prediction() : LightGBM cross validation out of fold train/predict 함수
# - plot_feature_importances() : LightGBM 피처 중요도 플롯 함수

# ## Library Import

# In[322]:


# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import os, sys, gc, warnings, random

import datetime
import dateutil.relativedelta

# Data manipulation
import pandas as pd 
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from tqdm.notebook import trange, tqdm

from IPython.display import display

from catboost import CatBoostClassifier, Pool, cv

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 1000


# ## Fix Seed 함수 정의

# In[323]:


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
SEED = 42
seed_everything(SEED)


# ## 레이블 생성 함수 정의
# 
# - 레이블 생성 함수는 연월을 인자로 받아서 고객 아이디별로 총 구매액이 인자로 받는 구매액 임계값을 넘는지 여부를 바이너리 레이블로 생성하는 함수

# In[324]:


TOTAL_THRES = 300

'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label


# In[ ]:





# ## 평가지표 출력 함수 정의

# In[325]:


def print_score(label, pred, prob_thres=0.5):
    print('Precision: {:.5f}'.format(precision_score(label, pred>prob_thres)))
    print('Recall: {:.5f}'.format(recall_score(label, pred>prob_thres)))
    print('F1 Score: {:.5f}'.format(f1_score(label, pred>prob_thres)))
    print('ROC AUC Score: {:.5f}'.format(roc_auc_score(label, pred)))


# ## 데이터 전처리 함수 정의
# - 범주형 변수 레이블 인코딩
# - 결측치 중위값으로 채움

# In[326]:


def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te


# ## 피처 엔지니어링 함수 정의
# - Pandas groupby aggregation 함수를 사용한 피처 엔지니어링

# In[327]:


def feature_engineering2(df, year_month):
    df = df.copy()
    
    df["month_order"] = df["year_month"].replace(df["year_month"].unique(),list(range(len(df["year_month"].unique()))))

    
    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()
    
    #added last order date
    df['last_order_date_by_cust_id']=df.groupby(['customer_id'])['order_date'].last()
    df['last_order_date_by_cust_id']=df.groupby(['customer_id'])['price'].last()
    df['last_order_date_by_cust_id']=df.groupby(['customer_id'])['quantity'].last()
    df['last_order_date_by_cust_id']=df.groupby(['customer_id'])['country'].last()
    

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()  
    
    # country 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_country'] = df.groupby(['country'])['total'].cumsum()
    df['cumsum_quantity_by_country'] = df.groupby(['country'])['quantity'].cumsum()
    df['cumsum_price_by_country'] = df.groupby(['country'])['price'].cumsum()
    

    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','std','count','skew']
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
        'cumsum_total_by_country': agg_func,
        'cumsum_quantity_by_country': agg_func,
        'cumsum_price_by_country': agg_func
    }
   
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features


# ## LightGBM cross validation out of fold train/predict 함수 정의

# In[328]:


def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

         # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )
        
        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi


# ## LightGBM 피처 중요도 플롯 함수 정의

# In[329]:


def plot_feature_importances(df, n=20, color='blue', figsize=(12,8)):
    # 피처 중요도 순으로 내림차순 정렬
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # 피처 중요도 정규화 및 누적 중요도 계산
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    plt.style.use('fivethirtyeight')
    # 피처 중요도 순으로 n개까지 바플롯으로 그리기
    df.loc[:n, :].plot.barh(y='importance_normalized', 
                            x='feature', color=color, 
                            edgecolor='k', figsize=figsize,
                            legend=False)

    plt.xlabel('Normalized Importance', size=18); plt.ylabel(''); 
    plt.title(f'Top {n} Most Important Features', size=18)
    plt.gca().invert_yaxis()
    
    return df


# In[348]:


#Catboost 모델 추가 


# In[1]:


def make_catboost_oof_prediction(x_train, y_train):
    x_train = pd.read_csv('../input/train.csv')
    y_train = (label['total'] > 300).astype(int)
    
    # 테스트 데이터 예측값을 저장할 변수
    
    train_pool = Pool(x_train, y_train, Model_cols)
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    
    #Split data among train and validation
    from sklearn.model_selection import train_test_split
    x_train, x_validation, y_train, y_validation=train_test_split(train_data, target, train_size=0.75, random_state=42)

        
    #Model cols
    Model_cols = ['order_id', 'product_id', 'description', 'quantity', 'order_date', 'price',
    'customer_id ', 'country', 'total']
        
    # Dec_prediction, a list of categorical features
    Dec_prediction = np.where(train[Model_cols].dtypes != np.float)[0]
    for ind in Dec_prediction:
           print (train.columns[ind])
        
        
    # Catboost 모델 훈련          
    model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    depth=3,
    iterations=300,
    eval_metric='Accuracy',
    random_seed=42,
    logging_level='Silent',
    allow_writing_files=False
    )
    cv_data = cv(
    train_pool,
    model.get_params(),
    fold_count=5
    )
        
    # Validation 데이터 예측
    val_preds = model.predict(x_val)
        
    # Validation index에 예측값 저장 
    y_oof[val_idx] = val_preds
        
    # 폴드별 Validation 스코어 측정
    print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
    print('-'*80)

    # score 변수에 폴드별 평균 Validation 스코어 저장
    score += roc_auc_score(y_val, val_preds) / folds
        
    # 테스트 데이터 예측하고 평균해서 저장
    test_preds += clf.predict(x_test) / folds
        
    # 폴드별 피처 중요도 저장
    fi[f'fold_{fold+1}'] = model.feature_importance()

    del x_tr, x_val, y_tr, y_val
    gc.collect()
        
    print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']), 
    cv_data['test-Accuracy-std'][cv_data['test-Accuracy-mean'].idxmax(axis=0)],
    cv_data['test-Accuracy-mean'].idxmax(axis=0)
    ))
    
    print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))

    model.fit(train_pool);
    model.score(x_train, y_train)
        
    return y_oof, test_preds


# ## Train 데이터 읽기
# - 9개의 feature를 가진 780502개의 train data

# In[368]:


data = pd.read_csv("../input/train.csv", parse_dates=["order_date"])
print(data.shape) # data shape 확인
data.head()


# ## Pandas info() 함수로 데이터 타입 및 null 체크

# In[369]:


data.info()


# ## Pandas isna(), sum() 함수로 null 데이터 개수 체크

# In[370]:


data.isna().sum()


# ## Pandas describe() 함수에 include='all' 인자 설정으로 수치형, 범주형 데이터 기본 통계량 확인

# In[371]:


data.describe(include='all')


# ## 2011년 11월 Label 데이터 생성

# In[372]:


label_2011_11 = generate_label(data, '2011-11')['label']


# ## Label 데이터 분포 플롯

# In[373]:


sns.countplot(label_2011_11);
label_2011_11.value_counts()


# ## LightGBM Hyperparameter 선언
# - early stopping 기능을 사용하기 위해 n_estimators, early_stopping_rounds 설정

# In[374]:


#sns.countplot(label_2011_10);
#label_2011_10.value_counts()


# In[375]:


model_params = {
    'objective': 'binary', # 이진 분류
    'boosting_type': 'gbdt',
    'metric': 'auc', # 평가 지표 설정
    'feature_fraction': 0.8, # 피처 샘플링 비율
    'bagging_fraction': 0.8, # 데이터 샘플링 비율
    'bagging_freq': 1,
    'n_estimators': 15000, # 트리 개수
    'early_stopping_rounds': 100,
    'seed': SEED,
    'verbose': -1,
    'n_jobs': -1,    
}


# In[376]:


all_train = pd.DataFrame()
all_y = pd.Series()
data['year_month'] = data['order_date'].dt.strftime('%Y-%m')
year_month = '2011-11'

for ym in sorted(data[data["year_month"]<=year_month]["year_month"].unique()):
    if ym == '2009-12' or ym == '2010-01':
        continue
    print(ym)
    train, test, y, train_features = feature_engineering2(data,ym)
    all_train = pd.concat([all_train,train])
    all_y = pd.concat([all_y,y])## 2011년 11월 베이스라인 모델 예측


# ## 2011년 11월 베이스라인 모델 예측
# - Validation 성능 측정

# In[377]:


#train, test, y, features = feature_engineering2(data, '2011-11')
y_oof, test_preds_2011_11, fi = make_lgb_oof_prediction(all_train.reset_index(),all_y.reset_index()[0], test, features, model_params=model_params)
print_score(label_2011_11, test_preds_2011_11)
print(test.shape)


# In[378]:


print(test_preds.shape)


# In[ ]:


#train catboost (data, )


# ## 2011년 11월 베이스라인 모델 예측 데이터 분포

# In[343]:


sns.distplot(test_preds_2011_11)
plt.show()


# ## 2011년 12월 베이스라인 모델 예측
# - 테스트 데이터 제출

# In[344]:


train, test, y, features = feature_engineering2(data, '2011-12')
y_oof, test_preds, fi = make_lgb_oof_prediction(all_train.reset_index(), all_y.reset_index()[0], test, features, model_params=model_params)


# In[ ]:





# ## 2011년 12월 베이스라인 모델 예측 데이터 분포

# In[345]:


sns.distplot(test_preds)
plt.show()


# ## 피처 중요도 플롯

# In[346]:


fi = plot_feature_importances(fi)


# # 제출 파일 생성

# In[347]:


output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)
submission = pd.read_csv('../input/sample_submission.csv')
submission['probability'] = test_preds
submission.to_csv(os.path.join(output_dir, 'output.csv'), index=False)


# In[ ]:





# In[ ]:





# In[ ]:




