#개인 실습 예제 kaggle에서 perth house prices 데이터셋을 다운로드 받았다.
#이 데이터로 모델 학습 이후 예측까지 만드는 코드를 실습해볼 예정이다
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
df = pd.read_csv('C:/Users/gemma/OneDrive/Desktop/sparta/database/all_perth_310121.csv')


print(df.info())
print(df.head())

#대략적인 정보를 확인하기 위해 info와 head 함수를 사용했다. 
#일단 주택에 관련된 정보인것을 확인할수있고, 침실수,화장실수,차고차량수,토지면적,건축연도를 통해서 가격을 한번 예측해보려고한다.
#결측치는 garage에 약 2500개 가량 존재 하기에 주차를 할수 없다고 가정을하고 결측치 처리를 값 0 으로 변경해 주려고한다.
#건축연도 build_year 같은경우에도 결측치가 3000개 넘게 존재 하는걸 확인했고, 결측치 값을 수정해주기 위해 describe로 전체 통계값을 확인해보았다.

print(df.describe())

#일단은 단순하게 시작을 했는데 통계값을 확인해보니, 이상치도 어느정도 존재하는것 같다. 토지면적이 999999로 찍혀 있거나 , 주차차량가능수 같은경우도 99대가 최대값인걸 보니 
#이상치도 어느정도 처리를 해줘야 할거 같다. 
#이후 이상치를 삭제해보려고 햇는데 문득 든 생각은 999999 나 99대의 주차댓수 이런식으로 데이터값을 냈다면 아마 주택가격부분도 저렇게 비슷한 이상치가 존재 할거라고 생각했고, 
#그럼 이상치가 들어간 행을 전부다 일단 지워 보는건 어떨까? 라는 생각을 했다. 어차피 특징은 3만개가 넘어가니 무난히 학습을 시킬것으로 예상했다.

df['GARAGE']=df['GARAGE'].fillna(0)

print(df.isnull().sum())

#이후 결측치를 확인해 보았는데 0으로 반환되었다.

features =['BEDROOMS','BATHROOMS','GARAGE','LAND_AREA']
target = 'PRICE'
#내가 알던건 변수에 값을 선언할때 df[['BEDROOMS,'BATHROOMS'~~'BUILD_YEAR']] 이렇게 입력을 해야 값이 선언이 되는걸로 알고있었는데,
#어차피 features는 문자열의 리스트이기 때문에 pandas에서 각 항목의 열 이름으로 바로 인식 한다는걸 이번 실습을 통해 알게되었다.

sns.displot(df['PRICE'])
np.log1p(df['PRICE'])

#IQR을 사용한 이상치 확인
def replace_outliers_with_mode(df, columns):
    df_copy = df.copy()
    for column in columns:
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
                # 최빈값 계산
        mode = df_copy[column].mode().iloc[0]
        
        # 이상치를 최빈값으로 대체
        df_copy.loc[df_copy[column] < lower_bound, column] = mode
        df_copy.loc[df_copy[column] > upper_bound, column] = mode
    return df_copy
#내가 인터넷강의를 통해서 들은 IQR 방식에서의 가장 표준이 되는 코드라고 알고있다. 1사분위수(Q1) 계산 (25번째 백분위수) , 3사분위수(Q3) 계산 (75번째 백분위수) ,하한값,상한값에
#대한 값  0.25,0.75,1.5는 가장 기본이 되는 값이다.
#이후에 df에  상한값과 하한값 사이에 있는 값들만 필터링하여 남기고 나머지는 제거 해주었다.


# 특성과 타겟 변수의 이상치 제거
df_clean = replace_outliers_with_mode(df, features + [target])

print(df_clean)
#이후에 코드를 확인해보니 데이터가 33000개 가량이 14000개로 줄어들었다. 약 18000개 가량이 삭제된 셈인데
#너무 데이터가 적은게 아닐까 싶은데 개인적으로 직접 자료를 찾아서 내가 직접 특징이나 가격까지 파악해서 실습하는 과정이라 자유롭기에
#일단 진행해보기로 했다.

# 특성과 타겟 분리
X = df_clean[features]
y = df_clean[target]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#내가 특정한 특징들의 값은 편차가 너무나 크다. 침실같은경우 최대값이 16이지만 토지 면적의 최대값은 999999이다.
#그렇다는건 모델을 학습하고 예측을 했을때 침실의 수가 가격을 예측하는데 거의 의미가 부여되지 않지 않을까 싶어서 스케일링을 진행하였다.

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

#r-squared score 값은 0.12 ... 12%?
#MSE는 35301086098... 이건 뭐..
#이게 내가 너무 많은 열을 삭제해서 학습이 제대로 되지 않은 걸까 ?
#아니면 학습할때 필요한 특징이 너무 적었던게 문제 인걸까 ?
#왜 이렇게 값이 낮게 나온지 궁금해서 gpt에게 물어봤고,
#개선 방안을 알려줬는데 ,
#1.이상치 처리 방법 개선
#2.더많은 특성 추가
#3.선형회귀 대신 비선형모델 사용
#4.새로운 특성을 만들어서 추가
#등을 개선방안으로 알려주었다.
#자 그럼 일단 해결할수 있는 부분 부터 시작해보자. 첫번째는 이상치를 제거하는게 아니라 평균값으로 바꿔보자. 음? 근데 사실 의미가 있을까 싶은데 일단 진행.
#이곳에서 진행하면 코드가 꼬여서 문제될까 걱정되서 다른 .py를 만들어 거기서 결측치를 평균값으로 변경해봤는데..
# (이후 좀더 공부해본 결과 이상치의값은 평균값에 영향을 많이 줄수있으니까 ,median 으로 중간값을 넣어줬어야된다는걸 깨달음 내일 좀더 수정하면서 진행 예정)
#      column_mean = df_copy[column].mean()
#        df_copy.loc[df_copy[column] < lower_bound, column] = column_mean
#        df_copy.loc[df_copy[column] > upper_bound, column] = column_mean
# #Mean Squared Error: 45740963892.314 , R-squared Score: 0.1501386267087671 예측률은 15% 정도로 상승했지만 MSE는 더 높이 치솟았다.
#그래서 이번에는 다시 이상치를 제거하고 IQR값을 변경해보았다.
#q1의 값을 0.2로 q3의 값을 0.8으로 계산 해봤고,
#Mean Squared Error: 52182901863.88958 ,R-squared Score: 0.2030289455062677 예측률은 20프로 상승 MSE는 더 높아졌다. 예측률이 올라갔으면 내 생각으로는 MSE도 당연히 0으로 가까워져야될텐데 왜 반비례할까 ? 궁금해졌다.
#gpt를 통해 확인해 봤고 
#R2 = 얼마나 잘 나타내는지 확인
#MSE = 예측 값과 실제 값의 차이를 제곱해서 평균
#즉 R2는 상대적인 성능을 나타내는거고 MSE는 절대적인 오차를 나타내기 때문에 다를수있다
#그렇다면 R2는 모델에 많은 변수가 추가되면 올라갈수있고 , MSE는 이상치가 많으면 높아질수있으니까
#이상치를 좀더 많이 제거해주면 왠지 치솟기만하는 MSE를 잡아낼수있을거같다. 
#Mean Squared Error: 21139569104.618073,R-squared Score: 0.06492206196231509 예측률은 0프로 지만 MSE는 60프로이상 줄이는데 성공했다.
#의미가 있나... 여러가지 값들을 수정해 보았지만 만족할만한 값을 찾아내지 못했다.
#내일 튜터님들한테 물어보고 마저 학습하자
print('랜덤 포레스트')

#특성 추가 같은 경우 코드를 전체적으로 손봐야할수도 있을거같아서 일단은 random forest를 진행해본다
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
#random forest 모델을 만들어줌
model2 = DecisionTreeClassifier(random_state=42) #random_state 는 결정트리 모델의 랜덤성을 제어하는데 사용됨.
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred2)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred2)}")
#정확도는 0.004234.. 한가지의 모델만 더 실험해보고 이후에도 비슷한 결과를 얻게 된다면,
#내가 진행한 방식이 잘못된거니 다시한번 뜯어서 수정해봐야겠다.


#... 무슨 문제인지는 모르겠는데 xgboost를 설치하고 나서부터 이후로 numpy 관련해서 오류가 계속 발생해서 
#진행하고있던 이 프로젝트가 더이상 진행되지않는다..
# 이전 도전과제 같은경우도 더 이상 진행이 되지않는다.
#힘빠지네..뭔가 하던걸 마무리 못하고 어정쩡하게 끝나버리니까.

#1030 문제 해결 !
#진행을 못하는동안 왜그렇게 예측값이 낮았는지 곰곰히 생각을 해봤는데 아무리 생각해도, 건축년도를 집어 넣은게 예측값에 크게 영향을 주는것 같다는 생각을했다.
#방 갯수, 화장실 갯수 주차차량,토지면적 등은 주택가격에 영향을 주기 쉽지만
#건축년도 같은경우는 1800년대에 지었지만 오히려 가치가 높은 경우도 있을수 있으니까 ?
#그래서 특징을 오히려 빼주면 예측치가 올라갈거 같다는 희망을 가진상태에서 바로 진행했지만 
#Mse 값은 400억? R-squared Score:0.08382252636820098
#랜덤포레스트 같은경우 0.005 오히려 더 낮은 값을 보여준다.
#이후 데이터가 너무 적어서 그럴수 있을거같아서 이상치를 제거하는게 아닌 최빈값으로 바꿔주었지만
#Mean Squared Error: 49956150586.94633 R-squared Score: 0.08722231836268579 랜덤포레스트 Accuracy: 0.036690433749257276
#아무리 생각해도 진행이 혼자 어려워서 튜터님께 문의 드렸지만 퇴근이슈로
#코드랑 데이터 튜터님께 전달하고 내일 같이 해결예정

