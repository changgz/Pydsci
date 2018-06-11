
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#拒绝推断" data-toc-modified-id="拒绝推断-0.1"><span class="toc-item-num">0.1&nbsp;&nbsp;</span>拒绝推断</a></span><ul class="toc-item"><li><span><a href="#第一步准备数据集：把解释变量和被解释变量分开，这是KNN这个函数的要求" data-toc-modified-id="第一步准备数据集：把解释变量和被解释变量分开，这是KNN这个函数的要求-0.1.1"><span class="toc-item-num">0.1.1&nbsp;&nbsp;</span>第一步准备数据集：把解释变量和被解释变量分开，这是KNN这个函数的要求</a></span></li><li><span><a href="#第二步：进行缺失值填补和标准化，这也是knn这个函数的要求" data-toc-modified-id="第二步：进行缺失值填补和标准化，这也是knn这个函数的要求-0.1.2"><span class="toc-item-num">0.1.2&nbsp;&nbsp;</span>第二步：进行缺失值填补和标准化，这也是knn这个函数的要求</a></span></li><li><span><a href="#第三步：建模并预测" data-toc-modified-id="第三步：建模并预测-0.1.3"><span class="toc-item-num">0.1.3&nbsp;&nbsp;</span>第三步：建模并预测</a></span></li><li><span><a href="#第四步：将审核通过的申请者和未通过的申请者进行合并" data-toc-modified-id="第四步：将审核通过的申请者和未通过的申请者进行合并-0.1.4"><span class="toc-item-num">0.1.4&nbsp;&nbsp;</span>第四步：将审核通过的申请者和未通过的申请者进行合并</a></span></li></ul></li><li><span><a href="#建立违约预测模型" data-toc-modified-id="建立违约预测模型-0.2"><span class="toc-item-num">0.2&nbsp;&nbsp;</span>建立违约预测模型</a></span><ul class="toc-item"><li><span><a href="#粗筛变量" data-toc-modified-id="粗筛变量-0.2.1"><span class="toc-item-num">0.2.1&nbsp;&nbsp;</span>粗筛变量</a></span></li><li><span><a href="#变量细筛与数据清洗" data-toc-modified-id="变量细筛与数据清洗-0.2.2"><span class="toc-item-num">0.2.2&nbsp;&nbsp;</span>变量细筛与数据清洗</a></span></li><li><span><a href="#变量分箱WOE转换" data-toc-modified-id="变量分箱WOE转换-0.2.3"><span class="toc-item-num">0.2.3&nbsp;&nbsp;</span>变量分箱WOE转换</a></span></li><li><span><a href="#构造分类模型" data-toc-modified-id="构造分类模型-0.2.4"><span class="toc-item-num">0.2.4&nbsp;&nbsp;</span>构造分类模型</a></span></li><li><span><a href="#检验模型" data-toc-modified-id="检验模型-0.2.5"><span class="toc-item-num">0.2.5&nbsp;&nbsp;</span>检验模型</a></span></li><li><span><a href="#评分卡开发" data-toc-modified-id="评分卡开发-0.2.6"><span class="toc-item-num">0.2.6&nbsp;&nbsp;</span>评分卡开发</a></span></li></ul></li></ul></li></ul></div>

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#get_ipython().magic('matplotlib inline')


# In[2]:

os.chdir(r'D:\Python_Training\script_Python\8logistic')


# In[3]:

accepts = pd.read_csv('accepts.csv')
rejects = pd.read_csv('rejects.csv')


# In[ ]:

'''
#信用风险建模案例
##数据说明：本数据是一份汽车贷款违约数据
##名称---中文含义
##application_id---申请者ID
##account_number---帐户号
##bad_ind---是否违约
##vehicle_year---汽车购买时间
##vehicle_make---汽车制造商
##bankruptcy_ind---曾经破产标识
##tot_derog---五年内信用不良事件数量(比如手机欠费消号)
##tot_tr---全部帐户数量
##age_oldest_tr---最久账号存续时间(月)
##tot_open_tr---在使用帐户数量
##tot_rev_tr---在使用可循环贷款帐户数量(比如信用卡)
##tot_rev_debt---在使用可循环贷款帐户余额(比如信用卡欠款)
##tot_rev_line---可循环贷款帐户限额(信用卡授权额度)
##rev_util---可循环贷款帐户使用比例(余额/限额)
##fico_score---FICO打分
##purch_price---汽车购买金额(元)
##msrp---建议售价
##down_pyt---分期付款的首次交款
##loan_term---贷款期限(月)
##loan_amt---贷款金额
##ltv---贷款金额/建议售价*100
##tot_income---月均收入(元)
##veh_mileage---行使历程(Mile)
##used_ind---是否使用
##weight---样本权重
'''

##################################################################################################################
# ## 一、拒绝推断

# ### 第一步准备数据集：把解释变量和被解释变量分开，这是KNN这个函数的要求

# In[4]:
#取出部分变量用于做KNN：由于KNN算法要求使用连续变量，因此仅选了部分重要的连续变量用于做KNN模型
accepts_x = accepts[["tot_derog","age_oldest_tr","rev_util","fico_score","ltv"]]
# In[5]:

accepts_y = accepts['bad_ind']
# In[6]:

rejects_x = rejects[["tot_derog","age_oldest_tr","rev_util","fico_score","ltv"]]

# In[ ]:
# ### 第二步：进行缺失值填补和标准化，这也是knn这个函数的要求

# In[ ]:
#查看一下数据集的信息
rejects_x.info()


# In[ ]:
## 定义缺失值替换函数
def Myfillna_median(df):
    for i in df.columns:
        median = df[i].median()
        df[i].fillna(value=median, inplace=True)
    return df

# In[ ]:
# 缺失值填补
accepts_x_filled=Myfillna_median(df=accepts_x)

rejects_x_filled=Myfillna_median(df=rejects_x)

# In[8]:

# 标准化数据
from sklearn.preprocessing import Normalizer
accepts_x_norm = pd.DataFrame(Normalizer().fit_transform(accepts_x_filled))
accepts_x_norm.columns = accepts_x_filled.columns

rejects_x_norm = pd.DataFrame(Normalizer().fit_transform(rejects_x_filled))
rejects_x_norm.columns = rejects_x_filled.columns


# ### 第三步：建模并预测

# In[9]:

# 利用knn模型进行预测，做拒绝推断
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
neigh.fit(accepts_x_norm, accepts_y) 


# In[10]:

rejects['bad_ind'] = neigh.predict(rejects_x_norm)


# ### 第四步：将审核通过的申请者和未通过的申请者进行合并

# In[ ]:

# accepts的数据是针对于违约用户的过度抽样
#因此，rejects也要进行同样比例的抽样


# In[11]:

rejects_res = rejects[rejects['bad_ind'] == 0].sample(1340)
rejects_res = pd.concat([rejects_res, rejects[rejects['bad_ind'] == 1]], axis = 0)


# In[12]:

data = pd.concat([accepts.iloc[:, 2:-1], rejects_res.iloc[:,1:]], axis = 0)

##################################################################################################################
# ## 二、建立违约预测模型

# ### 粗筛变量

# In[13]:

# 分类变量转换
bankruptcy_dict = {'N':0, 'Y':1}
data.bankruptcy_ind = data.bankruptcy_ind.map(bankruptcy_dict)


# In[14]:

# 盖帽法处理年份变量中的异常值，并将年份其转化为距现在多长时间
# 此处只是一个示例，所有连续变量都要按此方法进行处理
year_min = data.vehicle_year.quantile(0.1)
year_max = data.vehicle_year.quantile(0.99)
data.vehicle_year = data.vehicle_year.map(lambda x: year_min if x <= year_min else x)
data.vehicle_year = data.vehicle_year.map(lambda x: year_max if x >= year_max else x)

data.vehicle_year = data.vehicle_year.map(lambda x: 2018 - x)


# In[15]:

data.drop(['vehicle_make'], axis = 1, inplace = True)


# In[ ]:
data_filled=Myfillna_median(df=data)
# In[17]:

X = data_filled[['age_oldest_tr', 'bankruptcy_ind', 'down_pyt', 'fico_score',
       'loan_amt', 'loan_term', 'ltv', 'msrp', 'purch_price', 'rev_util',
       'tot_derog', 'tot_income', 'tot_open_tr', 'tot_rev_debt',
       'tot_rev_line', 'tot_rev_tr', 'tot_tr', 'used_ind', 'veh_mileage',
       'vehicle_year']]
y = data_filled['bad_ind']


# In[18]:

# 利用随机森林填补变量
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X,y)


# In[19]:

importances = list(clf.feature_importances_)
importances_order = importances.copy()
importances_order.sort(reverse=True)

cols = list(X.columns)
col_top = []
for i in importances_order[:9]:
    col_top.append((i,cols[importances.index(i)]))
col_top


# In[20]:

col = [i[1] for i in col_top]


# ### 变量细筛与数据清洗

# In[21]:

from woe import WoE
import warnings
warnings.filterwarnings("ignore")


# In[22]:

data_filled.head()


# In[23]:

iv_c = {}
for i in col:
    try:
        iv_c[i] = WoE(v_type='c').fit(data_filled[i],data_filled['bad_ind']).optimize().iv 
    except:
        print(i)
    
pd.Series(iv_c).sort_values(ascending=False)


# ### 变量分箱WOE转换

# In[24]:

WOE_c = data_filled[col].apply(lambda col:WoE(v_type='c',qnt_num=5).fit(col,data_filled['bad_ind']).optimize().fit_transform(col,data_filled['bad_ind']))


# In[25]:

WOE_c.head()


# ### 构造分类模型

# In[26]:

# 划分数据集
from sklearn.cross_validation import train_test_split
X = WOE_c
y = data_filled['bad_ind']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


# In[27]:

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:

# 构建逻辑回归模型，进行违约概率预测
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,classification_report 
lr = LogisticRegression(C = 1, penalty = 'l1')
lr.fit(X_train,y_train.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[46]:

## 加入代价敏感参数，重新计算
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,classification_report 
lr = LogisticRegression(C = 1, penalty = 'l1', class_weight='balanced')
lr.fit(X_train,y_train.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# ### 检验模型

# In[47]:

from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(y_test,y_pred, drop_intermediate=False) ###计算真正率和假正率  
roc_auc = auc(fpr,tpr) ###计算auc的值  
  
plt.figure()  
lw = 2  
plt.figure(figsize=(10,10))  
plt.plot(fpr, tpr, color='darkorange',  
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线  
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')  
plt.legend(loc="lower right")  
plt.show()


# In[31]:

# 利用sklearn.metrics中的roc_curve算出tpr，fpr作图


fig, ax = plt.subplots()
ax.plot(1 - threshold, tpr, label='tpr') # ks曲线要按照预测概率降序排列，所以需要1-threshold镜像
ax.plot(1 - threshold, fpr, label='fpr')
ax.plot(1 - threshold, tpr-fpr,label='KS')
plt.xlabel('score')
plt.title('KS Curve')
#plt.xticks(np.arange(0,1,0.2), np.arange(1,0,-0.2))
#plt.xticks(np.arange(0,1,0.2), np.arange(score.max(),score.min(),-0.2*(data['反欺诈评分卡总分'].max() - data['反欺诈评分卡总分'].min())))
plt.figure(figsize=(20,20))
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')

plt.show()
#%%