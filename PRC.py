from typing import Dict
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.metrics import plot_roc_curve,roc_curve,auc,roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

y_test= pd.read_excel(r'F:\多类样本不均衡分组增量学习\result\ROC+PRC+F1-score，ACC\GHIL_labels.xlsx',header=None).values
y_score = pd.read_excel(r'F:\多类样本不均衡分组增量学习\result\ROC+PRC+F1-score，ACC\GHIL_pre_score.xlsx',header=None).values

plt.figure(figsize=(200, 100),dpi=150)
plt.plot((0, 1), (0, 1), ':', color = 'grey')
plt.style.use('seaborn-darkgrid')

# precision recall curve
plt.subplot(1, 2, 1)
precision = dict()
recall = dict()
average_precision = dict()

#第一类
precision[1], recall[1], _ = precision_recall_curve(y_test[:, 0],y_score[:, 0])
average_precision[1] = average_precision_score(y_test[:, 0],y_score[:, 0])
plt.plot(recall[1], precision[1], lw=1.25, label='Healthy       ' + 'AP=%0.2f'%average_precision[1])

#第2类
precision[2], recall[2], _ = precision_recall_curve(y_test[:, 1],y_score[:, 1])
average_precision[2] = average_precision_score(y_test[:, 1],y_score[:, 1])
plt.plot(recall[2], precision[2], lw=1.25, label='Hepatitis B  ' + 'AP=%0.2f'%average_precision[2])

#第3类
precision[3], recall[3], _ = precision_recall_curve(y_test[:, 2],y_score[:, 2])
average_precision[3] = average_precision_score(y_test[:, 2],y_score[:, 2])
plt.plot(recall[3], precision[3], lw=1.25, label='Hepatitis C  ' + 'AP=%0.2f'%average_precision[3])

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.legend(loc="lower left")
plt.title("PRC curve of GHIL")

# roc curve
plt.subplot(1, 2, 2)
fpr = dict()
tpr = dict()
AUC = dict()

#第一类：
[fpr[1], tpr[1], _] = roc_curve(y_test[:, 0],y_score[:, 0])
AUC[1] = auc(fpr[1], tpr[1])
plt.plot(fpr[1], tpr[1], lw=1.25, label='Health         ' + 'AUC=%0.2f'%AUC[1])

#第二类：
[fpr[2], tpr[2], _] = roc_curve(y_test[:, 1],y_score[:, 1])
AUC[2] = auc(fpr[2], tpr[2])
plt.plot(fpr[2], tpr[2], lw=1.25, label='Hepatitis B  '+'AUC=%0.2f'%AUC[2])

#第3类：
[fpr[3], tpr[3], _] = roc_curve(y_test[:, 2],y_score[:, 2])
AUC[3] = auc(fpr[3], tpr[3])
plt.plot(fpr[3], tpr[3], lw=1.25, label='Hepatitis C  '+'AUC=%0.2f'%AUC[3])

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="best")
plt.title("ROC curve of GHIL")
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.show()

###计算F1score,ACC,PRC,Recall###
y_true= pd.read_excel(r'F:\多类样本不均衡分组增量学习\result\CNN_true_labels.xlsx',header=None).values
y_pre = pd.read_excel(r'F:\多类样本不均衡分组增量学习\result\CNN_pres.xlsx',header=None).values

sw = compute_sample_weight(class_weight='balanced',y=y_true)

prec =precision_score(y_true, y_pre, average='weighted',sample_weight=sw)
recall = recall_score(y_true, y_pre, average='weighted',sample_weight=sw)
f1score = f1_score(y_true, y_pre, average='weighted',sample_weight=sw)

acc = accuracy_score(y_true, y_pre)

print(acc)
print('precision_score',prec,'recall_score',recall,'F1-score',f1score)