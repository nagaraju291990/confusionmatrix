from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support as score

with open('gold.txt', 'r', encoding='utf-8') as file:
    y_true = file.read().split("\n")
with open('predicted.txt', 'r', encoding='utf-8') as file:
    y_pred = file.read().split("\n")


classes = ["ADJ","ADJ ","ADP","ADV","ADV ","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB"]

cm = confusion_matrix(y_true, y_pred, labels=classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
#disp.plot()
## this will set image size
fig, ax = plt.subplots(figsize=(50,50))
## this will make labels look neat on x-axis
disp.plot(ax=ax, xticks_rotation='vertical')
plt.show()

##this section calculates precision, recall and f1
from sklearn.metrics import precision_score, recall_score, f1_score

p = precision_score(y_true, y_pred, labels=classes, average='weighted')
r = recall_score(y_true, y_pred, labels=classes, average='weighted')
f1 = f1_score(y_true, y_pred, labels=classes, average='weighted')

print(p, r, f1)
print('precision: {}'.format(p))
print('recall: {}'.format(r))
print('fscore: {}'.format(f1))
# # precision, recall, fscore, support = score(y_true, y_pred)
# # print('precision: {}'.format(precision))
# # print('recall: {}'.format(recall))
# # print('fscore: {}'.format(fscore))
# # print('support: {}'.format(support))