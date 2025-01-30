import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def reduce_dimensions(vectors, labels, num_dimensions=2):
    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)  # fixed-width numpy strings

    # reduce using t-SNE，一定要指定随机种子，这样每次降维后的结果才能一样
    tsne = TSNE(n_components=num_dimensions, random_state=42, learning_rate='auto', n_iter=1000)
    vectors = tsne.fit_transform(vectors)

    x_vals = np.asarray([v[0] for v in vectors])
    y_vals = np.asarray([v[1] for v in vectors])
    return x_vals, y_vals, labels


def plot_word2vec(x_vals, y_vals, labels, is_show=False, is_save="True"):

    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
    plt.rcParams['legend.fontsize'] = 14

    random.seed(0)

    plt.figure(figsize=(10, 10), dpi=300)
    scatter = plt.scatter(x_vals, y_vals,c=labels,cmap='plasma')
    plt.legend(handles=scatter.legend_elements()[0],labels=['Western Han', 'Eastern Han', 'Western Jin', 'Southern Song',
              'Southern Liang', 'Northern Qi','Tang', 'Later Jin', 'Song',
              'Yuan', 'Ming', 'Qing'], loc='best',frameon=False,fontweight='bold')



    if is_save:
        plt.savefig("./cluster_distance.svg",format='svg')
    if is_show:
        plt.show()

def plant_matrix_confusion(y_true,y_pred):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
    labels = ['Western Han', 'Eastern Han', 'Western Jin', 'Southern Song',
              'Southern Liang', 'Northern Qi','Tang', 'Later Jin', 'Song',
              'Yuan', 'Ming', 'Qing']
    label_map = {label: i for i, label in enumerate(labels)}
    true_labels = np.array([label_map[label] for label in y_true])
    predicted_labels = np.array([label_map[label] for label in y_pred])
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_norm * 100, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.gcf().subplots_adjust(bottom=0.4)


    plt.savefig('./confunsion.svg', format='svg')
    plt.show()