import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score
from tensorflow.keras import layers, Model
from sklearn.neighbors import KNeighborsClassifier
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mpl_toolkits.mplot3d import Axes3D

units = 200
best_model_path = "best_model/EE/"
h_data_path = "h_data/"
# initilizers = tf.keras.initializers.GlorotNormal(seed=np.randi())
def create_initializer(h_data_path, seed=None):
    save_directory = os.path.join(h_data_path, Parameter.name, "train")
    return tf.keras.initializers.GlorotNormal(seed=seed)


class Parameter(object):
    name = "alzheimer"
    num_classes = 2
    hidden_dim = 1920


class data_loader:
    def load_data(self, data_path):
        # input: data path, .npy file configuration "元件驗證"
        # output: np array, data size is dependent on data
        data = []
        for folder_name in classes:
            emotion_folder_path = os.path.join(data_path, folder_name)
            file_list = glob(os.path.join(emotion_folder_path, "{}_*.npy".format(folder_name)))
            class_data = []
            for file_path in file_list:
                np_array = np.load(file_path)
                class_data.append(np_array)
            # 合并该类别的所有数据
            combined_class_data = np.concatenate(class_data, axis=0)
            data.append(combined_class_data)
        shapes = [np.shape(arr) for arr in data]
        print(shapes)  # This will print the shapes of each array
        return np.array(data)


def ds_generator(data, label):
    # 設定隨機種子
    tf.random.set_seed(42)

    # 獲取類別數與每類樣本數
    num_classes, data_num, data_dim = data.shape

    # 對每個類別的資料和標籤分別打亂

    shuffled_indices = np.random.permutation(data.shape[1])
    shuffled_indices_tensor = tf.constant(shuffled_indices, dtype=tf.int32)

    shuffled_data = tf.gather(data, shuffled_indices_tensor, axis=1)
    shuffled_label = tf.gather(label, shuffled_indices_tensor, axis=1)

    # 計算每一類要取的數量
    half_batch_size = batch_size // 2

    # 初始化一個空的 tf.data.Dataset
    dataset = None

    for i in range(0, data_num, half_batch_size):
        # 提取第0類和第1類的 batch 資料
        class_0_data = shuffled_data[0, i:i + half_batch_size, :]
        class_0_labels = shuffled_label[0, i:i + half_batch_size]

        class_1_data = shuffled_data[1, i:i + half_batch_size, :]
        class_1_labels = shuffled_label[1, i:i + half_batch_size]

        # 合併兩類資料
        batch_data = tf.concat([class_0_data, class_1_data], axis=0)
        batch_labels = tf.concat([class_0_labels, class_1_labels], axis=0)

        # 建立單個 batch 的 dataset
        batch_dataset = tf.data.Dataset.from_tensor_slices((batch_data, batch_labels))

        # 將 batch 加入總 dataset 中
        if dataset is None:
            dataset = batch_dataset
        else:
            dataset = dataset.concatenate(batch_dataset)

    return dataset


def shuffle_per_class(data, labels):
    shuffled_data = []
    shuffled_labels = []
    for i in range(Parameter.num_classes):
        idx = tf.range(data.shape[1])
        idx = tf.random.shuffle(idx)
        shuffled_data.append(tf.gather(data[i], idx))
        shuffled_labels.append(tf.gather(labels[i], idx))
    return tf.stack(shuffled_data), tf.stack(shuffled_labels)


def tf_gather(x: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
    complete_indices = np.array(np.where(indices > -1))
    complete_indices[axis] = tf.reshape(indices, [-1])
    flat_ind = np.ravel_multi_index(tuple(complete_indices), np.array(x.shape))
    return tf.reshape(tf.gather(tf.reshape(x, [-1]), flat_ind), indices.shape)


class embedding_expansion(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(embedding_expansion, self).__init__()
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.function_f = self.function_f(input_dim, embedding_dim, name)
        self.classifier_g = self.classifier_g(input_dim, num_classes, name)
        self.checkpoint = tf.train.Checkpoint(
            function_f=self.function_f,
            classifier_g=self.classifier_g
        )

    def call(self, inputs, training=False):
        predictions = []
        x = inputs
        x = self.function_f.call(x, training=training)
        x = self.classifier_g.call(x, training=training)
        predictions.append(x)

        return predictions

    class function_f(tf.keras.Model):
        def __init__(self, input_dim, embedding_dim, name):
            super().__init__()
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.d_name = name
            self.build(input_shape=(None, input_dim))

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l0'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l1'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='f{}_l2'.format(self.d_name)), ]
        @tf.function
        def call(self, inputs, training=False):
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    class classifier_g(tf.keras.Model):
        def __init__(self, input_dim, num_classes, name, units=units):  # n為內插的點的個數
            super().__init__()
            self.clf_name = Parameter.name
            self.units = units
            self.d_name = name
            self.emb_dim = input_dim
            self.num_classes = num_classes
            self.build(input_shape=(None, input_dim))
            self.acc = tf.keras.metrics.SparseCategoricalAccuracy()

        def build(self, input_shape):
            self.Layers = [
                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l0'.format(self.d_name)),
                layers.LeakyReLU(alpha=0.3),

                layers.Dense(units=units,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l1'.format(self.d_name)),

                layers.LeakyReLU(alpha=0.3),
                layers.Dense(units=2,
                             activation='linear',
                             kernel_initializer=create_initializer(h_data_path),
                             bias_initializer=create_initializer(h_data_path),
                             name='g{}_l2'.format(self.d_name)), ]

        @tf.function
        def call(self, inputs, training=False):
            # print("call input:", type(inputs))
            x = inputs
            for layer in self.Layers:
                x = layer(x, training=training)
            return x

    def fit(self, data_d, labels, val_data, val_labels, n_inner_pts, batch_size, num_epochs, training):
        if training:
            shuffled_data, shuffled_labels = shuffle_per_class(data_d, labels)
            # 將每個類別分批
            batch_per_class = batch_size // Parameter.num_classes

            # 將每個類別資料切分成批次
            datasets = []
            for i in range(Parameter.num_classes):
                ds = tf.data.Dataset.from_tensor_slices((shuffled_data[i], shuffled_labels[i]))
                ds = ds.batch(batch_per_class)
                datasets.append(ds)

            # 合併不同類別的批次資料
            train_ds = tf.data.Dataset.zip(tuple(datasets))
            # val
            test_ds = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
            test_ds = test_ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
            test_ds = test_ds.shuffle(500).batch(batch_size)

            final_epoch_loss = []
            final_epoch_msloss = []
            val_acc_epochs = []
            best_accuracy = 0.0
            best_auroc = 0.0

            Auroc = []
            num_inner_pts = n_inner_pts
            best_epoch_num = 0
            for epoch in range(num_epochs):
                all_aug_data, all_aug_label = [], []
                all_org_data, all_org_label = [],[]
                epoch_msloss, epoch_loss = [], []

                auroc_epoch = 0.0
                epoch_accuracy = 0.0
                for class0, class1 in train_ds:

                    batch_data = tf.concat([class0[0],class1[0]], axis=0)
                    batch_labels = tf.concat([class0[1],class1[1]], axis=0)

                    batch_msloss, batch_loss, org_data, org_label, aug_data, aug_label = self.train_step(batch_data, batch_labels, num_inner_pts, cls_num=2)
                    # batch = batch + 1
                    epoch_loss.append(batch_loss)
                    epoch_msloss.append(batch_msloss)
                    # 將當前 batch 的 aug_data 和 aug_label 添加到列表中
                    all_aug_data.append(aug_data)
                    all_aug_label.append(aug_label)
                    all_org_data.append(org_data)
                    all_org_label.append(org_label)
                current_epoch_loss = tf.reduce_mean(epoch_loss).numpy()
                current_epoch_msloss = tf.reduce_mean(epoch_msloss).numpy()
                # epoch_loss = []
                print("Epoch {}: loss={}".format(epoch + 1, current_epoch_loss))
                final_epoch_loss.append(current_epoch_loss)
                final_epoch_msloss.append(current_epoch_msloss)
                # 在 epoch 结束时，将所有 batch 的 augmented data 和 label 拼接起来
                all_aug_data = tf.concat(all_aug_data, axis=0)
                all_aug_label = tf.concat(all_aug_label, axis=0)
                all_org_data = tf.concat(all_org_data, axis=0)
                all_org_label = tf.concat(all_org_label, axis=0)
                # 計算準確率
                correct_count = 0
                total_count = 0
                val_predictions = []
                val_labels = []
                all_preds = []
                all_labels = []
                for val_batch_data, val_batch_labels in test_ds:

                    predictions = self(val_batch_data)
                    predictions = predictions[0]
                    predictions_new = tf.argmax(predictions, axis=1).numpy()
                    all_preds.extend(predictions_new)
                    all_labels.extend(val_batch_labels.numpy())
                    for j in range(len(predictions_new)):
                        if predictions_new[j] == val_batch_labels[j]:
                            correct_count += 1
                        total_count += 1

                    # 然后进行 softmax 处理
                    predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
                    # 二元 roc_auc_score 需要取正例標籤的機率
                    predictions_prob = predictions_prob[:, 1]
                    auroc = roc_auc_score(val_batch_labels, predictions_prob)

                    Auroc.append(auroc)  # Auroc is list of all batches in 1 epoch

                epoch_accuracy = correct_count / total_count  # one epoch
                f1 = f1_score(all_labels, all_preds)
                auroc_epoch = tf.reduce_mean(Auroc)
                # accuracy higher then save model
                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_auroc = auroc_epoch
                    best_epoch_num = epoch + 1
                    best_aug_data = all_aug_data
                    best_aug_label = all_aug_label
                    best_org_data = all_org_data
                    best_org_label = all_org_label
                    best_f1 = f1
                    self.function_f.save_weights(
                        './EE_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

                    self.classifier_g.save_weights(
                        './EE_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))

                val_acc_epochs.append(epoch_accuracy)
            mean_val_acc = tf.reduce_mean(val_acc_epochs)

            print("best epoch:", best_epoch_num)
            print("mean_acc=", mean_val_acc)
            print("best_acc=", best_accuracy)
            print("best auroc= ", best_auroc)
            print("best f1: ", best_f1)

            # 將 org_data 和 aug_data 合併後進行正規化
            all_data = np.concatenate([best_org_data, best_aug_data], axis=0)
            scaler = StandardScaler()
            all_data_normalized = scaler.fit_transform(all_data)

            # 獲取標籤
            labels = np.concatenate([best_org_label, best_aug_label], axis=0)

            # 執行 t-SNE 降維 (3D)
            tsne = TSNE(n_components=3, random_state=42)
            tsne_results = tsne.fit_transform(all_data_normalized)

            # t-SNE 結果分開為原始數據和擴增數據
            tsne_original = tsne_results[:len(best_org_data)]
            tsne_aug = tsne_results[len(best_org_data):]

            # 建立 3D 圖形
            fig = plt.figure(figsize=(10, 7), dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            # 設置顏色和標記形狀
            colors = ['blue', 'red']
            shapes = ['o', '^']

            # 繪製原始數據 (藍色)
            for label in np.unique(best_org_label):
                mask = (best_org_label == label)
                ax.scatter(tsne_original[mask, 0], tsne_original[mask, 1], tsne_original[mask, 2],
                           marker=shapes[label], color=colors[0], label=f'Original Class {label + 1}', alpha=0.6)

            # 繪製擴增數據 (紅色)
            for label in np.unique(best_aug_label):
                mask = (best_aug_label == label)
                ax.scatter(tsne_aug[mask, 0], tsne_aug[mask, 1], tsne_aug[mask, 2],
                           marker=shapes[label], color=colors[1], label=f'Augmented Class {label + 1}', alpha=0.6)

            # 設置標題和軸標籤
            ax.set_title("t-SNE 3D Visualization of Normalized Original and Augmented Data", fontsize=16)
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.set_zlabel("t-SNE Dimension 3")

            # 設置視角角度
            ax.view_init(elev=30, azim=45)  # 可調整不同視角

            # 顯示圖例
            ax.legend(fontsize=16)
            plt.savefig(f'EE_{Parameter.name}_TSNE.png', dpi=100, bbox_inches='tight')
            plt.show()
            # 保存 augmented data 和 labels
            all_aug_data = all_data_normalized[len(best_org_data):]

            file_path = './Aug_data/EE_aug_data_{}'.format(Parameter.name)
            np.savez(file_path, all_aug_data=all_aug_data, all_aug_label=all_aug_label.numpy())
            print("Augmented data and labels saved to", file_path)
            save_dir = 'EE_weight'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # print("best recall=", best_recall, "best_precision=", best_precision)
            plt.plot(range(num_epochs), val_acc_epochs)
            plt.title('Epoch accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('accuracy')
            plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))  # 將圖片保存到 'plots' 文件夾中
            plt.close()  # 關閉圖形，防止它在屏幕上顯示

            return best_accuracy, best_auroc, best_f1

        else:
            val_data_num = val_data.shape[1]
            val_flattened_data = tf.reshape(val_data, (2 * val_data_num, Parameter.hidden_dim))
            val_flattened_labels = tf.reshape(val_labels, (2 * val_data_num,))

            test_accuracy, test_auroc, test_f1 = self.evaluate(val_flattened_data, val_flattened_labels)
            print("Acc = ", test_accuracy, "AUROC = ", test_auroc, "F1 = ", test_f1)

            return test_accuracy, test_auroc, test_f1

    # @tf.function
    def train_step(self, batch_data, batch_labels, num_inner_pts, cls_num):
        with tf.GradientTape(persistent=True) as tape:
            phi_data = self.function_f(batch_data)
            _ = self.classifier_g(phi_data)
            # Forward pass
            batch_data_input = [None, None]
            half_batch_size = tf.shape(batch_data)[0] // 2
            batch_data_input[0] = batch_data[:half_batch_size, :]
            batch_data_input[1] = batch_data[half_batch_size:, :]

            batch_label_input = [None, None]
            batch_label_input[0] = batch_labels[:half_batch_size]
            batch_label_input[1] = batch_labels[half_batch_size:]

            emb_cls, embedded_data = [], []
            data, label = [], []
            all_embedded_data, all_augmented_data = [], []
            new_embeddings_list, new_labels_list = [], []
            output_org, output_aug = [], []

            CE_loss = tf.constant(0)
            for cls_index in range(cls_num):
                emb_cls.append(self.function_f.call(batch_data_input[cls_index]))   #No.index class mapping
                #augmentation
                embedded_data, embedded_labels, new_embeddings, new_labels = get_embedding_aug(
                    emb_cls[cls_index],
                    batch_label_input[cls_index],
                    num_inner_pts,
                    num_instance=len(emb_cls[cls_index]),
                    l2_norm=True
                )
                output_org.append(self.classifier_g.call(emb_cls[cls_index], training=True))

                label.append(embedded_labels)  # 获取标签
                data.append(embedded_data)
                output_aug.append(self.classifier_g.call(embedded_data, training=True))

                # 收集新生成的 embeddings 和 labels
                new_embeddings_list.append(new_embeddings)
                new_labels_list.append(new_labels)

            # 将收集的列表转换为 TensorFlow 张量
            all_generated_data = tf.concat(new_embeddings_list, axis=0)
            all_generated_label = tf.concat(new_labels_list, axis=0)

            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            CE_loss_org = criterion(batch_label_input, output_org)    #原始batch_label reshape跟經過g預測的label
            CE_loss_aug = criterion(label, output_aug)

            embedded_data_aug = []
            embedded_labels = []
            CE_loss = CE_loss_org + CE_loss_aug
            # CE_loss = CE_loss_aug
            # 转换为 TensorFlow 张量
            embedded_data_aug = tf.concat(data, axis=0)
            embedded_labels = tf.concat(label, axis=0)
            msloss = ms_loss(embedded_labels, embedded_data_aug)

        # Compute gradients
        gradients1 = tape.gradient(msloss, self.function_f.trainable_variables)
        gradients2 = tape.gradient(CE_loss, self.classifier_g.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients1, self.function_f.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients2, self.classifier_g.trainable_variables))

        return msloss, CE_loss, phi_data, batch_labels, all_generated_data, all_generated_label

    def evaluate(self, val_data, val_label):
        predictions = self(val_data, training=False)
        # calculate acc
        correct_count = 0
        total_count = 0
        # test labels
        val_labels_array = np.array(val_label)
        # get max index as predict label
        predictions = predictions[0]
        predictions_new = tf.argmax(predictions, axis=1).numpy()

        # calculate test acc
        for j in range(len(predictions_new)):  # len(predictions_new[i])=16
            if np.all(predictions_new[j] == val_label[j]):
                correct_count += 1
            total_count += 1

        # 然后进行 softmax 处理
        predictions_prob = tf.keras.activations.softmax(predictions, axis=-1)
        # 二元 roc_auc_score 需要取正例標籤的機率
        predictions_prob = predictions_prob[:, 1]
        auroc = roc_auc_score(val_labels_array, predictions_prob)
        # one batch accuracy
        accuracy = correct_count / total_count
        f1 = f1_score(val_labels_array, predictions_new)
        return accuracy, auroc, f1


def get_embedding_aug(embeddings, labels, n_inner_pts, num_instance, l2_norm=True):
    batch_size = tf.shape(embeddings)[0]

    # 檢查 batch size 是否為偶數
    assert num_instance % 2 == 0, 'num_instance should be even number for proper pairing'

    # 建立 swap 配對 (每兩筆做一對)，例如 [0,1,2,3,4,5] -> swap_pairs: [1,0,3,2,5,4]
    swap_indices = tf.range(batch_size)
    swap_indices = tf.where(swap_indices % 2 == 0, swap_indices + 1, swap_indices - 1)

    # 對所有 embeddings 做 swap，anchor / positive pairing
    pos = tf.gather(embeddings, tf.range(0, batch_size, 2))      # 只取偶數位
    anchor = tf.gather(embeddings, tf.gather(swap_indices, tf.range(0, batch_size, 2)))

    # 對應的 labels 也要同步取偶數位
    paired_labels = tf.gather(labels, tf.range(0, batch_size, 2))

    # 儲存生成內插點
    generated_embeddings = []
    generated_labels = []

    total_length = float(n_inner_pts + 1)

    for n_idx in range(n_inner_pts):
        left_length = float(n_idx + 1)
        right_length = total_length - left_length

        # 插值產生新點
        inner_pts = (anchor * left_length + pos * right_length) / total_length
        # 假設 anchor, pos, inner_pts 都是 shape (N, 200) 的 Tensor
        is_on_line, err = check_interpolation_on_line(anchor, pos, inner_pts)

        # 印出結果
        print("有幾個內插點在線段上：", tf.reduce_sum(tf.cast(is_on_line, tf.int32)).numpy())
        print("平均偏差誤差：", tf.reduce_mean(err).numpy())
        if l2_norm:
            inner_pts = tf.math.l2_normalize(inner_pts, axis=1)

        generated_embeddings.append(inner_pts)
        generated_labels.append(paired_labels)

    # 合併成最終輸出
    generated_embeddings = tf.concat(generated_embeddings, axis=0)
    generated_labels = tf.concat(generated_labels, axis=0)

    # 合併原始與生成資料
    concat_embeddings = tf.concat([embeddings, generated_embeddings], axis=0)
    concat_labels = tf.concat([labels, generated_labels], axis=0)

    return concat_embeddings, concat_labels, generated_embeddings, generated_labels


def check_interpolation_on_line(anchor, pos, inner_pts, tol=1e-3):
    """
    驗證每個 inner_pts 是否在線段 anchor - pos 上。

    Parameters:
        anchor: Tensor, shape (N, D)
        pos: Tensor, shape (N, D)
        inner_pts: Tensor, shape (N, D)
        tol: 容許誤差，float，預設 1e-3

    Returns:
        Boolean Tensor of shape (N,) 表示是否在線段上
        Float Tensor of shape (N,) 表示每個內插點的誤差值（越接近0越在線上）
    """

    # 各段長度
    d_ab = tf.norm(pos - anchor, axis=1)  # AB 長度
    d_ap = tf.norm(inner_pts - anchor, axis=1)  # AP 長度
    d_pb = tf.norm(pos - inner_pts, axis=1)  # PB 長度

    # 誤差：內插點偏離線段的程度
    err = tf.abs((d_ap + d_pb) - d_ab)

    # 判斷誤差是否在容許範圍內
    is_on_line = err < tol

    return is_on_line, err


def ms_loss(labels, embeddings, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=True):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''
    # 1. 對於 i這個點的內差點做刪除處理 留下滿足下列條件的點
    # .       \max{(p,n)\inN{y[i],y[j]}}  s{p,n}    > \min_{y[i]=y[k]} s_{i,k}-eps
    # 2. loss
    # 1/N\sum _i {
    #  \sum_{k\in P_i} 1/\alpha \log{1 + e^{-\alpha{s_{i,k}-\lambda}}
    # .  \sum_{k\in N_i} 1/\beta \log{e^\beta{s_{i,k}-\lambda}}
    # }
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    # print(embeddings)
    labels = tf.reshape(labels, [-1, 1])
    # print(labels)
    batch_size = tf.size(labels)        #192
    adjacency = tf.equal(labels, tf.transpose(labels))      #(192, 192)
    adjacency_not = tf.logical_not(adjacency)
    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)
    sim_mat = sim_mat / tf.reduce_max(sim_mat)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        # 1. 對於 i這個點的內差點做刪除處理 留下滿足下列條件的點
        # .       \max{(p,n)\inN{y[i],y[j]}}  s{p,n}    > \min_{y[i]=y[k]} s_{i,k}-eps

        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    # 2. loss
    # 1/N\sum _i {
    #  \sum_{k\in P_i} 1/\alpha \log{1 + e^{-\alpha{s_{i,k}-\lambda}}
    # .  \sum_{k\in N_i} 1/\beta \log{e^\beta{s_{i,k}-\lambda}}
    # }
    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.math.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.math.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta     #inf

    loss = tf.reduce_mean(pos_term + neg_term)
    return loss





if __name__ == '__main__':
    print(tf.__version__)
    data_path = "/home/wanling/PycharmProjects/pythonProject/{}/train/".format(Parameter.name)
    test_data_path = "/home/wanling/PycharmProjects/pythonProject/{}/test/".format(Parameter.name)
    # classes = ["no", "yes"]
    # classes = ["Covid-19", "Normal"]
    classes = ["Demented", "NonDemented"]
    # classes = ["with_mask", "without_mask"]
    # classes = ["Positive", "Negative"]
    # classes = ['commyn', 'wesmea']
    # classes = ['notsmoking', 'smoking']
    # classes = ["bleached_corals", "healthy_corals"]
    num_runs = 3  # 设置运行次数
    batch_size = 64
    num_epochs = 200
    learning_rate = 0.0001
    variance = []
    acc_list = []
    auroc_list = []
    f1_list = []
    for run in range(num_runs):
        print(f"Run {run + 1}:")

        data_d = data_loader().load_data(data_path)         #load data_d
        data_val = data_loader().load_data(test_data_path)  #load test data
        # print(data_d.shape)

        class1_labels = [0] * (data_d.shape[1])  # make labels:covid->0, normal->1
        class2_labels = [1] * (data_d.shape[1])
        labels = np.vstack((class1_labels, class2_labels))  # shape:(2,500)
        # print("label:", labels.shape)

        val_class1_labels = [0] * (data_val.shape[1])
        val_class2_labels = [1] * (data_val.shape[1])
        val_labels = np.vstack((val_class1_labels, val_class2_labels))

        data_d = data_d / 255.0  # 正規化數據
        data_val = data_val / 255.0

        # 轉換 data_d 為 Tensor
        data_tensor = tf.convert_to_tensor(data_d, dtype=tf.float32)
        val_data_tensor = tf.convert_to_tensor(data_val, dtype=tf.float32)

        # 轉換 labels 為 Tensor
        label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
        val_label_tensor = tf.convert_to_tensor(val_labels, dtype=tf.int32)

        name = Parameter.name

        ee = embedding_expansion(input_dim=1920, embedding_dim=units, num_classes=2)
        f_model = ee.function_f
        dummy_input = tf.random.normal(shape=(batch_size, 1920))  # 替換為實際的輸入形狀和數據
        _ = ee.function_f(dummy_input, training=False)
        # 載入保存的權重
        f_model.load_weights('./EE_weight/function_f/f_model_weights_{}.h5'.format(Parameter.name))

        # 印出新模型的摘要
        f_model.summary()
        d_model = ee.classifier_g

        dummy_input2 = tf.random.normal(shape=(batch_size, 200))
        _ = ee.classifier_g(dummy_input2, training=False)
        d_model.load_weights('./EE_weight/classifier_g/g_model_weights_{}.h5'.format(Parameter.name))
        print("Trainable variables for classifier_g.Layers:")

        ee.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate))

        acc, auroc, f1 = ee.fit(data_tensor, label_tensor,
                         val_data_tensor, val_label_tensor,
                         n_inner_pts=2, batch_size=batch_size,
                         num_epochs=num_epochs, training=False)
        acc_list.append(acc)  # 保存测试准确率
        auroc_list.append(auroc)
        f1_list.append(f1)
    mean_accuracy = np.mean(acc_list)
    mean_auroc = np.mean(auroc_list)
    mean_f1 = np.mean(f1_list)
    f1_std = np.std(f1_list)
    print("Mean accuracy:", mean_accuracy)
    acc_std = np.std(acc_list)
    print("std:", acc_std)
    print("Mean auroc:", mean_auroc)
    auroc_std = np.std(auroc_list)
    print("auroc std:", auroc_std)
    print("Mean f1:", mean_f1)
    print("f1 std:", f1_std)
    # 將結果寫入文字檔
    with open(f"EE_{Parameter.name}_results.txt", "w") as file:
        file.write(f"All Runs Acc = {mean_accuracy}\n")
        file.write(f"All Runs std = {acc_std}\n")
        file.write(f"All Runs AUROC = {mean_auroc}\n")
        file.write(f"All Runs AUROC std = {auroc_std}\n")
        file.write(f"All Runs F1 = {mean_f1}\n")
        file.write(f"All Runs F1 std = {f1_std}\n")
    print(f"結果已寫入 EE_{Parameter.name}_results.txt")
    print("-------------------------------------------------------------")




