from kmeans import KMeans
from ext import gen_data

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering")

        tk.Label(root, text="Количество точек:").grid(row=0, column=0)
        self.num_points_entry = tk.Entry(root)
        self.num_points_entry.grid(row=0, column=1)
        self.num_points_entry.insert(0, "300")  # Значение по умолчанию

        tk.Label(root, text="Количество кластеров:").grid(row=1, column=0)
        self.num_clusters_entry = tk.Entry(root)
        self.num_clusters_entry.grid(row=1, column=1)
        self.num_clusters_entry.insert(0, "3")

        tk.Label(root, text="Количество эпох:").grid(row=2, column=0)
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.grid(row=2, column=1)
        self.epochs_entry.insert(0, "10")

        # Кнопка запуска
        self.run_button = tk.Button(root, text="Запустить", command=self.run_kmeans)
        self.run_button.grid(row=3, column=0, columnspan=2)

        # Поля для графиков
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))  # 2 графика в 1 окне
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2)

    def run_kmeans(self):
        try:
            self.axs[0].clear()
            self.axs[1].clear()

            num_points = int(self.num_points_entry.get())
            num_clusters = int(self.num_clusters_entry.get())
            epochs = int(self.epochs_entry.get())

            if num_points <= 0 or num_clusters <= 0 or epochs <= 0:
                raise ValueError("Все значения должны быть больше 0")

            # Генерация данных
            data = gen_data(num_points)

            kmeans = KMeans(2, num_clusters)

            # Начальная инициализация центроидов
            kmeans.init_centers(data)
            # График 1: начальная расстановка центроидов
            self.axs[0].scatter(data[:, 0], data[:, 1], s=10, alpha=0.7)
            self.axs[0].scatter(np.array(kmeans.centers)[:, 0], np.array(kmeans.centers)[:, 1], s=300, c='red', marker='X')
            self.axs[0].set_title('Первые центроиды')

            clusters, ans_labels = kmeans.fit(data, epochs)
            # График 2: финальный результат
            self.axs[1].scatter(data[:, 0], data[:, 1], c=ans_labels, cmap='viridis')
            self.axs[1].scatter(np.array(kmeans.centers)[:, 0], np.array(kmeans.centers)[:, 1], s=300, c='red', marker='X')
            self.axs[1].set_title('Результаты кластеризации KMeans')

            self.canvas.draw()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный ввод: {e}")