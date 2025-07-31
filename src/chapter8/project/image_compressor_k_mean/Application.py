import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import os
import threading
import time


class ImageCompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像压缩可视化工具")
        self.root.geometry("1050x800")
        self.root.minsize(900, 700)

        # 数据初始化
        self.original_image = None
        self.compressed_image = None
        self.image_path = None
        self.compression_thread = None
        self.compression_progress = 0  # 跟踪压缩进度百分比

        # 创建界面组件
        self._create_widgets()

    def _create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 图像对比区域（上部）
        image_frame = ttk.LabelFrame(main_frame, text="图像对比", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 原始图像区域
        original_frame = ttk.Frame(image_frame)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(original_frame, text="压缩前").pack(anchor=tk.W)
        self.original_canvas = tk.Canvas(original_frame, bg="#f0f0f0", relief=tk.SUNKEN, bd=1)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        self.original_info = ttk.Label(original_frame, text="未加载图像", font=("Arial", 8))
        self.original_info.pack(anchor=tk.W, pady=(5, 0))

        # 压缩后图像区域
        compressed_frame = ttk.Frame(image_frame)
        compressed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(compressed_frame, text="压缩后").pack(anchor=tk.W)
        self.compressed_canvas = tk.Canvas(compressed_frame, bg="#f0f0f0", relief=tk.SUNKEN, bd=1)
        self.compressed_canvas.pack(fill=tk.BOTH, expand=True)
        self.compressed_info = ttk.Label(compressed_frame, text="未压缩图像", font=("Arial", 8))
        self.compressed_info.pack(anchor=tk.W, pady=(5, 0))

        # 2. 控制区域（中部）
        control_frame = ttk.LabelFrame(main_frame, text="压缩设置", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 快捷选项区域
        preset_frame = ttk.LabelFrame(control_frame, text="快捷选项")
        preset_frame.pack(side=tk.LEFT, padx=(0, 15), pady=5)

        ttk.Button(preset_frame, text="快速", command=lambda: self._apply_preset(16, 25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="普通", command=lambda: self._apply_preset(32, 50)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="高质量", command=lambda: self._apply_preset(64, 100)).pack(side=tk.LEFT, padx=2)

        # 加载图像按钮
        ttk.Button(control_frame, text="加载图像", command=self.load_image).pack(side=tk.LEFT, padx=(0, 10))

        # 保存图像按钮
        self.save_button = ttk.Button(control_frame, text="保存压缩图像", command=self.save_compressed_image,
                                      state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))

        # 颜色数量选择（自定义）
        color_frame = ttk.Frame(control_frame)
        color_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(color_frame, text="颜色数量:").pack(side=tk.LEFT)
        self.num_colors = tk.IntVar(value=16)
        ttk.Entry(color_frame, textvariable=self.num_colors, width=5).pack(side=tk.LEFT)

        # 最大迭代次数选择
        iter_frame = ttk.Frame(control_frame)
        iter_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(iter_frame, text="最大迭代次数:").pack(side=tk.LEFT)
        self.max_iterations = tk.IntVar(value=50)
        ttk.Entry(iter_frame, textvariable=self.max_iterations, width=5).pack(side=tk.LEFT)

        # 算法选择
        algo_frame = ttk.Frame(control_frame)
        algo_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(algo_frame, text="算法:").pack(side=tk.LEFT)
        self.algorithm = tk.StringVar(value="mean")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm,
                                  values=["mean", "max"], width=5)
        algo_combo.pack(side=tk.LEFT)
        algo_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)

        # Max算法子选项
        self.max_frame = ttk.Frame(control_frame)
        ttk.Label(self.max_frame, text="模式:").pack(side=tk.LEFT)
        self.max_mode = tk.StringVar(value="avg")
        ttk.Combobox(self.max_frame, textvariable=self.max_mode,
                     values=["avg", "lu", "sa", "mix"], width=5).pack(side=tk.LEFT)

        # 混合模式权重
        self.mix_frame = ttk.Frame(control_frame)
        ttk.Label(self.mix_frame, text="亮度权重:").pack(side=tk.LEFT)
        self.lum_weight = tk.DoubleVar(value=0.5)
        ttk.Scale(self.mix_frame, variable=self.lum_weight, from_=0, to=1,
                  orient=tk.HORIZONTAL, length=100, command=self._update_mix_label).pack(side=tk.LEFT, padx=5)
        self.mix_label = ttk.Label(self.mix_frame, text="0.5", width=5)
        self.mix_label.pack(side=tk.LEFT)

        # 压缩按钮
        self.compress_button = ttk.Button(control_frame, text="开始压缩", command=self.start_compression)
        self.compress_button.pack(side=tk.RIGHT, padx=(10, 0))

        # 初始隐藏max相关选项
        self._on_algorithm_change(None)

        # 3. 进度区域
        progress_frame = ttk.LabelFrame(main_frame, text="压缩进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        self.progress_label = ttk.Label(progress_frame, text="等待开始...")
        self.progress_label.pack(anchor=tk.W)

        # 4. 报告区域
        report_frame = ttk.LabelFrame(main_frame, text="压缩报告", padding="10")
        report_frame.pack(fill=tk.BOTH, expand=True)

        self.report_text = tk.Text(report_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.report_text.pack(fill=tk.BOTH, expand=True)

        # 绑定画布大小变化事件
        self.original_canvas.bind("<Configure>", lambda e: self._display_original_image())
        self.compressed_canvas.bind("<Configure>", lambda e: self._display_compressed_image())

    def _apply_preset(self, num_colors, max_iter):
        """应用快捷选项"""
        self.num_colors.set(num_colors)
        self.max_iterations.set(max_iter)
        self.algorithm.set("mean")  # 快捷选项默认使用mean算法
        self._on_algorithm_change(None)

    def _on_algorithm_change(self, event):
        """算法选择变化时显示/隐藏相关选项"""
        if self.algorithm.get() == "max":
            self.max_frame.pack(side=tk.LEFT, padx=(0, 10))
            self._update_mix_visibility()
        else:
            self.max_frame.pack_forget()
            self.mix_frame.pack_forget()

    def _update_mix_visibility(self):
        """更新混合模式选项的可见性"""
        if self.max_mode.get() == "mix":
            self.mix_frame.pack(side=tk.LEFT, padx=(0, 10))
        else:
            self.mix_frame.pack_forget()

    def _update_mix_label(self, value):
        """更新混合模式权重标签"""
        self.mix_label.config(text=f"{float(value):.1f}")

    def load_image(self):
        """加载图像文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )

        if not file_path:
            return

        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert("RGB")
            self._display_original_image()

            # 重置压缩图像和报告
            self.compressed_image = None
            self.compressed_canvas.delete("all")
            self.compressed_info.config(text="未压缩图像")
            self.report_text.config(state=tk.NORMAL)
            self.report_text.delete(1.0, tk.END)
            self.report_text.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def save_compressed_image(self):
        """保存压缩后的图像"""
        if not self.compressed_image:
            messagebox.showwarning("警告", "没有可保存的压缩图像")
            return

        # 获取原始图像的扩展名
        original_ext = os.path.splitext(self.image_path)[1].lower() if self.image_path else ".png"
        if original_ext not in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
            original_ext = ".png"

        # 询问保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=original_ext,
            filetypes=[
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg;*.jpeg"),
                ("BMP图像", "*.bmp"),
                ("GIF图像", "*.gif"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            try:
                self.compressed_image.save(file_path)
                messagebox.showinfo("成功", f"图像已保存至:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存图像失败: {str(e)}")

    def _display_original_image(self):
        """在画布上显示原始图像"""
        if not self.original_image:
            return

        # 获取画布尺寸
        canvas_width = self.original_canvas.winfo_width() or 400
        canvas_height = self.original_canvas.winfo_height() or 300

        # 调整图像大小以适应画布
        img = self.original_image.copy()
        img.thumbnail((canvas_width, canvas_height))

        # 显示图像
        self.original_canvas.delete("all")
        photo = ImageTk.PhotoImage(img)
        self.original_photo = photo  # 保持引用
        self.original_canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER
        )

        # 更新图像信息
        width, height = self.original_image.size
        size_kb = os.path.getsize(self.image_path) / 1024
        self.original_info.config(
            text=f"尺寸: {width}x{height}  大小: {size_kb:.2f} KB"
        )

    def _display_compressed_image(self):
        """在画布上显示压缩后的图像"""
        if not self.compressed_image:
            return

        # 获取画布尺寸
        canvas_width = self.compressed_canvas.winfo_width() or 400
        canvas_height = self.compressed_canvas.winfo_height() or 300

        # 调整图像大小以适应画布
        img = self.compressed_image.copy()
        img.thumbnail((canvas_width, canvas_height))

        # 显示图像
        self.compressed_canvas.delete("all")
        photo = ImageTk.PhotoImage(img)
        self.compressed_photo = photo  # 保持引用
        self.compressed_canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER
        )

        # 更新图像信息
        width, height = self.compressed_image.size
        self.compressed_info.config(text=f"尺寸: {width}x{height}")

    def start_compression(self):
        """开始压缩过程（在新线程中运行以避免界面冻结）"""
        # 验证输入参数
        try:
            num_colors = self.num_colors.get()
            max_iter = self.max_iterations.get()

            if num_colors < 2 or num_colors > 256:
                messagebox.showwarning("参数错误", "颜色数量应在2-256之间")
                return

            if max_iter < 1 or max_iter > 1000:
                messagebox.showwarning("参数错误", "最大迭代次数应在1-1000之间")
                return
        except:
            messagebox.showwarning("参数错误", "请输入有效的数字")
            return

        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图像")
            return

        if self.compression_thread and self.compression_thread.is_alive():
            messagebox.showinfo("提示", "压缩正在进行中...")
            return

        # 禁用按钮并重置进度条
        self.compress_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="准备压缩...")
        self.compression_progress = 0

        # 打印开始信息到控制台
        print("\n" + "=" * 50)
        print(f"开始图像压缩 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"颜色数量: {self.num_colors.get()}")
        print(f"最大迭代次数: {self.max_iterations.get()}")
        print(f"算法: {self.algorithm.get()}")
        if self.algorithm.get() == "max":
            print(f"模式: {self.max_mode.get()}")
            if self.max_mode.get() == "mix":
                print(f"亮度权重: {self.lum_weight.get():.1f}")
        print("=" * 50)

        # 启动压缩线程
        self.compression_thread = threading.Thread(target=self._perform_compression)
        self.compression_thread.daemon = True
        self.compression_thread.start()

        # 定期检查压缩进度
        self.root.after(100, self._check_compression_progress)

    def _check_compression_progress(self):
        """检查压缩进度并更新UI"""
        if self.compression_thread and self.compression_thread.is_alive():
            self.root.after(100, self._check_compression_progress)
        else:
            self.progress_label.config(text="压缩完成!")
            self.compress_button.config(state=tk.NORMAL)
            if self.compressed_image:
                self.save_button.config(state=tk.NORMAL)

    def _update_progress(self, value, message):
        """更新进度条和标签"""
        self.compression_progress = value
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(text=message))
        print(f"进度: {value:.1f}% - {message}")

    def _perform_compression(self):
        """执行图像压缩（在后台线程中运行）"""
        try:
            # 获取参数
            k = self.num_colors.get()
            max_iter = self.max_iterations.get()

            # 更新进度
            self._update_progress(5, "准备图像数据...")

            # 准备图像数据
            image_np = np.array(self.original_image)
            height, width, channels = image_np.shape
            pixels = image_np.reshape(-1, channels).astype(np.float32) / 255.0

            # 更新进度
            self._update_progress(10, "初始化K-means聚类...")

            # 计算进度步长
            progress_range = 50  # 10%到60%之间的进度范围
            progress_step = progress_range / max_iter

            # 分阶段运行K-means以跟踪进度
            # 第一次运行获取初始聚类中心
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=1)
            kmeans.fit(pixels)
            self._update_progress(10 + progress_step, f"K-means迭代 1/{max_iter}")

            # 逐步迭代并更新进度
            for i in range(1, max_iter):
                # 使用上一次的聚类中心继续迭代
                kmeans = KMeans(n_clusters=k, init=kmeans.cluster_centers_,
                                random_state=42, n_init=1, max_iter=1)
                kmeans.fit(pixels)

                # 更新进度
                current_progress = 10 + (i + 1) * progress_step
                self._update_progress(current_progress, f"K-means迭代 {i + 1}/{max_iter}")

                # 检查是否已收敛
                if kmeans.n_iter_ < 1:  # 如果本次迭代没有实际更新，说明已收敛
                    break

            # 获取最终结果
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            actual_iter = kmeans.n_iter_ + 1  # 加上第一次迭代

            # 更新进度至60%
            self._update_progress(60, "K-means聚类完成")

            # 根据选择的算法确定替换颜色
            algorithm = self.algorithm.get()
            max_mode = self.max_mode.get() if algorithm == "max" else None
            lum_weight = self.lum_weight.get() if max_mode == "mix" else 0.5

            if algorithm == "mean":
                # 使用聚类中心（平均值）
                compressed_pixels = centroids[labels]
                self._update_progress(80, "使用均值算法生成图像...")
            else:  # max算法
                # 为每个聚类找到最合适的代表像素
                compressed_pixels = np.zeros_like(pixels)
                total_clusters = k
                for i in range(k):
                    # 更新进度
                    cluster_progress = 60 + 20 * (i + 1) / total_clusters
                    self._update_progress(cluster_progress, f"处理聚类 {i + 1}/{total_clusters}...")

                    # 获取当前聚类的所有像素
                    cluster_mask = labels == i
                    cluster_pixels = pixels[cluster_mask]

                    if len(cluster_pixels) == 0:
                        # 空聚类使用中心
                        selected_pixel = centroids[i]
                    else:
                        # 根据不同模式选择像素
                        if max_mode == "avg":  # RGB平均值
                            scores = cluster_pixels.mean(axis=1)
                        elif max_mode == "lu":  # 亮度
                            scores = 0.299 * cluster_pixels[:, 0] + 0.587 * cluster_pixels[:,
                                                                            1] + 0.114 * cluster_pixels[:, 2]
                        elif max_mode == "sa":  # 饱和度
                            max_vals = np.max(cluster_pixels, axis=1)
                            min_vals = np.min(cluster_pixels, axis=1)
                            delta = max_vals - min_vals
                            scores = np.where(max_vals == 0, 0, delta / max_vals)
                        else:  # 混合模式
                            # 亮度
                            lum = 0.299 * cluster_pixels[:, 0] + 0.587 * cluster_pixels[:, 1] + 0.114 * cluster_pixels[
                                                                                                        :, 2]
                            # 饱和度
                            max_vals = np.max(cluster_pixels, axis=1)
                            min_vals = np.min(cluster_pixels, axis=1)
                            delta = max_vals - min_vals
                            sat = np.where(max_vals == 0, 0, delta / max_vals)
                            # 归一化
                            lum_norm = (lum - lum.min()) / (lum.max() - lum.min() + 1e-9)
                            sat_norm = (sat - sat.min()) / (sat.max() - sat.min() + 1e-9)
                            # 加权
                            scores = lum_weight * lum_norm + (1 - lum_weight) * sat_norm

                        # 选择得分最高的像素
                        selected_idx = np.argmax(scores)
                        selected_pixel = cluster_pixels[selected_idx]

                    compressed_pixels[cluster_mask] = selected_pixel

            # 更新进度
            self._update_progress(85, "生成压缩图像...")

            # 转换回图像格式
            compressed_pixels = (compressed_pixels * 255).astype(np.uint8)
            compressed_image_np = compressed_pixels.reshape(height, width, channels)
            self.compressed_image = Image.fromarray(compressed_image_np)

            # 更新进度
            self._update_progress(90, "计算图像大小...")

            # 保存临时文件用于计算大小
            temp_path = "temp_compressed.jpg"
            self.compressed_image.save(temp_path)
            compressed_size_kb = os.path.getsize(temp_path) / 1024
            os.remove(temp_path)

            # 更新进度
            self._update_progress(100, "压缩完成!")
            print(f"压缩完成 - {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 更新UI
            self.root.after(0, self._display_compressed_image)
            self.root.after(0, lambda: self._update_report(compressed_size_kb, max_iter, actual_iter))

        except Exception as e:
            error_msg = f"压缩过程中发生错误: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("压缩错误", error_msg))
            self.root.after(0, lambda: self.progress_label.config(text="压缩失败"))
            self.root.after(0, lambda: self.compress_button.config(state=tk.NORMAL))
            print(f"错误: {error_msg}")

    def _update_report(self, compressed_size_kb, max_iter, actual_iter):
        """更新压缩报告"""
        if not self.image_path:
            return

        original_size_kb = os.path.getsize(self.image_path) / 1024
        # 压缩率计算：压缩后/压缩前的百分比
        compression_ratio = (compressed_size_kb / original_size_kb) * 100

        algorithm = self.algorithm.get()
        if algorithm == "mean":
            algo_name = "均值聚类"
        else:
            mode_map = {
                "avg": "RGB平均值最大",
                "lu": "亮度最大",
                "sa": "饱和度最大",
                "mix": f"亮度-饱和度混合 (亮度权重: {self.lum_weight.get():.1f})"
            }
            algo_name = f"最大值模式 ({mode_map[self.max_mode.get()]})"

        report = [
            f"压缩算法: {algo_name}",
            f"颜色数量: {self.num_colors.get()}",
            f"最大迭代次数: {max_iter}",
            f"原始大小: {original_size_kb:.2f} KB",
            f"压缩后大小: {compressed_size_kb:.2f} KB",
            f"压缩率: {compression_ratio:.2f}% (压缩后/压缩前)"
        ]

        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, "\n".join(report))
        self.report_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressionApp(root)
    root.mainloop()
