import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.regression.quantile_regression import QuantReg
from torch.utils.data import DataLoader, TensorDataset
import math
import os
import tempfile
import shutil

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
TRAIN_YEARS = list(range(2003, 2019))  # 训练集：2004-2018
VAL_YEARS = list(range(2019, 2022))  # 验证集：2019-2021
TEST_YEARS = list(range(2022, 2025))  # 测试集：2022-2023
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 3
BATCH_SIZE = 16
EPOCHS = 1
SAVE_PATH = 'output/best_model.pth'
READ_PATH = 'output/0.84.pth'
OUTPUT_DIR = 'output'  # 相对路径


# NSE 计算函数
def nash_sutcliffe_efficiency(y_true, y_pred):
    """计算纳什-萨特克利夫效率系数 (NSE)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mean_observed = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - mean_observed) ** 2)
    if denominator == 0:
        return np.nan  # 避免除以零
    return 1 - (numerator / denominator)


# 数据加载与处理（3个月周期编码）
def load_and_split_data(file_path):
    try:
        variables = ["地下水供水量", "入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量", "地下水储量变化（相比去年）"]
        dfs = []
        for var in variables:
            df = pd.read_excel(file_path, sheet_name=var)
            if df.empty:
                raise ValueError(f"工作表 {var} 为空或未找到")

            df['年份'] = df['年份'].astype(int)
            df['月份'] = df['月份'].astype(int)
            df = df.set_index(['年份', '月份'])[[var]]
            dfs.append(df)
        full_df = pd.concat(dfs, axis=1).dropna()

        # 添加月份时间编码（3个月周期）
        months = full_df.index.get_level_values('月份')
        full_df['month_sin'] = np.sin(2 * np.pi * (months - 1) / 3)
        full_df['month_cos'] = np.cos(2 * np.pi * (months - 1) / 3)

        # 对特征进行对数变换以稳定方差
        for feature in ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量"]:
            full_df[feature] = np.log1p(full_df[feature])

        print("合并后的 full_df[target] 前几行：")
        print(full_df["地下水储量变化（相比去年）"].head())
        print("月份时间编码示例（3个月周期）：")
        print(full_df[['month_sin', 'month_cos']].head())

        train_mask = full_df.index.get_level_values('年份').isin(TRAIN_YEARS)
        val_mask = full_df.index.get_level_values('年份').isin(VAL_YEARS)
        test_mask = full_df.index.get_level_values('年份').isin(TEST_YEARS)
        test_df_subset = full_df[test_mask]
        print("分割后的 test_df[target] 前几行：")
        print(test_df_subset["地下水储量变化（相比去年）"].head())
        return full_df[train_mask], full_df[val_mask], test_df_subset
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        raise
    except Exception as e:
        print(f"加载数据错误：{str(e)}")
        raise


# 加载数据
train_df, val_df, test_df = load_and_split_data("副本-北京市水资源公报数据（按类别）（月尺度）.xlsx")
print("原始 test_df：")
print(test_df)

# 特征配置
features = ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量", "month_sin", "month_cos"]
target = "地下水储量变化（相比去年）"

# 数据标准化
scaler_X = StandardScaler().fit(train_df[features].values)
scaler_y = StandardScaler().fit(train_df[[target]].values)
print("scaler_y 均值:", scaler_y.mean_)
print("scaler_y 标准差:", scaler_y.scale_)

# 验证 scaler_y 的逆变换
test_target = test_df[[target]].values
test_target_scaled = scaler_y.transform(test_target)
test_target_unscaled = scaler_y.inverse_transform(test_target_scaled)
print("原始 test_df[target] 与逆变换值对比：")
comparison_scaler = pd.DataFrame({
    '原始值': test_target.flatten(),
    '逆变换值': test_target_unscaled.flatten()
})
print(comparison_scaler.head(10))


def scale_dataset(df):
    return (
        scaler_X.transform(df[features].values),
        scaler_y.transform(df[[target]].values)
    )


X_train, y_train = scale_dataset(train_df)
X_val, y_val = scale_dataset(val_df)
X_test, y_test = scale_dataset(test_df)


# 转换为张量
def to_tensor(data):
    return torch.FloatTensor(data)


X_train_t = to_tensor(X_train)
y_train_t = to_tensor(y_train).squeeze()
X_val_t = to_tensor(X_val)
y_val_t = to_tensor(y_val).squeeze()
X_test_t = to_tensor(X_test)
y_test_t = to_tensor(y_test).squeeze()

# 创建DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)


# 模型定义
class PositionalEncoding(nn.Module):
    """保留原有的位置编码，用于为不同的水文特征注入维度索引先验"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class GLU(nn.Module):
    """Gated Linear Unit (门控线性单元)"""

    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        return torch.sigmoid(self.w1(x)) * self.w2(x)


class WaterTransformer(nn.Module):
    """
    Multi-Scale Gated Hydro-Fusion Network (MS-GHFN)
    适配单时间步特征输入的重构版本
    """

    def __init__(self, num_features, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # ==========================================
        # 模块 1: GVSN 门控变量选择网络
        # ==========================================
        self.input_proj = nn.Linear(1, d_model)
        self.glu = GLU(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=num_features)

        # ==========================================
        # 模块 2: 物理约束注意力机制 (映射为特征间因果矩阵)
        # ==========================================
        # 可训练的物理约束系数 \lambda
        self.lambda_phy = nn.Parameter(torch.tensor(0.1))
        # 特征间的物理连通性先验矩阵 \Phi，替代原有的时间滞后算子
        self.phi_matrix = nn.Parameter(torch.abs(torch.randn(num_features, num_features)))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=False, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # ==========================================
        # 模块 3: 双流物理融合策略 (Dual-Stream Physics Fusion)
        # ==========================================
        # 3a. 深层非线性残差流 F_deep
        self.deep_stream = nn.Sequential(
            nn.Linear(d_model * num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # 3b. 线性物理基线流 (替代原先写死的 influence_weights)
        self.physical_stream = nn.Linear(num_features, 1, bias=True)

        # 3c. 自适应“物理-数据”置信度辨别器 (\beta_t)
        self.beta_gate = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # ==========================================
        # 阶段 1: 门控特征表达
        # ==========================================
        # x shape: [batch_size, num_features]
        x_exp = x.unsqueeze(-1)  # [batch_size, num_features, 1]

        # 线性映射 \eta(x)
        eta_x = self.input_proj(x_exp)  # [batch_size, num_features, d_model]

        # 公式: \tilde{\chi} = LayerNorm(\chi + GLU_{\omega}(\eta(\chi)))
        glu_out = self.glu(eta_x)
        H = self.layer_norm(eta_x + glu_out)  # [batch_size, num_features, d_model]

        # ==========================================
        # 阶段 2: 物理约束注意力计算
        # ==========================================
        # 转换为 Transformer 期望的维度 [num_features, batch_size, d_model]
        H = H.permute(1, 0, 2)
        H = self.pos_encoder(H)

        # 构造注意力掩码: -\lambda * \Phi
        # PyTorch 的 Transformer 使用加性掩码，此操作在 softmax 前执行，实现了抑制噪声连接的物理约束
        physics_mask = -torch.abs(self.lambda_phy) * torch.abs(self.phi_matrix)

        H_encoded = self.transformer(H, mask=physics_mask)  # [num_features, batch_size, d_model]

        # 展平特征 [batch_size, num_features * d_model]
        H_encoded = H_encoded.permute(1, 0, 2)
        H_flat = H_encoded.reshape(batch_size, -1)

        # ==========================================
        # 阶段 3: 双流解码与动态平衡
        # ==========================================
        # 1. 深度网络输出 \mathcal{F}_{deep}(H_t)
        y_deep = self.deep_stream(H_flat)  # [batch_size, 1]

        # 2. 宏观水量平衡线性基线 W_{phy}^{\top}X_t + b_{phy}
        y_phy = self.physical_stream(x)  # [batch_size, 1]

        # 3. 计算实时融合系数 \beta_t \in [0, 1]
        beta_t = self.beta_gate(x)  # [batch_size, 1]

        # 4. 融合公式: \hat{y}_t = \beta_t * y_deep + (1 - \beta_t) * y_phy
        y_hat = beta_t * y_deep + (1.0 - beta_t) * y_phy

        return y_hat.squeeze()  # 返回 [batch_size] 形状以匹配原训练循环


# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WaterTransformer(
    num_features=len(features),
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS
).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 训练循环
best_val_loss = float('inf')
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Epoch {epoch + 1:03d} | 保存最佳模型 | 验证损失: {avg_val_loss:.4f}", flush=True)
    print(f"Epoch {epoch + 1:03d} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}", flush=True)


# 文件写入辅助函数
def ensure_file_writable(file_path):
    if os.path.exists(file_path):
        try:
            os.rename(file_path, file_path)
        except PermissionError:
            print(f"错误：文件 {file_path} 被占用，请关闭相关程序或暂停OneDrive同步")
            raise


# 可视化模块
def create_dates(df):
    return pd.to_datetime(
        df.index.get_level_values('年份').astype(str) + '-' +
        df.index.get_level_values('月份').astype(str) + '-01'
    )






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import shutil
import tempfile

def nash_sutcliffe_efficiency(observed, predicted):
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    return 1 - numerator / denominator if denominator != 0 else np.nan

def ensure_file_writable(filepath):
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except PermissionError:
            raise PermissionError(f"无法删除 {filepath}，请检查文件是否被占用或暂停OneDrive同步")

def create_dates(df):
    return pd.to_datetime(df.index.get_level_values('年份').astype(str) + '-' +
                         df.index.get_level_values('月份').astype(str))

def plot_time_series(loader, title, df, is_test=False, error_models=None, qr_models=None):
    """
    绘制时间序列预测结果，包括真实值、预测值、校正后预测值（如果适用）。
    使用双轴在一张图上展示地下水储量变化和所有特征趋势，特征使用密集虚线并优化可见性，移除置信区间，突出主数据，提升专业性。
    移除特征统计表格，将结果保存为图像和Excel文件，并返回误差模型（用于测试集校正）。
    改进：引入分级误差校正（正负各3个等级）和月份误差校正。

    参数：
        loader: DataLoader - 数据加载器，包含输入特征和目标值
        title: str - 图像标题（如“训练集预测效果”）
        df: pd.DataFrame - 输入数据，包含特征和目标值，索引为(年份, 月份)
        is_test: bool - 是否为测试集，影响校正的计算
        error_models: dict - 误差模型（正值和负值各3个等级 + 月份模型），用于校正测试集预测
        qr_models: dict - 分位数回归模型（未使用，保留兼容性）

    返回：
        error_models: dict - 训练的误差模型（仅验证集返回）
        qr_models: dict - 分位数回归模型（始终返回None，保留兼容性）
    """
    try:
        # 标题转换逻辑
        if title == "测试集预测效果":
            title = "Test Set Prediction Results"
        # 加载模型和数据
        model.load_state_dict(torch.load(READ_PATH))
        model.eval()
        dates = create_dates(df)
        X_raw = df[features].values
        y_true = df[[target]].values.flatten()
        months = df.index.get_level_values('月份').values

        # 进行预测
        with torch.no_grad():
            tensor_X = torch.FloatTensor(scaler_X.transform(X_raw)).to(device)
            preds = model(tensor_X).cpu().numpy()
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

        # 初始化误差模型和置信区间
        lower_bound = None
        upper_bound = None
        error_models_out = error_models if error_models else {}

        if title.startswith("验证集"):
            # 计算验证集的预测误差
            errors = y_true - preds
            mean_error = np.mean(errors)
            error_std = np.std(errors)
            nse_orig = nash_sutcliffe_efficiency(y_true, preds)
            print(f"验证集平均预测误差: {mean_error:.4f} 亿立方米", flush=True)
            print(f"验证集误差标准差: {error_std:.4f} 亿立方米", flush=True)
            print(f"验证集原始预测 NSE: {nse_orig:.4f}", flush=True)

            # 使用经验分位数计算置信区间（仅用于Excel输出）
            lower_quantile = np.percentile(errors, 5)
            upper_quantile = np.percentile(errors, 95)
            lower_bound = preds + lower_quantile
            upper_bound = preds + upper_quantile

            # 训练分级误差模型（正值和负值各3个等级）
            positive_mask = preds >= 0
            negative_mask = preds < 0
            error_models_out = {
                'positive_low': None, 'positive_mid': None, 'positive_high': None,
                'negative_low': None, 'negative_mid': None, 'negative_high': None,
                'month_models': {m: None for m in range(1, 13)}
            }
            error_stds = {
                'positive_low': 0, 'positive_mid': 0, 'positive_high': 0,
                'negative_low': 0, 'negative_mid': 0, 'negative_high': 0
            }

            # 正值分级
            if np.any(positive_mask):
                pos_preds = preds[positive_mask]
                pos_errors = errors[positive_mask]
                pos_X_raw = X_raw[positive_mask]
                pos_months = months[positive_mask]
                # 分位数划分：低、中、高
                quantiles = np.percentile(pos_preds, [33.33, 66.67])
                low_mask = pos_preds <= quantiles[0]
                mid_mask = (pos_preds > quantiles[0]) & (pos_preds <= quantiles[1])
                high_mask = pos_preds > quantiles[1]

                # 低值模型
                if np.any(low_mask):
                    X_error_pos = np.hstack([pos_preds[low_mask].reshape(-1, 1), pos_X_raw[low_mask]])
                    errors_pos = pos_errors[low_mask]
                    error_models_out['positive_low'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['positive_low'].fit(X_error_pos, errors_pos)
                    error_stds['positive_low'] = np.std(errors_pos) if len(errors_pos) > 1 else 0
                    print(f"正值低值样本数: {len(errors_pos)}, 误差标准差: {error_stds['positive_low']:.4f}", flush=True)

                # 中值模型
                if np.any(mid_mask):
                    X_error_pos = np.hstack([pos_preds[mid_mask].reshape(-1, 1), pos_X_raw[mid_mask]])
                    errors_pos = pos_errors[mid_mask]
                    error_models_out['positive_mid'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['positive_mid'].fit(X_error_pos, errors_pos)
                    error_stds['positive_mid'] = np.std(errors_pos) if len(errors_pos) > 1 else 0
                    print(f"正值中值样本数: {len(errors_pos)}, 误差标准差: {error_stds['positive_mid']:.4f}", flush=True)

                # 高值模型
                if np.any(high_mask):
                    X_error_pos = np.hstack([pos_preds[high_mask].reshape(-1, 1), pos_X_raw[high_mask]])
                    errors_pos = pos_errors[high_mask]
                    error_models_out['positive_high'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['positive_high'].fit(X_error_pos, errors_pos)
                    error_stds['positive_high'] = np.std(errors_pos) if len(errors_pos) > 1 else 0
                    print(f"正值高值样本数: {len(errors_pos)}, 误差标准差: {error_stds['positive_high']:.4f}", flush=True)

            # 负值分级
            if np.any(negative_mask):
                neg_preds = preds[negative_mask]
                neg_errors = errors[negative_mask]
                neg_X_raw = X_raw[negative_mask]
                neg_months = months[negative_mask]
                # 分位数划分：低、中、高（负值按绝对值大小划分）
                quantiles = np.percentile(np.abs(neg_preds), [33.33, 66.67])
                low_mask = np.abs(neg_preds) <= quantiles[0]
                mid_mask = (np.abs(neg_preds) > quantiles[0]) & (np.abs(neg_preds) <= quantiles[1])
                high_mask = np.abs(neg_preds) > quantiles[1]

                # 低值模型
                if np.any(low_mask):
                    X_error_neg = np.hstack([neg_preds[low_mask].reshape(-1, 1), neg_X_raw[low_mask]])
                    errors_neg = neg_errors[low_mask]
                    error_models_out['negative_low'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['negative_low'].fit(X_error_neg, errors_neg)
                    error_stds['negative_low'] = np.std(errors_neg) if len(errors_neg) > 1 else 0
                    print(f"负值低值样本数: {len(errors_neg)}, 误差标准差: {error_stds['negative_low']:.4f}", flush=True)

                # 中值模型
                if np.any(mid_mask):
                    X_error_neg = np.hstack([neg_preds[mid_mask].reshape(-1, 1), neg_X_raw[mid_mask]])
                    errors_neg = neg_errors[mid_mask]
                    error_models_out['negative_mid'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['negative_mid'].fit(X_error_neg, errors_neg)
                    error_stds['negative_mid'] = np.std(errors_neg) if len(errors_neg) > 1 else 0
                    print(f"负值中值样本数: {len(errors_neg)}, 误差标准差: {error_stds['negative_mid']:.4f}", flush=True)

                # 高值模型
                if np.any(high_mask):
                    X_error_neg = np.hstack([neg_preds[high_mask].reshape(-1, 1), neg_X_raw[high_mask]])
                    errors_neg = neg_errors[high_mask]
                    error_models_out['negative_high'] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['negative_high'].fit(X_error_neg, errors_neg)
                    error_stds['negative_high'] = np.std(errors_neg) if len(errors_neg) > 1 else 0
                    print(f"负值高值样本数: {len(errors_neg)}, 误差标准差: {error_stds['negative_high']:.4f}", flush=True)

            # 训练月份误差模型
            for month in range(1, 13):
                month_mask = months == month
                if np.any(month_mask):
                    X_error_month = np.hstack([preds[month_mask].reshape(-1, 1), X_raw[month_mask]])
                    errors_month = errors[month_mask]
                    error_models_out['month_models'][month] = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                    error_models_out['month_models'][month].fit(X_error_month, errors_month)
                    month_error_std = np.std(errors_month) if len(errors_month) > 1 else 0
                    print(f"月份 {month} 样本数: {len(errors_month)}, 误差标准差: {month_error_std:.4f}", flush=True)

            # 保存验证集结果到Excel
            result_df = pd.DataFrame({
                '年份': df.index.get_level_values('年份'),
                '月份': df.index.get_level_values('月份'),
                '真实值 (亿立方米)': y_true,
                '预测值 (亿立方米)': preds,
                '预测误差 (亿立方米)': errors,
                '95%置信下界': lower_bound,
                '95%置信上界': upper_bound
            })

            output_path = f'{OUTPUT_DIR}/validation_results.xlsx'
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            ensure_file_writable(output_path)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
            try:
                with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                    result_df.to_excel(writer, sheet_name='Validation_Results', index=False)
                    stats_df = pd.DataFrame({
                        '指标': ['平均误差', '误差标准差', '正值低值误差标准差', '正值中值误差标准差',
                                '正值高值误差标准差', '负值低值误差标准差', '负值中值误差标准差',
                                '负值高值误差标准差', '原始NSE'],
                        '值': [mean_error, error_std, error_stds['positive_low'], error_stds['positive_mid'],
                              error_stds['positive_high'], error_stds['negative_low'], error_stds['negative_mid'],
                              error_stds['negative_high'], nse_orig]
                    })
                    stats_df.to_excel(writer, sheet_name='Validation_Stats', index=False)
                shutil.move(temp_file, output_path)
                print(f"验证集结果已保存到 {output_path}", flush=True)
            except Exception as e:
                print(f"保存验证集 Excel 失败，临时文件: {temp_file}, 错误: {str(e)}")
                raise

        # 测试集：应用分级误差模型和月份误差模型进行校正
        corrected_preds = preds.copy()
        if is_test and error_models:
            positive_mask = preds >= 0
            negative_mask = preds < 0

            # 分级校正
            if 'positive_low' in error_models and np.any(positive_mask):
                pos_preds = preds[positive_mask]
                pos_X_raw = X_raw[positive_mask]
                quantiles = np.percentile(pos_preds, [33.33, 66.67]) if len(pos_preds) > 2 else [np.min(pos_preds), np.max(pos_preds)]
                low_mask = pos_preds <= quantiles[0]
                mid_mask = (pos_preds > quantiles[0]) & (pos_preds <= quantiles[1])
                high_mask = pos_preds > quantiles[1]

                if np.any(low_mask) and error_models['positive_low']:
                    X_error_pos = np.hstack([pos_preds[low_mask].reshape(-1, 1), pos_X_raw[low_mask]])
                    corrections_pos = error_models['positive_low'].predict(X_error_pos)
                    corrected_preds[positive_mask][low_mask] = pos_preds[low_mask] + corrections_pos

                if np.any(mid_mask) and error_models['positive_mid']:
                    X_error_pos = np.hstack([pos_preds[mid_mask].reshape(-1, 1), pos_X_raw[mid_mask]])
                    corrections_pos = error_models['positive_mid'].predict(X_error_pos)
                    corrected_preds[positive_mask][mid_mask] = pos_preds[mid_mask] + corrections_pos

                if np.any(high_mask) and error_models['positive_high']:
                    X_error_pos = np.hstack([pos_preds[high_mask].reshape(-1, 1), pos_X_raw[high_mask]])
                    corrections_pos = error_models['positive_high'].predict(X_error_pos)
                    corrected_preds[positive_mask][high_mask] = pos_preds[high_mask] + corrections_pos

            if 'negative_low' in error_models and np.any(negative_mask):
                neg_preds = preds[negative_mask]
                neg_X_raw = X_raw[negative_mask]
                quantiles = np.percentile(np.abs(neg_preds), [33.33, 66.67]) if len(neg_preds) > 2 else [np.min(np.abs(neg_preds)), np.max(np.abs(neg_preds))]
                low_mask = np.abs(neg_preds) <= quantiles[0]
                mid_mask = (np.abs(neg_preds) > quantiles[0]) & (np.abs(neg_preds) <= quantiles[1])
                high_mask = np.abs(neg_preds) > quantiles[1]

                if np.any(low_mask) and error_models['negative_low']:
                    X_error_neg = np.hstack([neg_preds[low_mask].reshape(-1, 1), neg_X_raw[low_mask]])
                    corrections_neg = error_models['negative_low'].predict(X_error_neg)
                    corrected_preds[negative_mask][low_mask] = neg_preds[low_mask] + corrections_neg

                if np.any(mid_mask) and error_models['negative_mid']:
                    X_error_neg = np.hstack([neg_preds[mid_mask].reshape(-1, 1), neg_X_raw[mid_mask]])
                    corrections_neg = error_models['negative_mid'].predict(X_error_neg)
                    corrected_preds[negative_mask][mid_mask] = neg_preds[mid_mask] + corrections_neg

                if np.any(high_mask) and error_models['negative_high']:
                    X_error_neg = np.hstack([neg_preds[high_mask].reshape(-1, 1), neg_X_raw[high_mask]])
                    corrections_neg = error_models['negative_high'].predict(X_error_neg)
                    corrected_preds[negative_mask][high_mask] = neg_preds[high_mask] + corrections_neg

            # 月份校正
            month_corrected_preds = corrected_preds.copy()
            for month in range(1, 13):
                month_mask = months == month
                if np.any(month_mask) and error_models['month_models'][month]:
                    X_error_month = np.hstack([corrected_preds[month_mask].reshape(-1, 1), X_raw[month_mask]])
                    corrections_month = error_models['month_models'][month].predict(X_error_month)
                    month_corrected_preds[month_mask] = corrected_preds[month_mask] + corrections_month
            corrected_preds = month_corrected_preds

            # 使用验证集的误差分位数计算测试集的置信区间（仅用于Excel输出）
            with torch.no_grad():
                val_inputs = torch.FloatTensor(scaler_X.transform(val_df[features].values)).to(device)
                val_preds = model(val_inputs).cpu().numpy()
                val_preds = scaler_y.inverse_transform(val_preds.reshape(-1, 1)).flatten()
            errors_val = val_df[target].values.flatten() - val_preds
            lower_quantile = np.percentile(errors_val, 5)
            upper_quantile = np.percentile(errors_val, 95)
            lower_bound = corrected_preds + lower_quantile
            upper_bound = corrected_preds + upper_quantile

            # 计算评估指标
            rmse_orig = np.sqrt(mean_squared_error(y_true, preds))
            mae_orig = mean_absolute_error(y_true, preds)
            nse_orig = nash_sutcliffe_efficiency(y_true, preds)
            rmse_corr = np.sqrt(mean_squared_error(y_true, corrected_preds))
            mae_corr = mean_absolute_error(y_true, corrected_preds)
            nse_corr = nash_sutcliffe_efficiency(y_true, corrected_preds)

            pos_true = y_true[positive_mask]
            pos_preds = preds[positive_mask]
            pos_corrected = corrected_preds[positive_mask]
            neg_true = y_true[negative_mask]
            neg_preds = preds[negative_mask]
            neg_corrected = corrected_preds[negative_mask]
            rmse_orig_pos = np.sqrt(mean_squared_error(pos_true, pos_preds)) if len(pos_true) > 0 else 0
            mae_orig_pos = mean_absolute_error(pos_true, pos_preds) if len(pos_true) > 0 else 0
            nse_orig_pos = nash_sutcliffe_efficiency(pos_true, pos_preds) if len(pos_true) > 0 else np.nan
            rmse_corr_pos = np.sqrt(mean_squared_error(pos_true, pos_corrected)) if len(pos_true) > 0 else 0
            mae_corr_pos = mean_absolute_error(pos_true, pos_corrected) if len(pos_true) > 0 else 0
            nse_corr_pos = nash_sutcliffe_efficiency(pos_true, pos_corrected) if len(pos_true) > 0 else np.nan
            rmse_orig_neg = np.sqrt(mean_squared_error(neg_true, neg_preds)) if len(neg_true) > 0 else 0
            mae_orig_neg = mean_absolute_error(neg_true, neg_preds) if len(neg_true) > 0 else 0
            nse_orig_neg = nash_sutcliffe_efficiency(neg_true, neg_preds) if len(neg_true) > 0 else np.nan
            rmse_corr_neg = np.sqrt(mean_squared_error(neg_true, neg_corrected)) if len(neg_true) > 0 else 0
            mae_corr_neg = mean_absolute_error(neg_true, neg_corrected) if len(neg_true) > 0 else 0
            nse_corr_neg = nash_sutcliffe_efficiency(neg_true, neg_corrected) if len(neg_true) > 0 else np.nan

            print(f"测试集整体 - 原始预测 RMSE: {rmse_orig:.4f}, MAE: {mae_orig:.4f}, NSE: {nse_orig:.4f}", flush=True)
            print(f"测试集整体 - 校正后预测 RMSE: {rmse_corr:.4f}, MAE: {mae_corr:.4f}, NSE: {nse_corr:.4f}", flush=True)
            print(f"测试集正值 - 原始预测 RMSE: {rmse_orig_pos:.4f}, MAE: {mae_orig_pos:.4f}, NSE: {nse_orig_pos:.4f}", flush=True)
            print(f"测试集正值 - 校正后预测 RMSE: {rmse_corr_pos:.4f}, MAE: {mae_corr_pos:.4f}, NSE: {nse_corr_pos:.4f}", flush=True)
            print(f"测试集负值 - 原始预测 RMSE: {rmse_orig_neg:.4f}, MAE: {mae_orig_neg:.4f}, NSE: {nse_orig_neg:.4f}", flush=True)
            print(f"测试集负值 - 校正后预测 RMSE: {rmse_corr_neg:.4f}, MAE: {mae_corr_neg:.4f}, NSE: {nse_corr_neg:.4f}", flush=True)

            # 保存测试集结果到Excel
            test_result_df = pd.DataFrame({
                '年份': df.index.get_level_values('年份'),
                '月份': df.index.get_level_values('月份'),
                '真实值 (亿立方米)': y_true,
                '原始预测值 (亿立方米)': preds,
                '校正后预测值 (亿立方米)': corrected_preds,
                '95%置信下界': lower_bound,
                '95%置信上界': upper_bound
            })

            output_path = f'{OUTPUT_DIR}/test_results.xlsx'
            ensure_file_writable(output_path)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx').name
            try:
                with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                    test_result_df.to_excel(writer, sheet_name='Test_Results', index=False)
                    stats_df = pd.DataFrame({
                        '指标': ['整体原始RMSE', '整体原始MAE', '整体原始NSE',
                               '整体校正后RMSE', '整体校正后MAE', '整体校正后NSE',
                               '正值原始RMSE', '正值原始MAE', '正值原始NSE',
                               '正值校正后RMSE', '正值校正后MAE', '正值校正后NSE',
                               '负值原始RMSE', '负值原始MAE', '负值原始NSE',
                               '负值校正后RMSE', '负值校正后MAE', '负值校正后NSE'],
                        '值': [rmse_orig, mae_orig, nse_orig,
                              rmse_corr, mae_corr, nse_corr,
                              rmse_orig_pos, mae_orig_pos, nse_orig_pos,
                              rmse_corr_pos, mae_corr_pos, nse_corr_pos,
                              rmse_orig_neg, mae_orig_neg, nse_orig_neg,
                              rmse_corr_neg, mae_corr_neg, nse_corr_neg]
                    })
                    stats_df.to_excel(writer, sheet_name='Test_Stats', index=False)
                shutil.move(temp_file, output_path)
                print(f"测试集结果已保存到 {output_path}", flush=True)
            except Exception as e:
                print(f"保存测试集 Excel 失败，临时文件: {temp_file}, 错误: {str(e)}")
                raise

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib import font_manager

        # Set font to Arial for better English readability
        plt.rcParams['font.family'] = 'Arial'

        # 绘制时间序列图（仅主轴）
        fig, ax1 = plt.subplots(figsize=(18, 8), dpi=300)

        # 主轴（左）：地下水储量变化
        ax1.plot(dates, y_true, 'o-', markersize=8, label='True Values',
                 color='#1f77b4', linewidth=2.5, alpha=0.9)
        ax1.plot(dates, preds, 's--', markersize=7, label='Original Predictions',
                 color='#ff7f0e', linewidth=2.5, alpha=0.9)
        if is_test and error_models:
            ax1.plot(dates, corrected_preds, 'd-.', markersize=7,
                     label='Corrected Predictions', color='#2ca02c', linewidth=2.5, alpha=0.9)

        # 设置主轴
        ax1.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel(r'Groundwater Storage Change ($10^8 \, \mathrm{m}^3$)', fontsize=14,
                       fontweight='bold', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_title(f"{title} ({df.index[0][0]}-{df.index[-1][0]})",
                      fontsize=16, fontweight='bold', pad=20)

        # 设置时间轴格式
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gcf().autofmt_xdate()
        plt.xlim(dates[0] - pd.DateOffset(months=1),
                 dates[-1] + pd.DateOffset(months=1))

        # 添加图例（仅主轴）
        lines1, labels1 = ax1.get_legend_handles_labels()
        plt.legend(lines1, labels1,
                   loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3, fontsize=10, frameon=True, edgecolor='black')

        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{title}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

        # 返回误差模型（验证集）或None（训练集/测试集）
        if title.startswith("验证集"):
            return error_models_out, None
        return None, None
    except PermissionError as e:
        print(f"权限错误：无法写入 {OUTPUT_DIR}/validation_results.xlsx 或 {OUTPUT_DIR}/test_results.xlsx，请检查文件是否被占用或暂停OneDrive同步")
        raise
    except Exception as e:
        print(f"可视化错误：{str(e)}")
        raise


def plot_combined_time_series(train_loader, val_loader, test_loader, train_df, val_df, test_df):
    """
    绘制训练集、验证集和测试集的预测与实测地下水储量变化对比图，使用不同颜色区间区分。
    结果保存为高质量图像，包含统计表格。

    参数：
        train_loader, val_loader, test_loader: DataLoader - 数据加载器
        train_df, val_df, test_df: pd.DataFrame - 数据集，索引为(年份, 月份)
    """
    try:
        # 加载模型
        model.load_state_dict(torch.load(READ_PATH))
        model.eval()

        # 获取日期和真实值
        datasets = {'训练集': train_df, '验证集': val_df, '测试集': test_df}
        loaders = {'训练集': train_loader, '验证集': val_loader, '测试集': test_loader}
        all_dates = []
        all_y_true = []
        all_preds = []
        set_labels = []

        for set_name, df in datasets.items():
            dates = create_dates(df)
            X_raw = df[features].values
            y_true = df[[target]].values.flatten()

            # 进行预测
            with torch.no_grad():
                tensor_X = torch.FloatTensor(scaler_X.transform(X_raw)).to(device)
                preds = model(tensor_X).cpu().numpy()
            preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

            all_dates.extend(dates)
            all_y_true.extend(y_true)
            all_preds.extend(preds)
            set_labels.extend([set_name] * len(dates))

        # 创建DataFrame并按日期排序
        combined_df = pd.DataFrame({
            '日期': all_dates,
            '真实值': all_y_true,
            '预测值': all_preds,
            '数据集': set_labels
        })
        combined_df = combined_df.sort_values('日期').reset_index(drop=True)

        # 计算统计指标
        stats = {}
        for set_name in datasets.keys():
            mask = combined_df['数据集'] == set_name
            y_true_set = combined_df[mask]['真实值']
            y_pred_set = combined_df[mask]['预测值']
            rmse = np.sqrt(mean_squared_error(y_true_set, y_pred_set))
            mae = mean_absolute_error(y_true_set, y_pred_set)
            nse = nash_sutcliffe_efficiency(y_true_set, y_pred_set)
            stats[set_name] = {'RMSE': rmse, 'MAE': mae, 'NSE': nse}

        # 创建高级图形
        fig, ax = plt.subplots(figsize=(22, 10), dpi=300)

        # 绘制真实值和预测值
        ax.plot(combined_df['日期'], combined_df['真实值'], 'o-', markersize=6,
                label='真实值', color='#1f77b4', linewidth=2, alpha=0.9)
        ax.plot(combined_df['日期'], combined_df['预测值'], 's--', markersize=5,
                label='预测值', color='#ff7f0e', linewidth=2, alpha=0.9)

        # 添加数据集区间背景
        colors = {'训练集': '#e6f3ff', '验证集': '#fff7e6', '测试集': '#e6ffe6'}
        date_ranges = {
            '训练集': (pd.to_datetime(f"{TRAIN_YEARS[0]}-01-01"),
                    pd.to_datetime(f"{TRAIN_YEARS[-1] + 1}-01-01")),
            '验证集': (pd.to_datetime(f"{VAL_YEARS[0]}-01-01"),
                    pd.to_datetime(f"{VAL_YEARS[-1] + 1}-01-01")),
            '测试集': (pd.to_datetime(f"{TEST_YEARS[0]}-01-01"),
                    pd.to_datetime(f"{TEST_YEARS[-1] + 1}-01-01"))
        }

        for set_name, (start, end) in date_ranges.items():
            ax.axvspan(start, end, facecolor=colors[set_name], alpha=0.3,
                       label=f'{set_name}区间')

        # 设置坐标轴
        ax.set_xlabel('时间', fontsize=14, fontweight='bold')
        ax.set_ylabel('地下水储量变化 (亿立方米)', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

        # 设置时间轴格式
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()

        # 添加标题
        plt.title('地下水储量变化预测与实测对比 (2003-2023)', fontsize=16,
                  pad=20, fontweight='bold')

        # 创建统计表格
        cell_text = []
        for set_name in ['训练集', '验证集', '测试集']:
            cell_text.append([
                set_name,
                f"{stats[set_name]['RMSE']:.4f}",
                f"{stats[set_name]['MAE']:.4f}",
                f"{stats[set_name]['NSE']:.4f}"
            ])

        table = plt.table(cellText=cell_text,
                          colLabels=['数据集', 'RMSE', 'MAE', 'NSE'],
                          loc='bottom',
                          bbox=[0.15, -0.35, 0.7, 0.2],
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#F0F0F0')
                cell.set_text_props(weight='bold')

        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper left',
                           bbox_to_anchor=(0.01, 1.12), ncol=5,
                           fontsize=10, frameon=True, edgecolor='black')
        legend.get_frame().set_linewidth(1.5)

        # 调整布局
        plt.subplots_adjust(bottom=0.3)
        plt.tight_layout()

        # 保存图像
        output_path = f'{OUTPUT_DIR}/combined_time_series.png'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"综合时间序列图已保存到 {output_path}", flush=True)

    except Exception as e:
        print(f"综合时间序列可视化错误：{str(e)}")
        raise


# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='训练损失')
plt.plot(val_loss_history, label='验证损失')
plt.title('训练过程损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/loss_curve.png')
plt.close()

# 可视化结果
try:
    error_models, qr_models = plot_time_series(train_loader, "训练集预测效果", train_df)
    error_models, qr_models = plot_time_series(val_loader, "验证集预测效果", val_df)
    plot_time_series(test_loader, "测试集预测效果", test_df, is_test=True, error_models=error_models, qr_models=qr_models)
except Exception as e:
    print(f"可视化过程中错误：{str(e)}")
    raise

plot_combined_time_series(train_loader, val_loader, test_loader, train_df, val_df, test_df)
# 预测接口
def predict(input_dict):
    if '月份' not in input_dict:
        raise ValueError("输入字典必须包含 '月份' 字段（1-12）")
    month = input_dict['月份']
    month_sin = np.sin(2 * np.pi * (month - 1) / 3)
    month_cos = np.cos(2 * np.pi * (month - 1) / 3)
    # 对特征进行对数变换
    input_dict_transformed = input_dict.copy()
    for feature in ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量"]:
        input_dict_transformed[feature] = np.log1p(input_dict_transformed[feature])
    input_array = np.array(
        [input_dict_transformed[feature] for feature in features[:-2]] + [month_sin, month_cos]).reshape(1, -1)
    scaled_input = scaler_X.transform(input_array)
    model.load_state_dict(torch.load(READ_PATH))
    model.eval()
    with torch.no_grad():
        tensor_input = torch.FloatTensor(scaled_input).to(device)
        prediction = model(tensor_input).cpu().numpy()
    return scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]


# 示例预测
sample_input = {
    "入境水量": 3.2,
    "全市平均降水量": 450,
    "出境水量": 10,
    "生活用水量": 10.0,
    "农业用水量": 10.0,
    "月份": 6
}
try:
    print(f"\n预测结果: {predict(sample_input):.2f} 亿立方米", flush=True)
except Exception as e:
    print(f"预测错误：{str(e)}")





def perform_sensitivity_analysis(model, train_df, scaler_X, scaler_y, device, output_dir=OUTPUT_DIR):
    """
    执行单因子敏感性分析（OAT），评估每个输入特征对模型输出的影响。
    对每个特征在实际数据范围内施加扰动，计算预测输出的相对变化，生成敏感性指数和可视化。

    参数：
        model: WaterTransformer - 训练好的模型
        train_df: pd.DataFrame - 训练数据集，包含特征和目标值
        scaler_X: StandardScaler - 特征标准化器
        scaler_y: StandardScaler - 目标标准化器
        device: torch.device - 计算设备
        output_dir: str - 输出目录

    返回：
        sensitivity_df: pd.DataFrame - 敏感性分析结果
    """
    try:
        # 加载模型
        model.load_state_dict(torch.load(READ_PATH))
        model.eval()

        # 获取特征的实际范围（基于训练数据）
        feature_stats = {}
        for feature in features:
            if feature in ["month_sin", "month_cos"]:
                # 周期特征固定范围 [-1, 1]
                feature_stats[feature] = {'min': -1.0, 'max': 1.0, 'mean': 0.0}
            else:
                raw_values = train_df[feature].values
                # 逆对数变换以获取原始尺度
                original_values = np.expm1(raw_values) if feature not in ["month_sin", "month_cos"] else raw_values
                feature_stats[feature] = {
                    'min': original_values.min(),
                    'max': original_values.max(),
                    'mean': original_values.mean()
                }

        # 定义扰动水平（基于原始尺度）
        perturbation_levels = [-0.2, -0.1, 0.0, 0.1, 0.2]  # ±20%, ±10%, 0%

        # 创建基准输入（使用训练数据的均值）
        base_input = {}
        for feature in features[:-2]:  # 排除 month_sin, month_cos
            base_input[feature] = feature_stats[feature]['mean']
        base_input['月份'] = 6  # 假设6月作为基准月份
        month = base_input['月份']
        base_input['month_sin'] = np.sin(2 * np.pi * (month - 1) / 3)
        base_input['month_cos'] = np.cos(2 * np.pi * (month - 1) / 3)

        # 计算基准预测
        def predict_single(input_dict):
            input_dict_transformed = input_dict.copy()
            for feature in ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量"]:
                input_dict_transformed[feature] = np.log1p(input_dict_transformed[feature])
            input_array = np.array(
                [input_dict_transformed[f] for f in features[:-2]] +
                [input_dict_transformed['month_sin'], input_dict_transformed['month_cos']]
            ).reshape(1, -1)
            scaled_input = scaler_X.transform(input_array)
            with torch.no_grad():
                tensor_input = torch.FloatTensor(scaled_input).to(device)
                prediction = model(tensor_input).cpu().numpy()
            return scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]

        base_pred = predict_single(base_input)

        # 敏感性分析
        sensitivity_results = []
        for feature in features:
            # 跳过 month_sin, month_cos（周期特征影响较复杂，单独分析）
            if feature in ["month_sin", "month_cos"]:
                continue
            feature_results = {'特征': feature}
            mean_abs_change = 0
            changes = []
            for level in perturbation_levels:
                perturbed_input = base_input.copy()
                base_value = feature_stats[feature]['mean']
                # 施加扰动（原始尺度）
                perturbed_value = base_value * (1 + level)
                # 限制在实际范围内
                perturbed_value = np.clip(
                    perturbed_value,
                    feature_stats[feature]['min'],
                    feature_stats[feature]['max']
                )
                perturbed_input[feature] = perturbed_value
                perturbed_pred = predict_single(perturbed_input)
                # 计算相对变化
                rel_change = (perturbed_pred - base_pred) / abs(base_pred) if base_pred != 0 else 0
                feature_results[f'扰动_{level*100:+.0f}%'] = perturbed_pred
                feature_results[f'相对变化_{level*100:+.0f}%'] = rel_change
                changes.append(abs(rel_change))
            # 计算敏感性指数（平均绝对相对变化）
            sensitivity_index = np.mean(changes)
            feature_results['敏感性指数'] = sensitivity_index
            sensitivity_results.append(feature_results)

            # 可视化敏感性曲线
            plt.figure(figsize=(10, 6))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            levels = [level * 100 for level in perturbation_levels]
            preds = [feature_results[f'扰动_{level:+.0f}%'] for level in levels]
            plt.plot(levels, preds, marker='o', linewidth=2, color='#1f77b4')
            plt.axhline(base_pred, color='gray', linestyle='--', alpha=0.5, label='基准预测')
            plt.xlabel('扰动水平 (%)', fontsize=12, fontweight='bold')
            plt.ylabel('预测值 (亿立方米)', fontsize=12, fontweight='bold')
            plt.title(f'特征 "{feature}" 的敏感性分析', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=10)
            plt.tight_layout()
            output_path = f'{output_dir}/sensitivity_{feature}.png'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        # 保存敏感性结果到Excel
        sensitivity_df = pd.DataFrame(sensitivity_results)
        output_path = f'{output_dir}/sensitivity_analysis.xlsx'
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                sensitivity_df.to_excel(writer, sheet_name='Sensitivity_Results', index=False)
            print(f"敏感性分析结果已保存到 {output_path}")
        except Exception as e:
            print(f"保存敏感性分析 Excel 失败，错误: {str(e)}")
            raise

        # 打印敏感性分析结果
        print("\n敏感性分析结果：")
        for _, row in sensitivity_df.iterrows():
            feature = row['特征']
            index = row['敏感性指数']
            expected_sign = "正向" if feature in ["入境水量", "全市平均降水量"] else \
                           "负向" if feature in ["出境水量", "生活用水量", "农业用水量"] else "未知"
            actual_sign = "正向" if row['相对变化_+10%'] > 0 else "负向"
            print(f"{feature}: 敏感性指数={index:.4f}, 影响方向={actual_sign} (预期: {expected_sign})")

        return sensitivity_df

    except Exception as e:
        print(f"敏感性分析错误：{str(e)}")
        raise

# 调用敏感性分析（在主代码末尾添加）
try:
    sensitivity_df = perform_sensitivity_analysis(model, train_df, scaler_X, scaler_y, device)
except Exception as e:
    print(f"执行敏感性分析失败：{str(e)}")



# 加载训练好的模型并预测设计值数据
print("开始预测设计值数据...")

# 配置参数
DESIGN_EXCEL_PATH = '设计数据表-40-new.xlsx'
OUTPUT_DIR = 'output'


# 数据加载与处理（支持“年份”和“情景”两种索引）
def load_design_data(file_path):
    try:
        variables = ["地下水供水量", "入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量", "地下水储量变化（相比去年）"]
        required_features = ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量"]  # 模型所需的特征
        dfs = []

        xl = pd.ExcelFile(file_path)
        available_sheets = xl.sheet_names
        print(f"Available sheets in design file: {available_sheets}")

        loaded_vars = []
        for var in variables:
            if var not in available_sheets:
                print(f"Warning: Sheet '{var}' not found in design file, skipping")
                continue
            df = pd.read_excel(file_path, sheet_name=var)
            if df.empty:
                print(f"Warning: Sheet '{var}' is empty, skipping")
                continue

            first_column = df.columns[0]
            if first_column == "年份":
                df['年份'] = df['年份'].astype(int)
                df['月份'] = df['月份'].astype(int)
                df = df.set_index(['年份', '月份'])[[var]]
            elif first_column == "情景":
                df['月份'] = df['月份'].astype(int)
                df = df.set_index(['情景', '月份'])[[var]]
            else:
                raise ValueError(f"Sheet {var}'s first column must be '年份' or '情景', but found {first_column}")
            dfs.append(df)
            loaded_vars.append(var)

        if not dfs:
            raise ValueError("No valid sheets loaded from design file")

        full_df = pd.concat(dfs, axis=1).dropna()

        missing_features = [feat for feat in required_features if feat not in full_df.columns]
        if missing_features:
            raise ValueError(f"Design data missing required features: {missing_features}")

        months = full_df.index.get_level_values('月份')
        full_df['month_sin'] = np.sin(2 * np.pi * (months - 1) / 3)
        full_df['month_cos'] = np.cos(2 * np.pi * (months - 1) / 3)

        for feature in ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量"]:
            if feature in full_df.columns:
                full_df[feature] = np.log1p(full_df[feature])

        print("Loaded design data sample:")
        print(full_df.head())
        return full_df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        raise
    except Exception as e:
        print(f"Error loading design data: {str(e)}")
        raise


# 数据预处理（标准化）
def preprocess_data(df, scaler_X, scaler_y):
    X = scaler_X.transform(df[features].values)
    y = scaler_y.transform(df[[target]].values) if target in df.columns else None
    return torch.FloatTensor(X), y, df


# 加载训练数据以拟合标准化器
train_df, val_df, test_df = load_and_split_data("副本-北京市水资源公报数据（按类别）（月尺度）.xlsx")
scaler_X = StandardScaler().fit(train_df[features].values)
scaler_y = StandardScaler().fit(train_df[[target]].values)
print("Scalers generated")
print("scaler_X mean:", scaler_X.mean_)
print("scaler_X scale:", scaler_X.scale_)
print("scaler_y mean:", scaler_y.mean_)
print("scaler_y scale:", scaler_y.scale_)

# 加载设计值数据
design_df = load_design_data(DESIGN_EXCEL_PATH)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WaterTransformer(
    num_features=len(features),
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS
).to(device)
try:
    model.load_state_dict(torch.load(READ_PATH))
    print(f"Model loaded from {READ_PATH}")
except FileNotFoundError:
    print(f"Error: Model file {READ_PATH} not found")
    raise
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# 误差纠正：使用 QuantReg 拟合验证集和测试集的预测误差
val_test_df = pd.concat([val_df, test_df])
X_val_test, y_val_test = scale_dataset(val_test_df)
X_val_test_t = to_tensor(X_val_test).to(device)
y_val_test_t = to_tensor(y_val_test).squeeze()
model.eval()
with torch.no_grad():
    val_test_preds = model(X_val_test_t).cpu().numpy()
val_test_preds = scaler_y.inverse_transform(val_test_preds.reshape(-1, 1)).flatten()
val_test_true = scaler_y.inverse_transform(y_val_test).flatten()
val_test_errors = val_test_true - val_test_preds

from statsmodels.regression.quantile_regression import QuantReg

error_df = pd.DataFrame({
    'error': val_test_errors,
    'pred': val_test_preds
})
quantile_low = QuantReg(error_df['error'], error_df[['pred']]).fit(q=0.1)
quantile_high = QuantReg(error_df['error'], error_df[['pred']]).fit(q=0.9)
print("QuantReg model (based on validation and test sets) fitted")


# 预测函数（包含基于合并误差的置信区间）
def predict_data(df, model, scaler_X, scaler_y, device):
    model.eval()
    X_tensor, y_true, df = preprocess_data(df, scaler_X, scaler_y)
    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

    error_pred_df = pd.DataFrame({'pred': preds})
    interval_low = quantile_low.predict(error_pred_df[['pred']])
    interval_high = quantile_high.predict(error_pred_df[['pred']])
    lower_bound = preds + interval_low
    upper_bound = preds + interval_high

    index_level_0 = df.index.get_level_values(0)
    index_level_1 = df.index.get_level_values('月份')
    labels = [f"{idx0}-M{idx1:02d}" for idx0, idx1 in zip(index_level_0, index_level_1)]

    result_df = pd.DataFrame({
        'Index': labels,
        'Prediction (100 million m³)': preds,
        'Lower Bound (100 million m³)': lower_bound,
        'Upper Bound (100 million m³)': upper_bound
    })
    if y_true is not None:
        y_true = scaler_y.inverse_transform(y_true).flatten()
        result_df['True Value (100 million m³)'] = y_true
    return result_df, labels


# 可视化函数（高端设计，确保显示9条情景曲线，支持英文，恢复背景颜色）
# 可视化函数（高端设计，确保显示9条情景曲线，支持英文，恢复背景颜色）
def plot_predictions(result_df, title):
    import matplotlib.font_manager as fm
    # 适配新版 Matplotlib 的字体缓存重建方法
    try:
        fm.fontManager._load_fontmanager(try_read_cache=False)
    except AttributeError:
        pass

    # 移除 seaborn 样式，避免干扰背景颜色
    plt.style.use('default')

    # 使用 Arial 字体支持英文
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建图表，明确设置背景颜色
    fig, ax = plt.subplots(figsize=(18, 9), facecolor='#f0f0f0')

    # 定义中文到英文的情景名称映射
    scenario_mapping = {
        '情景1': 'Scenario1',
        '情景2': 'Scenario2',
        '情景3': 'Scenario3',
        '情景4': 'Scenario4',
        '情景5': 'Scenario5',
        '情景6': 'Scenario6',
        '情景7': 'Scenario7',
        '情景8': 'Scenario8',
        '情景9': 'Scenario9',
        # 根据你的 Excel 文件添加更多映射
    }

    # 提取所有情景
    scenarios = result_df.index.get_level_values(0).unique() if isinstance(result_df.index, pd.MultiIndex) else \
    result_df['Index'].str.split('-M').str[0].unique()
    months = range(1, 13)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(scenarios)))

    # 为每个情景分别绘制曲线和阴影
    for i, scenario in enumerate(scenarios):
        # 按情景提取数据
        if isinstance(result_df.index, pd.MultiIndex):
            scenario_data = result_df[result_df.index.get_level_values(0) == scenario]
        else:
            scenario_data = result_df[result_df['Index'].str.startswith(f"{scenario}-")]

        # 提取月份、预测值和置信区间
        month_values = [int(label.split('-M')[1]) for label in scenario_data['Index']]
        pred_values = scenario_data['Prediction (100 million m³)'].values
        lower_bound = scenario_data['Lower Bound (100 million m³)'].values
        upper_bound = scenario_data['Upper Bound (100 million m³)'].values

        # 构建完整月份数据
        scenario_dict = {m: (float('nan'), float('nan'), float('nan')) for m in months}
        for m, p, l, u in zip(month_values, pred_values, lower_bound, upper_bound):
            scenario_dict[m] = (p, l, u)

        # 提取绘图数据
        plot_months = []
        plot_preds = []
        plot_lower = []
        plot_upper = []
        for m in months:
            p, l, u = scenario_dict[m]
            if not np.isnan(p):  # 只绘制非 NaN 的数据
                plot_months.append(m)
                plot_preds.append(p)
                plot_lower.append(l)
                plot_upper.append(u)

        # 使用映射后的英文名称作为图例标签
        scenario_label = scenario_mapping.get(scenario, scenario)  # 如果没有映射，使用原名称
        ax.plot(plot_months, plot_preds, marker='o', linewidth=2.5, color=colors[i], label=f'{scenario_label}', zorder=5)
        ax.fill_between(plot_months, plot_lower, plot_upper, color=colors[i], alpha=0.2, zorder=3)

    ax.set_xlabel('Month', fontsize=20, fontweight='bold', color='#333333', labelpad=15)
    ax.set_ylabel('Change in Groundwater Storage (100 million m³)', fontsize=20, fontweight='bold', color='#333333', labelpad=15)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    ax.set_xticks(months)
    ax.tick_params(axis='both', which='major', labelsize=20, colors='#444444')
    # 恢复绘图区域的背景颜色
    ax.set_facecolor('#f9f9f9')

    ax.legend(title='Scenario', title_fontsize=20, fontsize=18, frameon=True, facecolor='#ffffff', edgecolor='#cccccc',
              fancybox=True, loc='best')

    ax.set_title(f'{title}', fontsize=24, fontweight='bold', color='#222222', pad=20)

    plt.tight_layout()
    output_path = f'{OUTPUT_DIR}/water_volume_prediction_results.png'
    # 明确指定背景颜色，保存时确保一致
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f0f0f0')
    plt.close(fig)
    print(f"Prediction plot saved to {output_path}")


# 进行预测（包含基于合并误差的置信区间）
result_df, labels = predict_data(design_df, model, scaler_X, scaler_y, device)
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = f'{OUTPUT_DIR}/design_predictions_with_intervals.xlsx'
result_df.to_excel(output_path, index=False)
print(f"Prediction results saved to {output_path}")

# 可视化预测结果（包含9条曲线，高端设计）
plot_predictions(result_df, "Design Scenario Prediction Results and Error Range (Outbound Water Volume 4 billion m³)")

# 单条预测示例
sample_input = {
    "入境水量": 3.2,
    "全市平均降水量": 450,
    "出境水量": 10,
    "生活用水量": 10.0,
    "农业用水量": 10.0,
    "月份": 6
}


def predict_single(input_dict, model, scaler_X, scaler_y, device):
    if '月份' not in input_dict:
        raise ValueError("输入字典必须包含 '月份' 字段（1-12）")
    month = input_dict['月份']
    month_sin = np.sin(2 * np.pi * (month - 1) / 3)
    month_cos = np.cos(2 * np.pi * (month - 1) / 3)
    input_dict_transformed = input_dict.copy()
    for feature in ["入境水量", "全市平均降水量", "出境水量", "生活用水量", "农业用水量"]:
        input_dict_transformed[feature] = np.log1p(input_dict_transformed[feature])
    input_array = np.array(
        [input_dict_transformed[feature] for feature in features[:-2]] + [month_sin, month_cos]).reshape(1, -1)
    scaled_input = scaler_X.transform(input_array)
    model.eval()
    with torch.no_grad():
        tensor_input = torch.FloatTensor(scaled_input).to(device)
        prediction = model(tensor_input).cpu().numpy()
    pred = scaler_y.inverse_transform(prediction.reshape(-1, 1))[0][0]
    error_pred_df = pd.DataFrame({'pred': [pred]})
    interval_low = quantile_low.predict(error_pred_df[['pred']])[0]
    interval_high = quantile_high.predict(error_pred_df[['pred']])[0]
    lower_bound = pred + interval_low
    upper_bound = pred + interval_high
    return pred, lower_bound, upper_bound


try:
    prediction, lower_bound, upper_bound = predict_single(sample_input, model, scaler_X, scaler_y, device)
    print(f"单条预测结果: {prediction:.2f} 亿立方米 (置信区间: [{lower_bound:.2f}, {upper_bound:.2f}])")
except Exception as e:
    print(f"单条预测失败：{str(e)}")


# 检查特征影响
def check_feature_influence(model, X_train, features, scaler_X, scaler_y, device):
    model.eval()
    X_check = torch.FloatTensor(X_train).to(device)
    with torch.no_grad():
        preds_check = model(X_check).cpu().numpy()
    feature_influences = {}
    for i, feature in enumerate(features):
        X_perturbed = X_check.clone()
        perturbation = 0.01 * X_check[:, i].std()
        X_perturbed[:, i] += perturbation
        with torch.no_grad():
            preds_perturbed = model(X_perturbed).cpu().numpy()
        influence = (preds_perturbed - preds_check).mean() / perturbation
        feature_influences[feature] = influence
    print("特征影响方向：")
    for feature, influence in feature_influences.items():
        if feature in ["入境水量", "全市平均降水量"]:
            expected_sign = "正向"
        elif feature in ["出境水量", "生活用水量", "农业用水量"]:
            expected_sign = "负向"
        else:
            expected_sign = "未知"
        actual_sign = "正向" if influence > 0 else "负向"
        print(f"{feature}: {actual_sign} (预期: {expected_sign}, 影响值: {influence:.4f})")


try:
    X_train = scaler_X.transform(design_df[features].values)
    check_feature_influence(model, X_train, features, scaler_X, scaler_y, device)
except Exception as e:
    print(f"特征影响检查失败：{str(e)}")

print("设计值预测完成")
