# #5月22日对该代码图像文字进行修改，若出现问题，请回溯历史代码。
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, expon, gamma, lognorm, weibull_min
# from copulae import GaussianCopula, StudentCopula, ClaytonCopula, FrankCopula, GumbelCopula
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd
# from matplotlib.lines import Line2D  # Added import for Line2D
#
# # Set global font parameters for larger text
# plt.rcParams.update({
#     'font.size': 14,              # Base font size
#     'axes.titlesize': 16,         # Title font size
#     'axes.labelsize': 14,         # Axis label font size
#     'xtick.labelsize': 12,        # X-axis tick label font size
#     'ytick.labelsize': 12,        # Y-axis tick label font size
#     'legend.fontsize': 12,        # Legend font size
#     'font.sans-serif': ['SimHei'], # Keep SimHei for Chinese characters
#     'axes.unicode_minus': False    # Handle minus sign for Chinese fonts
# })
#
# def fit_best_distribution(data):
#     distributions = [
#         {'name': 'Normal', 'dist': norm, 'fix_params': {}, 'k': 2},
#         {'name': 'Exponential', 'dist': expon, 'fix_params': {'floc': 0}, 'k': 1},
#         {'name': 'Gamma', 'dist': gamma, 'fix_params': {'floc': 0}, 'k': 2},
#         {'name': 'LogNormal', 'dist': lognorm, 'fix_params': {'floc': 0}, 'k': 2},
#         {'name': 'Weibull', 'dist': weibull_min, 'fix_params': {'floc': 0}, 'k': 2},
#     ]
#     results = []
#     for d in distributions:
#         dist = d['dist']
#         name = d['name']
#         fix_params = d['fix_params']
#         k = d['k']
#         try:
#             params = dist.fit(data, **fix_params)
#             log_likelihood = np.sum(dist.logpdf(data, *params))
#             aic = 2 * k - 2 * log_likelihood
#             results.append({'name': name, 'dist': dist, 'params': params, 'aic': aic, 'k': k})
#         except Exception as e:
#             continue
#     results.sort(key=lambda x: x['aic'])
#     best = results[0]
#     return best['name'], best['dist'], best['params'], best['aic']
#
# def plot_all_distributions(data, dist_list, title):
#     plt.figure(figsize=(12, 8))  # Slightly increased figure size
#     x = np.linspace(min(data), max(data), 200)
#
#     # Plot histogram
#     plt.hist(data, bins=15, density=True, alpha=0.3, color='gray', label='观测数据')
#
#     # Plot all distribution curves
#     for dist_info in dist_list:
#         dist = dist_info['dist']
#         params = dist_info['params']
#         label = f"{dist_info['name']} (AIC={dist_info['aic']:.1f})"
#         plt.plot(x, dist.pdf(x, *params), label=label)
#
#     plt.xlabel('观测值', fontsize=14)
#     plt.ylabel('指标值', fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True)
#     # plt.show()
#
# def plot_pdf_fit(data, dist_name, dist, params, aic, name):
#     n = len(data)
#     sorted_data = np.sort(data)
#     p_empirical = (np.arange(n) + 0.5) / n
#     p_theoretical = np.linspace(0.01, 0.99, 100)
#     quantiles_theoretical = dist.ppf(p_theoretical, *params)
#
#     plt.figure(figsize=(12, 8))  # Slightly increased figure size
#     plt.scatter(p_empirical, sorted_data, color='blue', alpha=0.6, label='观测值')
#     plt.plot(p_theoretical, quantiles_theoretical, 'r-', lw=2, label=f'Fitted {dist_name}')
#     plt.xlabel('概率', fontsize=14)
#     plt.ylabel('值', fontsize=14)
#     plt.title(f'{name}边缘分布拟合效果图: {dist_name} Fit (AIC={aic:.2f})', fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True)
#     # plt.show()
#
# if __name__ == "__main__":
#     # Data
#     x = np.array([5.29, 2.6, 4.18, 6.32, 4.59, 4.25, 3.45, 5.35, 3.03, 6.93, 4.71,
#                   8.62, 10.75, 3.59, 4.49, 7.13, 5.03, 8.19, 6.08, 6.61, 10.01, 9.03, 7.92,6.19])
#     y = np.array([462, 413, 453, 539, 468, 448, 499, 638, 448, 524, 552, 708, 501,
#                   439, 583, 660, 592, 590, 506, 560, 924, 482, 727,782])
#
#     def get_all_distributions(data):
#         distributions = [
#             {'name': 'Normal', 'dist': norm, 'fix_params': {}, 'k': 2},
#             {'name': 'Exponential', 'dist': expon, 'fix_params': {'floc': 0}, 'k': 1},
#             {'name': 'Gamma', 'dist': gamma, 'fix_params': {'floc': 0}, 'k': 2},
#             {'name': 'LogNormal', 'dist': lognorm, 'fix_params': {'floc': 0}, 'k': 2},
#             {'name': 'Weibull', 'dist': weibull_min, 'fix_params': {'floc': 0}, 'k': 2},
#         ]
#         results = []
#         for d in distributions:
#             dist = d['dist']
#             name = d['name']
#             fix_params = d['fix_params']
#             k = d['k']
#             try:
#                 params = dist.fit(data, **fix_params)
#                 log_likelihood = np.sum(dist.logpdf(data, *params))
#                 aic = 2 * k - 2 * log_likelihood
#                 results.append({'name': name, 'dist': dist, 'params': params, 'aic': aic})
#             except Exception as e:
#                 continue
#         results.sort(key=lambda x: x['aic'])
#         return results
#
#     # Get all distribution results
#     x_dists = get_all_distributions(x)
#     y_dists = get_all_distributions(y)
#
#     # Extract best distribution
#     dist_name_x, dist_x, params_x, aic_x = x_dists[0]['name'], x_dists[0]['dist'], x_dists[0]['params'], x_dists[0]['aic']
#     dist_name_y, dist_y, params_y, aic_y = y_dists[0]['name'], y_dists[0]['dist'], y_dists[0]['params'], y_dists[0]['aic']
#
#     # Plot all distributions
#     plot_all_distributions(x, x_dists, '入境水量边缘分布拟合效果对比图')
#     plot_all_distributions(y, y_dists, '降水量边缘分布拟合效果对比图')
#
#     print(f"X最优边缘分布: {dist_name_x} (AIC={aic_x:.2f})")
#     print(f"Y最优边缘分布: {dist_name_y} (AIC={aic_y:.2f})")
#     plot_pdf_fit(x, dist_name_x, dist_x, params_x, aic_x, '入境水量')
#     plot_pdf_fit(y, dist_name_y, dist_y, params_y, aic_y, '年平均降水量')
#
#     # Copula analysis
#     u = dist_x.cdf(x, *params_x)
#     v = dist_y.cdf(y, *params_y)
#     data_copula = np.column_stack((u, v))
#
#     copula_types = [
#         ('Gaussian', GaussianCopula),
#         ('Clayton', ClaytonCopula),
#         ('Frank', FrankCopula),
#         ('Gumbel', GumbelCopula),
#     ]
#
#     copula_results = []
#     for name, copula_class in copula_types:
#         try:
#             cop = copula_class(dim=2)
#             cop.fit(data_copula)
#             log_lik = cop.log_lik(data_copula)
#             params = cop.params
#             k = len(params) if isinstance(params, (list, np.ndarray)) else 1
#             aic = -2 * log_lik + 2 * k
#             copula_results.append({'name': name, 'aic': aic, 'copula': cop})
#             print(f"{name} Copula - AIC: {aic:.2f}")
#         except Exception as e:
#             print(f"Failed to fit {name}: {e}")
#
#     copula_results.sort(key=lambda x: x['aic'])
#     print("\nCopula Ranking by AIC:")
#     for i, res in enumerate(copula_results):
#         print(f"{i + 1}. {res['name']}: AIC={res['aic']:.2f}")
#
#     # 3D Copula visualization
#     grid_size = 30
#     x_grid = np.linspace(min(x), max(x), grid_size)
#     y_grid = np.linspace(min(y), max(y), grid_size)
#     x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
#
#     u_mesh = dist_x.cdf(x_mesh, *params_x)
#     v_mesh = dist_y.cdf(y_mesh, *params_y)
#     grid_points = np.column_stack((u_mesh.ravel(), v_mesh.ravel()))
#
#     for res in copula_results:
#         name = res['name']
#         cop = res['copula']
#         try:
#             if hasattr(cop, 'cdf'):
#                 cdf = cop.cdf(grid_points).reshape(x_mesh.shape)
#             else:
#                 from scipy.integrate import dblquad
#                 cdf = np.zeros_like(x_mesh)
#                 for i in range(x_mesh.shape[0]):
#                     for j in range(x_mesh.shape[1]):
#                         result, _ = dblquad(
#                             lambda u, v: cop.pdf([[u, v]]),
#                             0, u_mesh[i, j],
#                             lambda x: 0, lambda x: v_mesh[i, j]
#                         )
#                         cdf[i, j] = result
#                 print(f"[Warning] {name} CDF calculated by integration")
#
#             fig = plt.figure(figsize=(12, 8))  # Slightly increased figure size
#             ax = fig.add_subplot(111, projection='3d')
#             surf = ax.plot_surface(x_mesh, y_mesh, cdf, cmap=cm.coolwarm, alpha=0.8, rstride=1, cstride=1)
#             ax.scatter(x, y, cop.cdf(data_copula), c='red', s=30, alpha=0.9, label='观测值')
#             ax.set_xlabel('入境水量（亿立方米）', fontsize=14)
#             ax.set_ylabel('年平均降水量（mm）', fontsize=14)
#             ax.set_zlabel('C(u,v)', fontsize=14)
#             ax.set_zlim(0, 1)
#             ax.set_title(f'{name} Copula Function\n(入境水量边缘分布：{dist_name_x}, 年平均降水量边缘分布：{dist_name_y})', fontsize=16)
#             plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#             plt.legend(fontsize=12)
#             plt.tight_layout()
#             # plt.show()
#
#         except Exception as e:
#             print(f"Plot error for {name}: {e}")
#
# def calculate_category_probabilities(x_data, y_data, copula, x_dist, x_params, y_dist, y_params):
#     q_x = [x_dist.ppf(0.375, *x_params), x_dist.ppf(0.625, *x_params)]
#     q_y = [y_dist.ppf(0.375, *y_params), y_dist.ppf(0.625, *y_params)]
#
#     categories = ['枯', '平', '丰']
#     theory_probs = pd.DataFrame(index=categories, columns=categories)
#     actual_probs = pd.DataFrame(index=categories, columns=categories)
#
#     for i, x_cat in enumerate(categories):
#         for j, y_cat in enumerate(categories):
#             x_low = -np.inf if i == 0 else q_x[0] if i == 1 else q_x[1]
#             x_high = q_x[0] if i == 0 else q_x[1] if i == 1 else np.inf
#             y_low = -np.inf if j == 0 else q_y[0] if j == 1 else q_y[1]
#             y_high = q_y[0] if j == 0 else q_y[1] if j == 1 else np.inf
#
#             u_low = x_dist.cdf(x_low, *x_params)
#             u_high = x_dist.cdf(x_high, *x_params)
#             v_low = y_dist.cdf(y_low, *y_params)
#             v_high = y_dist.cdf(y_high, *y_params)
#             theory_prob = copula.cdf([u_high, v_high]) - copula.cdf([u_low, v_high]) \
#                           - copula.cdf([u_high, v_low]) + copula.cdf([u_low, v_low])
#             theory_probs.loc[x_cat, y_cat] = theory_prob
#             mask = (x_data > x_low) & (x_data <= x_high) & (y_data > y_low) & (y_data <= y_high)
#             actual_probs.loc[x_cat, y_cat] = np.mean(mask)
#
#     return theory_probs.astype(float), actual_probs.astype(float)
#
# # Use best Copula
# best_cop = copula_results[0]['copula']
# theory_probs, actual_probs = calculate_category_probabilities(x, y, best_cop, dist_x, params_x, dist_y, params_y)
#
# print("\n理论概率矩阵（最优Copula）:")
# print(theory_probs.applymap(lambda x: f"{x:.2%}"))
# print("\n实际观测概率:")
# print(actual_probs.applymap(lambda x: f"{x:.2%}"))
#
# # Scatter plot with categories
# plt.figure(figsize=(12, 8))  # Slightly increased figure size
# categories = ['枯', '平', '丰']
# colors = ['#1f77b4', '#2ca02c', '#d62728']
#
# q_x1 = dist_x.ppf(0.375, *params_x)
# q_x2 = dist_x.ppf(0.625, *params_x)
# q_y1 = dist_y.ppf(0.375, *params_y)
# q_y2 = dist_y.ppf(0.625, *params_y)
#
# plt.axvline(q_x1, color='gray', linestyle='--', alpha=0.7)
# plt.axvline(q_x2, color='gray', linestyle='--', alpha=0.7)
# plt.axhline(q_y1, color='gray', linestyle='--', alpha=0.7)
# plt.axhline(q_y2, color='gray', linestyle='--', alpha=0.7)
#
# for i in range(len(x)):
#     x_cat = 0 if x[i] <= q_x1 else 1 if x[i] <= q_x2 else 2
#     y_cat = 0 if y[i] <= q_y1 else 1 if y[i] <= q_y2 else 2
#     plt.scatter(x[i], y[i], color=colors[x_cat],
#                 marker='o' if y_cat == 0 else 's' if y_cat == 1 else '^',
#                 alpha=0.7, edgecolor='w')
#
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label='Y=枯', markerfacecolor='gray', markersize=10),
#     Line2D([0], [0], marker='s', color='w', label='Y=平', markerfacecolor='gray', markersize=10),
#     Line2D([0], [0], marker='^', color='w', label='Y=丰', markerfacecolor='gray', markersize=10),
#     Line2D([0], [0], color=colors[0], lw=4, label='X=枯'),
#     Line2D([0], [0], color=colors[1], lw=4, label='X=平'),
#     Line2D([0], [0], color=colors[2], lw=4, label='X=丰')
# ]
# plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
# plt.xlabel('入境水量（亿立方米）', fontsize=14)
# plt.ylabel('年平均降水量（mm）', fontsize=14)
# plt.title('观测值在9个区间的分布 (虚线: 37.5%和62.5%分位数)', fontsize=16)
# plt.grid(True, alpha=0.3)
# # plt.show()
#
# def plot_copula_qq(data_copula, copula, copula_name):
#     cdf_values = copula.cdf(data_copula)
#     empirical_quantiles = np.sort(cdf_values)
#     n = len(cdf_values)
#     theoretical_quantiles = (np.arange(1, n + 1) - 0.5) / n
#
#     plt.figure(figsize=(10, 8))  # Slightly increased figure size
#     plt.scatter(theoretical_quantiles, empirical_quantiles, color='blue', alpha=0.6, label='Q-Q点')
#     plt.plot([0, 1], [0, 1], 'r--', lw=2, label='y=x参考线')
#     plt.xlabel('理论分位数 (均匀分布)', fontsize=14)
#     plt.ylabel('经验分位数 (Clayton Copula CDF)', fontsize=14)
#     plt.title(f'{copula_name} Copula Q-Q图', fontsize=16)
#     plt.legend(fontsize=12)
#     plt.grid(True)
#     plt.tight_layout()
#     # plt.show()
#
# # Clayton Copula Q-Q plot
# clayton_cop = None
# for res in copula_results:
#     if res['name'] == 'Clayton':
#         clayton_cop = res['copula']
#         break
#
# if clayton_cop is not None:
#     plot_copula_qq(data_copula, clayton_cop, 'Clayton')
# else:
#     print("Clayton Copula未找到，可能拟合失败。")
#
# # Exceedance CDF plot
# dist_x = gamma
# params_x = gamma.fit(x, floc=0)
# dist_y = lognorm
# params_y = lognorm.fit(y, floc=0)
#
# u = dist_x.cdf(x, *params_x)
# v = dist_y.cdf(y, *params_y)
# data_copula = np.column_stack((u, v))
# copula = ClaytonCopula(dim=2)
# copula.fit(data_copula)
#
# categories = ['枯', '平', '丰']
# quantiles = {'枯': 0.05, '平': 0.5, '丰': 0.95}
# x_values = {cat: dist_x.ppf(quantiles[cat], *params_x) for cat in categories}
# y_values = {cat: dist_y.ppf(quantiles[cat], *params_y) for cat in categories}
#
# joint_probs = {}
# for x_cat in categories:
#     for y_cat in categories:
#         u_val = dist_x.cdf(x_values[x_cat], *params_x)
#         v_val = dist_y.cdf(y_values[y_cat], *params_y)
#         joint_prob = copula.cdf([u_val, v_val])
#         joint_probs[(x_cat, y_cat)] = {
#             'X': x_values[x_cat],
#             'Y': y_values[y_cat],
#             'Joint Probability': joint_prob
#         }
#
# print("\nRepresentative (X, Y) Combinations and Joint Probabilities (Clayton Copula):")
# print("-" * 60)
# for x_cat in categories:
#     for y_cat in categories:
#         data = joint_probs[(x_cat, y_cat)]
#         print(f"X={x_cat}, Y={y_cat}:")
#         print(f"  X value = {data['X']:.2f} 亿立方米")
#         print(f"  Y value = {data['Y']:.2f} mm")
#         print(f"  Joint Probability = {data['Joint Probability']:.2%}")
#         print("-" * 60)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Slightly increased figure size
# x_range = np.linspace(min(x), max(x), 100)
# x_cdf = 1 - dist_x.cdf(x_range, *params_x)
# ax1.plot(x_range, x_cdf, 'b-', lw=2, label='Gamma Exceedance CDF')
# for cat, q in quantiles.items():
#     x_val = x_values[cat]
#     exceedance_prob = 1 - q
#     ax1.axvline(x_val, color='gray', linestyle='--', alpha=0.7)
#     ax1.axhline(exceedance_prob, color='gray', linestyle='--', alpha=0.7)
#     ax1.scatter([x_val], [exceedance_prob], color='red', s=50, zorder=5)
#     ax1.text(x_val, exceedance_prob + 0.05, f'{cat} (X={x_val:.2f})', color='black', ha='center', fontsize=12)
# ax1.set_xlabel('入境水量（亿立方米）', fontsize=14)
# ax1.set_ylabel('频率', fontsize=14)
# ax1.set_title('入境水量 (Gamma) 理论频率曲线', fontsize=16)
# ax1.grid(True, alpha=0.3)
# ax1.legend(fontsize=12)
#
# y_range = np.linspace(min(y), max(y), 100)
# y_cdf = 1 - dist_y.cdf(y_range, *params_y)
# ax2.plot(y_range, y_cdf, 'g-', lw=2, label='LogNormal Exceedance CDF')
# for cat, q in quantiles.items():
#     y_val = y_values[cat]
#     exceedance_prob = 1 - q
#     ax2.axvline(y_val, color='gray', linestyle='--', alpha=0.7)
#     ax2.axhline(exceedance_prob, color='gray', linestyle='--', alpha=0.7)
#     ax2.scatter([y_val], [exceedance_prob], color='red', s=50, zorder=5)
#     ax2.text(y_val, exceedance_prob + 0.05, f'{cat} (Y={y_val:.2f})', color='black', ha='center', fontsize=12)
# ax2.set_xlabel('年平均降水量（mm）', fontsize=14)
# ax2.set_ylabel('频率', fontsize=14)
# ax2.set_title('年平均降水量 (LogNormal) 理论频率曲线', fontsize=16)
# ax2.grid(True, alpha=0.3)
# ax2.legend(fontsize=12)
#
# plt.tight_layout()
# plt.savefig('exceedance_cdf.png', dpi=300, bbox_inches='tight')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, gamma, lognorm, weibull_min
from copulae import GaussianCopula, StudentCopula, ClaytonCopula, FrankCopula, GumbelCopula
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.lines import Line2D

# Set global font parameters with differentiated sizes
plt.rcParams.update({
    'font.size': 12,              # Base font size for general text
    'axes.titlesize': 18,         # Larger title font size for emphasis
    'axes.labelsize': 14,         # Axis label font size
    'xtick.labelsize': 12,        # X-axis tick label font size
    'ytick.labelsize': 12,        # Y-axis tick label font size
    'legend.fontsize': 10,        # Smaller legend font size
    'font.sans-serif': ['Arial'], # Use Arial for compatibility
    'axes.unicode_minus': False,   # Handle minus sign
    'text.usetex': False          # Use matplotlib's mathtext for superscripts
})

def fit_best_distribution(data):
    distributions = [
        {'name': 'Normal', 'dist': norm, 'fix_params': {}, 'k': 2},
        {'name': 'Exponential', 'dist': expon, 'fix_params': {'floc': 0}, 'k': 1},
        {'name': 'Gamma', 'dist': gamma, 'fix_params': {'floc': 0}, 'k': 2},
        {'name': 'LogNormal', 'dist': lognorm, 'fix_params': {'floc': 0}, 'k': 2},
        {'name': 'Weibull', 'dist': weibull_min, 'fix_params': {'floc': 0}, 'k': 2},
    ]
    results = []
    for d in distributions:
        dist = d['dist']
        name = d['name']
        fix_params = d['fix_params']
        k = d['k']
        try:
            params = dist.fit(data, **fix_params)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            aic = 2 * k - 2 * log_likelihood
            results.append({'name': name, 'dist': dist, 'params': params, 'aic': aic, 'k': k})
        except Exception as e:
            continue
    results.sort(key=lambda x: x['aic'])
    best = results[0]
    return best['name'], best['dist'], best['params'], best['aic']


import matplotlib.pyplot as plt
import numpy as np


def plot_all_distributions(data, dist_list, title):
    # 保持画布大小不变，通过增大字体来适应论文排版（缩小后字体依然清晰）
    plt.figure(figsize=(12, 8))
    x = np.linspace(min(data), max(data), 200)

    # Plot histogram
    plt.hist(data, bins=15, density=True, alpha=0.3, color='gray', label='Observed Data')

    # Plot all distribution curves
    for dist_info in dist_list:
        dist = dist_info['dist']
        params = dist_info['params']
        label = f"{dist_info['name']} (AIC={dist_info['aic']:.1f})"
        # 建议稍微增加线宽(linewidth)以匹配大字体，但遵循“其他不要改”原则，此处保留默认或仅作微调
        plt.plot(x, dist.pdf(x, *params), label=label, linewidth=2.5)

        # 字体放大 3 倍以上
    plt.xlabel('Observed Values', fontsize=25)  # 原 14 -> 45
    plt.ylabel('Density', fontsize=25)  # 原 14 -> 45
    plt.title(title, fontsize=30)  # 原 18 -> 60
    plt.legend(fontsize=20)  # 原 10 -> 32

    # 设置坐标轴刻度字体大小（非常重要，否则刻度看不清）
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.grid(True)

    # 建议加上 layout 调整防止大字体被切边，如不需要可注释掉
    plt.tight_layout()

    # plt.show()

def plot_pdf_fit(data, dist_name, dist, params, aic, name):
    n = len(data)
    sorted_data = np.sort(data)
    p_empirical = (np.arange(n) + 0.5) / n
    p_theoretical = np.linspace(0.01, 0.99, 100)
    quantiles_theoretical = dist.ppf(p_theoretical, *params)

    plt.figure(figsize=(12, 8))
    plt.scatter(p_empirical, sorted_data, color='blue', alpha=0.6, label='Observed Values')
    plt.plot(p_theoretical, quantiles_theoretical, 'r-', lw=2, label=f'Fitted {dist_name}')
    plt.xlabel('Probability', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title(f'{name} Marginal Distribution Fit: {dist_name} (AIC={aic:.2f})', fontsize=18)
    plt.legend(fontsize=10)
    plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    # Data
    x = np.array([5.29, 2.6, 4.18, 6.32, 4.59, 4.25, 3.45, 5.35, 3.03, 6.93, 4.71,
                  8.62, 10.75, 3.59, 4.49, 7.13, 5.03, 8.19, 6.08, 6.61, 10.01, 9.03, 7.92, 6.19])
    y = np.array([462, 413, 453, 539, 468, 448, 499, 638, 448, 524, 552, 708, 501,
                  439, 583, 660, 592, 590, 506, 560, 924, 482, 727, 782])

    def get_all_distributions(data):
        distributions = [
            {'name': 'Normal', 'dist': norm, 'fix_params': {}, 'k': 2},
            {'name': 'Exponential', 'dist': expon, 'fix_params': {'floc': 0}, 'k': 1},
            {'name': 'Gamma', 'dist': gamma, 'fix_params': {'floc': 0}, 'k': 2},
            {'name': 'LogNormal', 'dist': lognorm, 'fix_params': {'floc': 0}, 'k': 2},
            {'name': 'Weibull', 'dist': weibull_min, 'fix_params': {'floc': 0}, 'k': 2},
        ]
        results = []
        for d in distributions:
            dist = d['dist']
            name = d['name']
            fix_params = d['fix_params']
            k = d['k']
            try:
                params = dist.fit(data, **fix_params)
                log_likelihood = np.sum(dist.logpdf(data, *params))
                aic = 2 * k - 2 * log_likelihood
                results.append({'name': name, 'dist': dist, 'params': params, 'aic': aic})
            except Exception as e:
                continue
        results.sort(key=lambda x: x['aic'])
        return results

    # Get all distribution results
    x_dists = get_all_distributions(x)
    y_dists = get_all_distributions(y)

    # Extract best distribution
    dist_name_x, dist_x, params_x, aic_x = x_dists[0]['name'], x_dists[0]['dist'], x_dists[0]['params'], x_dists[0]['aic']
    dist_name_y, dist_y, params_y, aic_y = y_dists[0]['name'], y_dists[0]['dist'], y_dists[0]['params'], y_dists[0]['aic']

    # Plot all distributions
    plot_all_distributions(x, x_dists, '')
    plot_all_distributions(y, y_dists, '')

    print(f"X Best Marginal Distribution: {dist_name_x} (AIC={aic_x:.2f})")
    print(f"Y Best Marginal Distribution: {dist_name_y} (AIC={aic_y:.2f})")
    plot_pdf_fit(x, dist_name_x, dist_x, params_x, aic_x, 'Annual Inbound Water Volume')
    plot_pdf_fit(y, dist_name_y, dist_y, params_y, aic_y, 'Annual Average Precipitation')

    # Copula analysis
    u = dist_x.cdf(x, *params_x)
    v = dist_y.cdf(y, *params_y)
    data_copula = np.column_stack((u, v))

    copula_types = [
        ('Gaussian', GaussianCopula),
        ('Clayton', ClaytonCopula),
        ('Frank', FrankCopula),
        ('Gumbel', GumbelCopula),
    ]

    copula_results = []
    for name, copula_class in copula_types:
        try:
            cop = copula_class(dim=2)
            cop.fit(data_copula)
            log_lik = cop.log_lik(data_copula)
            params = cop.params
            k = len(params) if isinstance(params, (list, np.ndarray)) else 1
            aic = -2 * log_lik + 2 * k
            copula_results.append({'name': name, 'aic': aic, 'copula': cop})
            print(f"{name} Copula - AIC: {aic:.2f}")
        except Exception as e:
            print(f"Failed to fit {name}: {e}")

    copula_results.sort(key=lambda x: x['aic'])
    print("\nCopula Ranking by AIC:")
    for i, res in enumerate(copula_results):
        print(f"{i + 1}. {res['name']}: AIC={res['aic']:.2f}")

    # 3D Copula visualization
    grid_size = 30
    x_grid = np.linspace(min(x), max(x), grid_size)
    y_grid = np.linspace(min(y), max(y), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    u_mesh = dist_x.cdf(x_mesh, *params_x)
    v_mesh = dist_y.cdf(y_mesh, *params_y)
    grid_points = np.column_stack((u_mesh.ravel(), v_mesh.ravel()))

    for res in copula_results:
        name = res['name']
        cop = res['copula']
        try:
            if hasattr(cop, 'cdf'):
                cdf = cop.cdf(grid_points).reshape(x_mesh.shape)
            else:
                from scipy.integrate import dblquad
                cdf = np.zeros_like(x_mesh)
                for i in range(x_mesh.shape[0]):
                    for j in range(x_mesh.shape[1]):
                        result, _ = dblquad(
                            lambda u, v: cop.pdf([[u, v]]),
                            0, u_mesh[i, j],
                            lambda x: 0, lambda x: v_mesh[i, j]
                        )
                        cdf[i, j] = result
                print(f"[Warning] {name} CDF calculated by integration")

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x_mesh, y_mesh, cdf, cmap=cm.coolwarm, alpha=0.8, rstride=1, cstride=1)
            ax.scatter(x, y, cop.cdf(data_copula), c='red', s=30, alpha=0.9, label='Observed Values')
            ax.set_xlabel('Annual Inbound Water Volume ($10^8$ m³)', fontsize=14)
            ax.set_ylabel('Annual Average Precipitation (mm)', fontsize=14)
            ax.set_zlabel('C(u,v)', fontsize=14)
            ax.set_zlim(0, 1)
            ax.set_title(f'{name} Copula Function\n(Annual Inbound Water Volume: {dist_name_x}, Annual Precipitation: {dist_name_y})', fontsize=18)
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            plt.legend(fontsize=10)
            plt.tight_layout()
            # plt.show()

        except Exception as e:
            print(f"Plot error for {name}: {e}")

def calculate_category_probabilities(x_data, y_data, copula, x_dist, x_params, y_dist, y_params):
    q_x = [x_dist.ppf(0.375, *x_params), x_dist.ppf(0.625, *x_params)]
    q_y = [y_dist.ppf(0.375, *y_params), y_dist.ppf(0.625, *y_params)]

    categories = ['Low', 'Medium', 'High']
    theory_probs = pd.DataFrame(index=categories, columns=categories)
    actual_probs = pd.DataFrame(index=categories, columns=categories)

    for i, x_cat in enumerate(categories):
        for j, y_cat in enumerate(categories):
            x_low = -np.inf if i == 0 else q_x[0] if i == 1 else q_x[1]
            x_high = q_x[0] if i == 0 else q_x[1] if i == 1 else np.inf
            y_low = -np.inf if j == 0 else q_y[0] if j == 1 else q_y[1]
            y_high = q_y[0] if j == 0 else q_y[1] if j == 1 else np.inf

            u_low = x_dist.cdf(x_low, *x_params)
            u_high = x_dist.cdf(x_high, *x_params)
            v_low = y_dist.cdf(y_low, *y_params)
            v_high = y_dist.cdf(y_high, *y_params)
            theory_prob = copula.cdf([u_high, v_high]) - copula.cdf([u_low, v_high]) \
                          - copula.cdf([u_high, v_low]) + copula.cdf([u_low, v_low])
            theory_probs.loc[x_cat, y_cat] = theory_prob
            mask = (x_data > x_low) & (x_data <= x_high) & (y_data > y_low) & (y_data <= y_high)
            actual_probs.loc[x_cat, y_cat] = np.mean(mask)

    return theory_probs.astype(float), actual_probs.astype(float)

# Use best Copula
best_cop = copula_results[0]['copula']
theory_probs, actual_probs = calculate_category_probabilities(x, y, best_cop, dist_x, params_x, dist_y, params_y)

print("\nTheoretical Probability Matrix (Best Copula):")
print(theory_probs.applymap(lambda x: f"{x:.2%}"))
print("\nActual Observed Probabilities:")
print(actual_probs.applymap(lambda x: f"{x:.2%}"))

# Scatter plot with categories
plt.figure(figsize=(12, 8))
categories = ['Low', 'Medium', 'High']
colors = ['#1f77b4', '#2ca02c', '#d62728']

q_x1 = dist_x.ppf(0.375, *params_x)
q_x2 = dist_x.ppf(0.625, *params_x)
q_y1 = dist_y.ppf(0.375, *params_y)
q_y2 = dist_y.ppf(0.625, *params_y)

plt.axvline(q_x1, color='gray', linestyle='--', alpha=0.7)
plt.axvline(q_x2, color='gray', linestyle='--', alpha=0.7)
plt.axhline(q_y1, color='gray', linestyle='--', alpha=0.7)
plt.axhline(q_y2, color='gray', linestyle='--', alpha=0.7)

for i in range(len(x)):
    x_cat = 0 if x[i] <= q_x1 else 1 if x[i] <= q_x2 else 2
    y_cat = 0 if y[i] <= q_y1 else 1 if y[i] <= q_y2 else 2
    plt.scatter(x[i], y[i], color=colors[x_cat],
                marker='o' if y_cat == 0 else 's' if y_cat == 1 else '^',
                alpha=0.7, edgecolor='w')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Y=Low', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='Y=Medium', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='Y=High', markerfacecolor='gray', markersize=10),
    Line2D([0], [0], color=colors[0], lw=4, label='X=Low'),
    Line2D([0], [0], color=colors[1], lw=4, label='X=Medium'),
    Line2D([0], [0], color=colors[2], lw=4, label='X=High')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=20)
plt.xlabel('Annual Inbound Water Volume ($10^8$ m³)', fontsize=20)
plt.ylabel('Annual Average Precipitation (mm)', fontsize=20)
plt.title('', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.grid(True, alpha=0.3)
# plt.show()

def plot_copula_qq(data_copula, copula, copula_name):
    cdf_values = copula.cdf(data_copula)
    empirical_quantiles = np.sort(cdf_values)
    n = len(cdf_values)
    theoretical_quantiles = (np.arange(1, n + 1) - 0.5) / n

    plt.figure(figsize=(10, 8))
    plt.scatter(theoretical_quantiles, empirical_quantiles, color='blue', alpha=0.6, label='Q-Q Points')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='y=x Reference Line')
    plt.xlabel('Theoretical Quantiles (Uniform Distribution)', fontsize=14)
    plt.ylabel('Empirical Quantiles (Clayton Copula CDF)', fontsize=14)
    plt.title(f'{copula_name} Copula Q-Q Plot', fontsize=18)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

# Clayton Copula Q-Q plot
clayton_cop = None
for res in copula_results:
    if res['name'] == 'Clayton':
        clayton_cop = res['copula']
        break

if clayton_cop is not None:
    plot_copula_qq(data_copula, clayton_cop, 'Clayton')
else:
    print("Clayton Copula not found, possibly due to fitting failure.")

# Exceedance CDF plot
dist_x = gamma
params_x = gamma.fit(x, floc=0)
dist_y = lognorm
params_y = lognorm.fit(y, floc=0)

u = dist_x.cdf(x, *params_x)
v = dist_y.cdf(y, *params_y)
data_copula = np.column_stack((u, v))
copula = ClaytonCopula(dim=2)
copula.fit(data_copula)

categories = ['Low', 'Medium', 'High']
quantiles = {'Low': 0.05, 'Medium': 0.5, 'High': 0.95}
x_values = {cat: dist_x.ppf(quantiles[cat], *params_x) for cat in categories}
y_values = {cat: dist_y.ppf(quantiles[cat], *params_y) for cat in categories}

joint_probs = {}
for x_cat in categories:
    for y_cat in categories:
        u_val = dist_x.cdf(x_values[x_cat], *params_x)
        v_val = dist_y.cdf(y_values[y_cat], *params_y)
        joint_prob = copula.cdf([u_val, v_val])
        joint_probs[(x_cat, y_cat)] = {
            'X': x_values[x_cat],
            'Y': y_values[y_cat],
            'Joint Probability': joint_prob
        }

print("\nRepresentative (X, Y) Combinations and Joint Probabilities (Clayton Copula):")
print("-" * 60)
for x_cat in categories:
    for y_cat in categories:
        data = joint_probs[(x_cat, y_cat)]
        print(f"X={x_cat}, Y={y_cat}:")
        print(f"  X value = {data['X']:.2f} $10^8$ m³")
        print(f"  Y value = {data['Y']:.2f} mm")
        print(f"  Joint Probability = {data['Joint Probability']:.2%}")
        print("-" * 60)

import matplotlib.pyplot as plt
import numpy as np

# 设置美化风格
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---------- 子图 1：Annual Inbound Water Volume ----------
x_range = np.linspace(min(x), max(x), 100)
x_cdf = 1 - dist_x.cdf(x_range, *params_x)
ax1.plot(x_range, x_cdf, color='#1f77b4', lw=2.5, label='Gamma Exceedance CDF')

for cat, q in quantiles.items():
    x_val = x_values[cat]
    exceedance_prob = 1 - q

    ax1.axvline(x_val, color='gray', linestyle='--', alpha=0.6)
    ax1.axhline(exceedance_prob, color='gray', linestyle='--', alpha=0.6)
    ax1.scatter([x_val], [exceedance_prob], color='crimson', s=60, zorder=5)

    # 动态偏移
    if exceedance_prob > 0.85:
        y_offset = -0.08
        va = 'top'
    else:
        y_offset = 0.1
        va = 'bottom'

    ax1.annotate(f'{cat}\nX={x_val:.2f}',
                 xy=(x_val, exceedance_prob),
                 xytext=(x_val, exceedance_prob + y_offset),
                 textcoords='data',
                 ha='center',
                 va=va,
                 fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

ax1.set_xlabel('Annual Inbound Water Volume ($10^8$ m³)')
ax1.set_ylabel('Exceedance Probability')
ax1.set_title('Annual Inbound Water Volume Exceedance Curve (Gamma)')
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.legend(loc='best', frameon=True)

# ---------- 子图 2：Annual Precipitation ----------
y_range = np.linspace(min(y), max(y), 100)
y_cdf = 1 - dist_y.cdf(y_range, *params_y)
ax2.plot(y_range, y_cdf, color='#2ca02c', lw=2.5, label='LogNormal Exceedance CDF')

for cat, q in quantiles.items():
    y_val = y_values[cat]
    exceedance_prob = 1 - q

    ax2.axvline(y_val, color='gray', linestyle='--', alpha=0.6)
    ax2.axhline(exceedance_prob, color='gray', linestyle='--', alpha=0.6)
    ax2.scatter([y_val], [exceedance_prob], color='crimson', s=60, zorder=5)

    # 动态偏移
    if exceedance_prob > 0.85:
        y_offset = -0.08
        va = 'top'
    else:
        y_offset = 0.1
        va = 'bottom'

    ax2.annotate(f'{cat}\nY={y_val:.2f}',
                 xy=(y_val, exceedance_prob),
                 xytext=(y_val, exceedance_prob + y_offset),
                 textcoords='data',
                 ha='center',
                 va=va,
                 fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

ax2.set_xlabel('Annual Average Precipitation (mm)')
ax2.set_ylabel('Exceedance Probability')
ax2.set_title('Annual Precipitation Exceedance Curve (LogNormal)')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.legend(loc='best', frameon=True)

# ---------- 布局调整 ----------
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# ---------- 输出 ----------
plt.savefig('exceedance_cdf_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================================================================
# 【论文技术路线专用】生成无坐标轴、无背景的纯净Copula函数面
# =========================================================================
print("\n正在生成纯净Copula函数面（无坐标轴版）...")

# 1. 准备绘图数据 (使用标准0-1空间展示方法论原理)
# 增加密度以获得更平滑的曲面效果
u_pure = np.linspace(0.001, 0.999, 100)
v_pure = np.linspace(0.001, 0.999, 100)
U_pure, V_pure = np.meshgrid(u_pure, v_pure)
grid_pure = np.column_stack((U_pure.ravel(), V_pure.ravel()))

# 2. 计算Z轴数据 (默认计算CDF函数面，如需密度面请将.cdf改为.pdf)
try:
    if hasattr(best_cop, 'cdf'):
        Z_pure = best_cop.cdf(grid_pure).reshape(U_pure.shape)
    else:
        # 备用计算逻辑
        from scipy.integrate import dblquad

        Z_pure = np.zeros_like(U_pure)
        for i in range(U_pure.shape[0]):
            for j in range(U_pure.shape[1]):
                val, _ = dblquad(lambda u, v: best_cop.pdf([[u, v]]),
                                 0, U_pure[i, j], lambda x: 0, lambda x: V_pure[i, j])
                Z_pure[i, j] = val
except Exception as e:
    print(f"计算出错，尝试使用PDF替代: {e}")
    Z_pure = best_cop.pdf(grid_pure).reshape(U_pure.shape)

# 3. 绘制纯净曲面
fig_pure = plt.figure(figsize=(10, 8))
ax_pure = fig_pure.add_subplot(111, projection='3d')

# 绘制曲面 (linewidth=0 去除曲面上的黑色网格线)
surf = ax_pure.plot_surface(U_pure, V_pure, Z_pure,
                            cmap=cm.coolwarm,
                            linewidth=0,
                            antialiased=True,
                            shade=True)

# 【核心修改】彻底关闭坐标轴、刻度、背景壁、网格
ax_pure.set_axis_off()

# 调整最佳视角 (elev=30, azim=-135 是比较标准的立体视角)
ax_pure.view_init(elev=30, azim=-135)

# 4. 保存图片 (transparent=True 实现背景透明，pad_inches=0 去除白边)
filename_pure = 'pure_copula_surface.png'
plt.savefig(filename_pure,
            transparent=True,
            bbox_inches='tight',
            pad_inches=0,
            dpi=300)

print(f"纯净函数面已保存为: {filename_pure}")
# plt.show()