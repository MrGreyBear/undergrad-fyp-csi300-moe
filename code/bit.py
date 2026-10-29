import matplotlib.pyplot as plt

# 1. 数据整合 (提取自您提供的所有图片)
# 数据顺序：9月 -> 10月 -> 11月 -> 12月
data_map = [
    # Image 1 (9.22 - 10.13)
    ('9.22', 8.2), ('9.23', 8.0), ('9.24', 8.5), ('9.25', 8.5), ('9.26', 8.0),
    ('9.29', 8.2), ('9.30', 8.3), ('10.9', 8.5), ('10.10', 8.2), ('10.13', 8.2),
    
    # Image 4 (10.15 - 10.29)
    ('10.15', 8.0), ('10.16', 7.5), ('10.17', 6.8), ('10.20', 6.8), ('10.21', 7.5),
    ('10.22', 7.2), ('10.23', 7.2), ('10.24', 8.2), ('10.27', 8.2), ('10.28', 8.2), ('10.29', 8.5),
    
    # Image 3 (10.30 - 11.10)
    ('10.30', 8.2), ('10.31', 8.0), ('11.3', 8.2), ('11.4', 7.5), ('11.5', 7.6),
    ('11.6', 7.8), ('11.7', 8.0), ('11.10', 8.0),
    
    # Image 2 (11.10 duplicate skipped, 11.11 - 11.24)
    ('11.11', 8.0), ('11.12', 7.8), ('11.13', 8.2), ('11.14', 7.8), ('11.17', 7.5),
    ('11.18', 7.0), ('11.19', 7.0), ('11.20', 6.8), ('11.21', 6.5), ('11.24', 6.5),
    
    # Image 5 (11.25 - 12.8)
    ('11.25', 6.5), ('11.26', 6.5), ('11.27', 6.5), ('11.28', 6.5), ('12.1', 6.8),
    ('12.2', 6.5), ('12.3', 6.5), ('12.4', 6.5), ('12.5', 7.2), ('12.8', 7.5)
]

dates = [x[0] for x in data_map]
scores = [x[1] for x in data_map]

# 2. 设置画布 (由于时间跨度长，设置得宽一些)
fig, ax = plt.subplots(figsize=(20, 6))

# 3. 绘制背景色带
# 绿色 (Bull Zone > 8)
ax.axhspan(8, 9, facecolor='#d9ead3', alpha=0.6) 
ax.text(0.005, 0.88, 'Bull Zone (>8)', transform=ax.transAxes, color='green', 
        fontweight='bold', fontsize=10, va='center')

# 黄色 (Neutral Zone 6-8)
ax.axhspan(6, 8, facecolor='#fff2cc', alpha=0.6)
ax.text(0.005, 0.60, 'Neutral Zone (6-8)', transform=ax.transAxes, color='#d69d00', 
        fontweight='bold', fontsize=10, va='center')

# 红色 (Bear Zone < 6)
ax.axhspan(4, 6, facecolor='#fce5cd', alpha=0.6)
ax.text(0.005, 0.20, 'Bear Zone (<6)', transform=ax.transAxes, color='red', 
        fontweight='bold', fontsize=10, va='center')

# 4. 绘制折线图
ax.plot(dates, scores, color='#ff9900', linewidth=2.5, marker='o', 
        markersize=6, markerfacecolor='#ff9900', markeredgecolor='white', markeredgewidth=1.5)

# 5. 设置坐标轴
ax.set_ylim(4, 9)
ax.set_ylabel('Market Sentiment Score', fontsize=12)
ax.set_title('China Stock Market Sentiment Score (Sep 22 - Dec 8, 2025)', fontsize=16, pad=20)

# 6. 美化图表
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.grid(False) 

# X轴标签优化：由于日期很多，旋转45度并隔几个显示一个，防止重叠
# 这里设置为每隔2个显示一个标签 (slice [::2])，如果觉得不够密可以去掉 [::2]
plt.xticks(range(0, len(dates), 2), dates[::2], rotation=45, fontsize=9)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()