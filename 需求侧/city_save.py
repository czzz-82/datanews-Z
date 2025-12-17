# ==================== 新增：城市对比分析功能 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib
import os

def create_city_comparison_analysis(df):
    """创建城市对比分析"""
    matplotlib.rc("font",family='FangSong')
    
    # 创建保存图片的文件夹
    os.makedirs('city', exist_ok=True)
    
    # 检查数据中是否有地区信息
    if '地区' not in df.columns:
        # 尝试从其他列提取地区信息
        print("数据中没有'地区'列，尝试从其他字段提取地区信息...")
        
        # 可能包含地区信息的列
        potential_city_columns = ['工作地点', '地点', '城市', '所在地', '区域']
        city_column = None
        
        for col in potential_city_columns:
            if col in df.columns:
                city_column = col
                break
        
        if city_column:
            df['地区'] = df[city_column]
            print(f"使用'{city_column}'列作为地区信息")
        else:
            print("无法找到地区信息，跳过城市对比分析")
            return
    
    # 清洗地区数据
    df['地区'] = df['地区'].fillna('未知')
    
    # 提取城市名称（去除省市后缀等）
    def clean_city_name(city_str):
        if pd.isna(city_str):
            return '未知'
        city_str = str(city_str)
        # 提取常见城市名称模式
        patterns = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', 
                   '重庆', '西安', '天津', '苏州', '长沙', '郑州', '青岛', '宁波',
                   '厦门', '福州', '沈阳', '大连', '长春', '哈尔滨', '济南', '合肥',
                   '南昌', '南宁', '昆明', '贵阳', '太原', '石家庄', '兰州', '银川',
                   '西宁', '乌鲁木齐', '海口', '三亚', '珠海', '东莞', '佛山', '中山']
        
        for pattern in patterns:
            if pattern in city_str:
                return pattern
        
        # 如果是简单的城市名，直接返回
        if len(city_str) <= 4:
            return city_str
        
        return '其他'
    
    df['城市_清洗'] = df['地区'].apply(clean_city_name)
    
    # 统计城市岗位数量
    city_counts = df['城市_清洗'].value_counts()
    
    # 过滤掉'未知'和'其他'
    city_counts = city_counts[~city_counts.index.isin(['未知', '其他'])]
    
    if len(city_counts) < 2:
        print(f"只有{len(city_counts)}个有效城市数据，无法进行城市对比分析")
        return
    
    # 创建城市对比分析图 - 现在每个子图单独保存
    
    # 子图1: 各城市岗位数量对比
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    top_cities = city_counts.head(15)
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(top_cities)))
    bars1 = ax1.bar(range(len(top_cities)), top_cities.values, color=colors1)
    ax1.set_title('1. 各城市岗位数量对比 (Top 15)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('城市', fontsize=12)
    ax1.set_ylabel('岗位数量', fontsize=12)
    ax1.set_xticks(range(len(top_cities)))
    ax1.set_xticklabels(top_cities.index, rotation=45, ha='right', fontsize=10)
    
    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('city/1_各城市岗位数量对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("已保存: city/1_各城市岗位数量对比.png")
    
    # 子图2: 各城市平均薪资对比
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    # 计算各城市平均薪资
    city_salary = df.groupby('城市_清洗')['薪资_月平均K'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
    city_salary = city_salary[~city_salary.index.isin(['未知', '其他'])]
    city_salary = city_salary[city_salary['count'] >= 3]  # 只保留至少有3个岗位的城市
    
    if len(city_salary) > 0:
        top_city_salary = city_salary.head(15)
        colors2 = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_city_salary)))
        bars2 = ax2.barh(range(len(top_city_salary)), top_city_salary['mean'], color=colors2)
        ax2.set_title('2. 各城市平均薪资对比 (Top 15)', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('平均月薪 (K)', fontsize=12)
        ax2.set_ylabel('城市', fontsize=12)
        ax2.set_yticks(range(len(top_city_salary)))
        ax2.set_yticklabels(top_city_salary.index, fontsize=10)
        
        # 添加误差线和样本数
        for i, (city, row) in enumerate(top_city_salary.iterrows()):
            mean_salary = row['mean']
            count = int(row['count'])
            std_salary = row['std']
            
            # 添加误差线
            ax2.errorbar(mean_salary, i, xerr=std_salary, fmt='none', ecolor='black', capsize=3, alpha=0.7)
            
            # 添加标注
            ax2.text(mean_salary + std_salary + 0.5, i, 
                    f'{mean_salary:.1f}±{std_salary:.1f}K\n(n={count})', 
                    ha='left', va='center', fontsize=8)
        
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2.text(0.5, 0.5, '没有足够的城市薪资数据', ha='center', va='center', fontsize=12)
        ax2.set_title('2. 各城市平均薪资对比', fontsize=16, fontweight='bold', pad=20)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/2_各城市平均薪资对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("已保存: city/2_各城市平均薪资对比.png")
    
    # 子图3: 热门城市行业分布
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    # 选择岗位数量最多的前5个城市
    top_5_cities = city_counts.head(5).index.tolist()
    
    if len(top_5_cities) >= 2:
        # 获取每个城市的前3个行业
        city_industry_data = {}
        
        for city in top_5_cities:
            city_df = df[df['城市_清洗'] == city]
            if len(city_df) > 0:
                top_industries = city_df['行业'].value_counts().head(3)
                city_industry_data[city] = top_industries
        
        # 准备堆叠柱状图数据
        industries_set = set()
        for industries in city_industry_data.values():
            industries_set.update(industries.index.tolist())
        
        # 创建DataFrame用于堆叠柱状图
        industry_df = pd.DataFrame(index=top_5_cities, columns=list(industries_set)).fillna(0)
        
        for city, industries in city_industry_data.items():
            for industry, count in industries.items():
                industry_df.loc[city, industry] = count
        
        # 绘制堆叠柱状图
        colors3 = plt.cm.tab20c(np.linspace(0, 1, len(industries_set)))
        bottom_vals = np.zeros(len(top_5_cities))
        
        for i, industry in enumerate(industry_df.columns):
            counts = industry_df[industry].values
            ax3.bar(range(len(top_5_cities)), counts, bottom=bottom_vals, 
                   label=industry, color=colors3[i], alpha=0.8)
            bottom_vals += counts
        
        ax3.set_title('3. 热门城市行业分布对比', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('城市', fontsize=12)
        ax3.set_ylabel('岗位数量', fontsize=12)
        ax3.set_xticks(range(len(top_5_cities)))
        ax3.set_xticklabels(top_5_cities, fontsize=10)
        ax3.legend(title='行业', fontsize=8, title_fontsize=9, loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, '没有足够的城市数据进行分析', ha='center', va='center', fontsize=12)
        ax3.set_title('3. 热门城市行业分布对比', fontsize=16, fontweight='bold', pad=20)
        ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/3_热门城市行业分布对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("已保存: city/3_热门城市行业分布对比.png")
    
    # 子图4: 城市薪资与岗位数量关系散点图
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # 合并城市统计数据
    if len(city_salary) > 0 and len(city_counts) > 0:
        # 创建城市数据DataFrame
        city_stats = pd.DataFrame({
            '城市': city_counts.index,
            '岗位数量': city_counts.values
        })
        
        # 合并薪资数据
        city_stats = city_stats.merge(city_salary[['mean', 'count']], 
                                     left_on='城市', right_index=True, how='inner')
        city_stats = city_stats.rename(columns={'mean': '平均薪资', 'count': '样本数'})
        
        # 只保留样本数大于等于3的城市
        city_stats = city_stats[city_stats['样本数'] >= 3]
        
        if len(city_stats) >= 3:
            # 创建散点图
            scatter = ax4.scatter(city_stats['平均薪资'], city_stats['岗位数量'], 
                                 s=city_stats['样本数']*10,  # 点的大小表示样本数
                                 c=city_stats['平均薪资'], cmap='viridis', alpha=0.7)
            
            ax4.set_title('4. 城市薪资与岗位数量关系', fontsize=16, fontweight='bold', pad=20)
            ax4.set_xlabel('平均月薪 (K)', fontsize=12)
            ax4.set_ylabel('岗位数量', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            # 添加城市标签
            for i, row in city_stats.iterrows():
                if row['岗位数量'] > city_stats['岗位数量'].quantile(0.7) or row['平均薪资'] > city_stats['平均薪资'].quantile(0.7):
                    ax4.annotate(row['城市'], (row['平均薪资'], row['岗位数量']), 
                                fontsize=9, alpha=0.8)
            
            # 添加趋势线
            try:
                z = np.polyfit(city_stats['平均薪资'], city_stats['岗位数量'], 1)
                p = np.poly1d(z)
                ax4.plot(city_stats['平均薪资'], p(city_stats['平均薪资']), 
                        "r--", alpha=0.5, label='趋势线')
                ax4.legend()
            except:
                pass
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax4, label='平均薪资 (K)')
        else:
            ax4.text(0.5, 0.5, '没有足够的城市数据绘制关系图', ha='center', va='center', fontsize=12)
            ax4.set_title('4. 城市薪资与岗位数量关系', fontsize=16, fontweight='bold', pad=20)
            ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, '没有足够的城市数据绘制关系图', ha='center', va='center', fontsize=12)
        ax4.set_title('4. 城市薪资与岗位数量关系', fontsize=16, fontweight='bold', pad=20)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/4_城市薪资与岗位数量关系.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("已保存: city/4_城市薪资与岗位数量关系.png")
    
    # 打印城市统计摘要
    print_city_summary(df, city_salary, city_counts)
    
    return city_salary

def print_city_summary(df, city_salary, city_counts):
    """打印城市对比统计摘要"""
    print("\n" + "="*60)
    print("城市对比分析摘要")
    print("="*60)
    
    print(f"\n1. 城市覆盖情况:")
    print(f"   - 分析城市总数: {len(city_counts)}")
    print(f"   - 总岗位数量: {len(df):,}")
    
    # 前5个城市岗位占比
    top_5_cities = city_counts.head(5)
    top_5_percentage = top_5_cities.sum() / len(df) * 100
    print(f"   - 前5大城市岗位占比: {top_5_percentage:.1f}%")
    
    print(f"\n2. 城市岗位数量Top 5:")
    for i, (city, count) in enumerate(top_5_cities.items(), 1):
        percentage = count / len(df) * 100
        print(f"   {i}. {city}: {count}个岗位 ({percentage:.1f}%)")
    
    print(f"\n3. 城市平均薪资Top 5:")
    if len(city_salary) > 0:
        top_5_salary = city_salary.head(5)
        for i, (city, row) in enumerate(top_5_salary.iterrows(), 1):
            print(f"   {i}. {city}: {row['mean']:.1f}±{row['std']:.1f}K/月 (样本:{int(row['count'])})")
    else:
        print("   数据不足，无法计算城市平均薪资")
    
    # 计算薪资差异
    if len(city_salary) >= 2:
        max_salary_city = city_salary.iloc[0]
        min_salary_city = city_salary.iloc[-1]
        salary_ratio = max_salary_city['mean'] / min_salary_city['mean']
        print(f"\n4. 薪资差异分析:")
        print(f"   - 最高薪资城市: {max_salary_city.name} ({max_salary_city['mean']:.1f}K)")
        print(f"   - 最低薪资城市: {min_salary_city.name} ({min_salary_city['mean']:.1f}K)")
        print(f"   - 薪资倍率: {salary_ratio:.2f}倍")
    
    # 热门城市行业分析
    print(f"\n5. 热门城市特色行业:")
    top_3_cities = city_counts.head(3).index.tolist()
    for city in top_3_cities:
        city_df = df[df['城市_清洗'] == city]
        if len(city_df) > 0:
            top_industry = city_df['行业'].value_counts().head(1)
            if len(top_industry) > 0:
                industry_name = top_industry.index[0]
                industry_count = top_industry.values[0]
                industry_percentage = industry_count / len(city_df) * 100
                print(f"   - {city}: {industry_name} ({industry_percentage:.1f}%)")

# ==================== 新增：基于行业映射的城市对比分析 ====================
def create_industry_based_city_comparison(df):
    """创建基于标准化行业分类的城市对比分析"""
    # 确保city文件夹存在
    os.makedirs('city', exist_ok=True)
    
    # 导入行业映射器
    try:
        from industry_mapper import RecruitmentIndustryMapper
        mapper = RecruitmentIndustryMapper()
        
        # 标准化行业分类
        df['标准行业'] = mapper.batch_map(df['行业'])
        
        print(f"行业标准化完成，共有{df['标准行业'].nunique()}个标准行业分类")
        
    except ImportError:
        print("未找到industry_mapper模块，使用原始行业数据")
        df['标准行业'] = df['行业']
    
    matplotlib.rc("font",family='FangSong')
    
    # 检查地区信息
    if '地区' not in df.columns:
        potential_city_columns = ['工作地点', '地点', '城市', '所在地', '区域']
        city_column = None
        
        for col in potential_city_columns:
            if col in df.columns:
                city_column = col
                break
        
        if city_column:
            df['地区'] = df[city_column]
        else:
            print("无法找到地区信息，跳过行业城市对比分析")
            return
    
    # 清洗地区数据
    df['地区'] = df['地区'].fillna('未知')
    
    def clean_city_name(city_str):
        if pd.isna(city_str):
            return '未知'
        city_str = str(city_str)
        patterns = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', 
                   '重庆', '西安', '天津', '苏州', '长沙', '郑州', '青岛', '宁波',
                   '厦门', '福州', '沈阳', '大连', '长春', '哈尔滨', '济南', '合肥',
                   '南昌', '南宁', '昆明', '贵阳', '太原', '石家庄', '兰州', '银川',
                   '西宁', '乌鲁木齐', '海口', '三亚', '珠海', '东莞', '佛山', '中山']
        
        for pattern in patterns:
            if pattern in city_str:
                return pattern
        
        if len(city_str) <= 4:
            return city_str
        
        return '其他'
    
    df['城市_清洗'] = df['地区'].apply(clean_city_name)
    
    # 过滤有效城市
    city_counts = df['城市_清洗'].value_counts()
    city_counts = city_counts[~city_counts.index.isin(['未知', '其他'])]
    
    if len(city_counts) < 3:
        print(f"只有{len(city_counts)}个有效城市数据，无法进行行业城市对比分析")
        return
    
    # 选择前6个城市
    top_cities = city_counts.head(6).index.tolist()
    
    # 选择前8个行业
    top_industries = df['标准行业'].value_counts().head(8).index.tolist()
    
    # 创建热力图数据
    heatmap_data = pd.DataFrame(index=top_cities, columns=top_industries)
    
    for city in top_cities:
        for industry in top_industries:
            # 计算该城市该行业的平均薪资
            mask = (df['城市_清洗'] == city) & (df['标准行业'] == industry)
            if mask.sum() > 0:
                avg_salary = df.loc[mask, '薪资_月平均K'].mean()
                heatmap_data.loc[city, industry] = avg_salary
            else:
                heatmap_data.loc[city, industry] = np.nan
    
    # 创建热力图
    plt.figure(figsize=(16, 10))
    
    # 使用seaborn创建热力图
    sns.heatmap(heatmap_data.astype(float), annot=True, fmt='.1f', cmap='YlOrRd', 
                linewidths=0.5, linecolor='gray', cbar_kws={'label': '平均薪资 (K)'})
    
    plt.title('5. 各城市不同行业平均薪资热力图', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('标准行业', fontsize=14)
    plt.ylabel('城市', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('city/5_各城市不同行业平均薪资热力图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: city/5_各城市不同行业平均薪资热力图.png")
    
    # 创建城市行业就业结构雷达图
    create_city_industry_radar_chart(df, top_cities)
    
    return heatmap_data

def create_city_industry_radar_chart(df, cities):
    """创建城市行业就业结构雷达图"""
    if len(cities) < 2:
        print("城市数量不足，无法创建雷达图")
        return
    
    # 选择前6个行业
    top_industries = df['标准行业'].value_counts().head(6).index.tolist()
    
    # 准备雷达图数据
    radar_data = []
    
    for city in cities[:4]:  # 最多显示4个城市
        city_data = []
        for industry in top_industries:
            # 计算该城市该行业的岗位占比
            city_total = (df['城市_清洗'] == city).sum()
            industry_city_count = ((df['城市_清洗'] == city) & (df['标准行业'] == industry)).sum()
            
            if city_total > 0:
                percentage = industry_city_count / city_total * 100
            else:
                percentage = 0
            
            city_data.append(percentage)
        
        radar_data.append({
            'city': city,
            'values': city_data
        })
    
    # 创建雷达图
    angles = np.linspace(0, 2*np.pi, len(top_industries), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(radar_data)))
    
    for idx, city_data in enumerate(radar_data):
        values = city_data['values']
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=city_data['city'], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # 设置角度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_industries, fontsize=11)
    
    ax.set_title('6. 城市行业就业结构对比雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('city/6_城市行业就业结构对比雷达图.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("已保存: city/6_城市行业就业结构对比雷达图.png")

# ==================== 新增：城市对比统计保存功能 ====================
def save_city_comparison_results(df, city_salary_stats=None, heatmap_data=None):
    """保存城市对比分析结果"""
    
    # 确保city文件夹存在
    os.makedirs('city', exist_ok=True)
    
    # 创建城市清洗列（如果不存在）
    if '城市_清洗' not in df.columns:
        # 简单清洗地区数据
        def clean_city(city_str):
            if pd.isna(city_str):
                return '未知'
            city_str = str(city_str)
            patterns = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', 
                       '重庆', '西安', '天津', '苏州', '长沙', '郑州', '青岛', '宁波']
            for pattern in patterns:
                if pattern in city_str:
                    return pattern
            return '其他'
        
        if '地区' in df.columns:
            df['城市_清洗'] = df['地区'].apply(clean_city)
        else:
            print("无法保存城市对比结果：缺少地区数据")
            return
    
    # 1. 保存城市基本统计
    city_stats = df.groupby('城市_清洗').agg({
        '薪资_月平均K': ['count', 'mean', 'std', 'median', 'min', 'max'],
        '经验要求': lambda x: x.mode()[0] if not x.mode().empty else '未知',
        '学历要求': lambda x: x.mode()[0] if not x.mode().empty else '未知'
    }).round(2)
    
    # 重命名列
    city_stats.columns = [
        '岗位数量', '平均薪资_K', '薪资标准差', '薪资中位数_K', 
        '最低薪资_K', '最高薪资_K', '最常见经验要求', '最常见学历要求'
    ]
    
    city_stats = city_stats.sort_values('平均薪资_K', ascending=False)
    city_stats.to_csv('city/city_comparison_summary.csv', encoding='utf-8-sig')
    print(f"城市对比统计已保存到: city/city_comparison_summary.csv")
    
    # 2. 保存各城市行业分布
    if '行业' in df.columns:
        city_industry_dist = df.groupby(['城市_清洗', '行业']).size().unstack(fill_value=0)
        city_industry_dist.to_csv('city/city_industry_distribution.csv', encoding='utf-8-sig')
        print(f"城市行业分布已保存到: city/city_industry_distribution.csv")
    
    # 3. 保存城市薪资详细数据
    city_salary_details = df[['城市_清洗', '行业', '薪资_月平均K', '经验要求', '学历要求']].copy()
    city_salary_details.to_csv('city/city_salary_details.csv', index=False, encoding='utf-8-sig')
    print(f"城市薪资详细数据已保存到: city/city_salary_details.csv")
    
    # 4. 保存热力图数据（如果存在）
    if heatmap_data is not None:
        heatmap_data.to_csv('city/city_industry_heatmap_data.csv', encoding='utf-8-sig')
        print(f"城市行业热力图数据已保存到: city/city_industry_heatmap_data.csv")
    
    # 5. 生成城市对比报告
    generate_city_comparison_report(city_stats)

def generate_city_comparison_report(city_stats):
    """生成城市对比报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("城市就业市场对比分析报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"分析城市数量: {len(city_stats)}")
    report_lines.append("")
    
    # 最高薪资城市
    top_salary_city = city_stats.iloc[0]
    report_lines.append("1. 最高薪资城市:")
    report_lines.append(f"   {city_stats.index[0]}: {top_salary_city['平均薪资_K']}K/月")
    report_lines.append(f"   岗位数量: {int(top_salary_city['岗位数量'])}")
    report_lines.append(f"   薪资范围: {top_salary_city['最低薪资_K']}K - {top_salary_city['最高薪资_K']}K")
    report_lines.append("")
    
    # 最多岗位城市
    most_jobs_city = city_stats.sort_values('岗位数量', ascending=False).iloc[0]
    most_jobs_city_name = city_stats.sort_values('岗位数量', ascending=False).index[0]
    report_lines.append("2. 最多岗位城市:")
    report_lines.append(f"   {most_jobs_city_name}: {int(most_jobs_city['岗位数量'])}个岗位")
    report_lines.append(f"   平均薪资: {most_jobs_city['平均薪资_K']}K/月")
    report_lines.append("")
    
    # 薪资差异分析
    if len(city_stats) >= 2:
        max_salary = city_stats['平均薪资_K'].max()
        min_salary = city_stats['平均薪资_K'].min()
        salary_gap = max_salary - min_salary
        salary_ratio = max_salary / min_salary
        
        report_lines.append("3. 薪资差异分析:")
        report_lines.append(f"   最高薪资: {max_salary}K/月 ({city_stats['平均薪资_K'].idxmax()})")
        report_lines.append(f"   最低薪资: {min_salary}K/月 ({city_stats['平均薪资_K'].idxmin()})")
        report_lines.append(f"   薪资差距: {salary_gap:.1f}K")
        report_lines.append(f"   薪资倍率: {salary_ratio:.2f}倍")
        report_lines.append("")
    
    # 城市梯队划分
    if len(city_stats) >= 5:
        report_lines.append("4. 城市梯队划分:")
        # 按薪资分位点划分
        q1 = city_stats['平均薪资_K'].quantile(0.8)  # 前20%
        q2 = city_stats['平均薪资_K'].quantile(0.5)  # 中位数
        
        tier1_cities = city_stats[city_stats['平均薪资_K'] >= q1].index.tolist()
        tier2_cities = city_stats[(city_stats['平均薪资_K'] >= q2) & (city_stats['平均薪资_K'] < q1)].index.tolist()
        tier3_cities = city_stats[city_stats['平均薪资_K'] < q2].index.tolist()
        
        report_lines.append(f"   第一梯队(高薪资城市, ≥{q1:.1f}K): {', '.join(tier1_cities)}")
        report_lines.append(f"   第二梯队(中等薪资城市, {q2:.1f}K-{q1:.1f}K): {', '.join(tier2_cities)}")
        report_lines.append(f"   第三梯队(较低薪资城市, <{q2:.1f}K): {', '.join(tier3_cities)}")
    
    # 保存报告
    report_text = '\n'.join(report_lines)
    with open('city/city_comparison_report.txt', 'w', encoding='utf-8-sig') as f:
        f.write(report_text)
    
    print(f"城市对比报告已保存到: city/city_comparison_report.txt")
    print("\n" + report_text)

# ==================== 新增：经验与学历的城市对比分析 ====================
def create_experience_education_city_comparison(df):
    """创建经验与学历的城市对比分析"""
    matplotlib.rc("font",family='FangSong')
    
    # 确保city文件夹存在
    os.makedirs('city', exist_ok=True)
    
    # 检查地区信息
    if '地区' not in df.columns:
        print("缺少地区信息，跳过经验学历城市对比分析")
        return
    
    # 清洗城市数据
    if '城市_清洗' not in df.columns:
        df['地区'] = df['地区'].fillna('未知')
        
        def clean_city_name(city_str):
            if pd.isna(city_str):
                return '未知'
            city_str = str(city_str)
            patterns = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', 
                       '重庆', '西安', '天津', '苏州', '长沙', '郑州', '青岛', '宁波']
            for pattern in patterns:
                if pattern in city_str:
                    return pattern
            return '其他'
        
        df['城市_清洗'] = df['地区'].apply(clean_city_name)
    
    # 过滤有效城市
    city_counts = df['城市_清洗'].value_counts()
    city_counts = city_counts[~city_counts.index.isin(['未知', '其他'])]
    
    if len(city_counts) < 3:
        print(f"只有{len(city_counts)}个有效城市数据，无法进行经验学历城市对比分析")
        return
    
    # 选择前6个城市
    top_cities = city_counts.head(6).index.tolist()
    
    # 创建子图 - 现在每个子图单独保存
    
    # 子图1: 各城市经验要求分布
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    exp_data_list = []
    exp_labels = []
    
    for city in top_cities:
        city_df = df[df['城市_清洗'] == city]
        if len(city_df) > 0:
            exp_dist = city_df['经验要求'].value_counts(normalize=True) * 100
            
            # 定义经验顺序
            exp_order = ['经验不限', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
            
            # 按顺序整理数据
            city_exp_data = []
            for exp in exp_order:
                if exp in exp_dist.index:
                    city_exp_data.append(exp_dist[exp])
                else:
                    city_exp_data.append(0)
            
            exp_data_list.append(city_exp_data)
            exp_labels.append(city)
    
    if exp_data_list:
        x = np.arange(len(exp_order))
        width = 0.12
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_cities)))
        
        for i, (city_data, color) in enumerate(zip(exp_data_list, colors)):
            ax1.bar(x + i*width, city_data, width, label=exp_labels[i], color=color)
        
        ax1.set_title('7. 各城市经验要求分布对比', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('经验要求', fontsize=12)
        ax1.set_ylabel('占比 (%)', fontsize=12)
        ax1.set_xticks(x + width*(len(top_cities)-1)/2)
        ax1.set_xticklabels(exp_order, rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
    else:
        ax1.text(0.5, 0.5, '没有经验要求数据', ha='center', va='center', fontsize=12)
        ax1.set_title('7. 各城市经验要求分布对比', fontsize=16, fontweight='bold', pad=20)
        ax1.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/7_各城市经验要求分布对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("已保存: city/7_各城市经验要求分布对比.png")
    
    # 子图2: 各城市学历要求分布
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    edu_data_list = []
    edu_labels = []
    
    for city in top_cities:
        city_df = df[df['城市_清洗'] == city]
        if len(city_df) > 0:
            edu_dist = city_df['学历要求'].value_counts(normalize=True) * 100
            
            # 定义学历顺序
            edu_order = ['学历不限', '初中及以下', '高中', '本科', '硕士', '博士']
            
            # 按顺序整理数据
            city_edu_data = []
            for edu in edu_order:
                if edu in edu_dist.index:
                    city_edu_data.append(edu_dist[edu])
                else:
                    city_edu_data.append(0)
            
            edu_data_list.append(city_edu_data)
            edu_labels.append(city)
    
    if edu_data_list:
        x = np.arange(len(edu_order))
        width = 0.12
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_cities)))
        
        for i, (city_data, color) in enumerate(zip(edu_data_list, colors)):
            ax2.bar(x + i*width, city_data, width, label=edu_labels[i], color=color)
        
        ax2.set_title('8. 各城市学历要求分布对比', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('学历要求', fontsize=12)
        ax2.set_ylabel('占比 (%)', fontsize=12)
        ax2.set_xticks(x + width*(len(top_cities)-1)/2)
        ax2.set_xticklabels(edu_order, rotation=45, ha='right', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, '没有学历要求数据', ha='center', va='center', fontsize=12)
        ax2.set_title('8. 各城市学历要求分布对比', fontsize=16, fontweight='bold', pad=20)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/8_各城市学历要求分布对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("已保存: city/8_各城市学历要求分布对比.png")
    
    # 子图3: 各城市平均公司规模对比
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    city_size_data = []
    size_labels = []
    
    for city in top_cities:
        city_df = df[df['城市_清洗'] == city]
        if len(city_df) > 0 and '公司规模_数值' in city_df.columns:
            # 过滤掉NaN值
            valid_sizes = city_df['公司规模_数值'].dropna()
            if len(valid_sizes) > 0:
                avg_size = valid_sizes.mean()
                city_size_data.append(avg_size)
                size_labels.append(city)
    
    if city_size_data:
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(city_size_data)))
        bars = ax3.bar(range(len(city_size_data)), city_size_data, color=colors)
        ax3.set_title('9. 各城市平均公司规模对比', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('城市', fontsize=12)
        ax3.set_ylabel('平均公司规模 (人)', fontsize=12)
        ax3.set_xticks(range(len(city_size_data)))
        ax3.set_xticklabels(size_labels, rotation=45, ha='right', fontsize=10)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # 检查height是否为NaN
            if pd.notna(height):
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, '没有公司规模数据', ha='center', va='center', fontsize=12)
        ax3.set_title('9. 各城市平均公司规模对比', fontsize=16, fontweight='bold', pad=20)
        ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/9_各城市平均公司规模对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("已保存: city/9_各城市平均公司规模对比.png")
    
    # 子图4: 各城市技能标签对比
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    top_skills_by_city = {}
    
    for city in top_cities:
        city_df = df[df['城市_清洗'] == city]
        if len(city_df) > 0:
            # 提取技能标签
            all_skills = []
            for skills in city_df['技能标签列表']:
                if isinstance(skills, str) and skills.startswith('[') and skills.endswith(']'):
                    try:
                        import ast
                        skills_list = ast.literal_eval(skills)
                    except:
                        skills_list = [s.strip().strip("'\"") for s in skills.strip('[]').split(',') if s.strip()]
                elif isinstance(skills, str):
                    skills_list = [s.strip() for s in skills.split(',') if s.strip()]
                elif isinstance(skills, list):
                    skills_list = skills
                else:
                    skills_list = []
                
                all_skills.extend([str(skill).strip() for skill in skills_list])
            
            # 统计技能频率
            if all_skills:
                skill_counts = Counter(all_skills)
                # 过滤掉空字符串和None
                valid_skill_counts = {skill: count for skill, count in skill_counts.items() 
                                     if skill and skill.strip()}
                if valid_skill_counts:
                    top_skills = Counter(valid_skill_counts).most_common(5)
                    top_skills_by_city[city] = [skill[0] for skill in top_skills]
    
    if top_skills_by_city:
        # 创建技能对比表格
        skill_table_data = []
        max_skills = max(len(skills) for skills in top_skills_by_city.values())
        
        for city in top_cities:
            if city in top_skills_by_city:
                city_skills = top_skills_by_city[city]
                # 补齐长度
                city_skills.extend([''] * (max_skills - len(city_skills)))
                skill_table_data.append([city] + city_skills)
        
        # 创建表格
        col_labels = ['城市'] + [f'技能{i+1}' for i in range(max_skills)]
        table = ax4.table(cellText=skill_table_data, colLabels=col_labels, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax4.set_title('10. 各城市热门技能标签对比', fontsize=16, fontweight='bold', pad=20)
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, '没有足够的技能标签数据', ha='center', va='center', fontsize=12)
        ax4.set_title('10. 各城市热门技能标签对比', fontsize=16, fontweight='bold', pad=20)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('city/10_各城市热门技能标签对比.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("已保存: city/10_各城市热门技能标签对比.png")

# ==================== 主程序调用示例 ====================
def run_city_comparison_analysis(cleaned_data_path):
    """运行城市对比分析的独立函数"""
    print("=" * 60)
    print("开始城市对比分析")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv(cleaned_data_path, encoding='utf-8-sig')
    print(f"加载数据: {df.shape}")
    
    # 基本城市对比分析
    print("\n1. 基本城市对比分析...")
    city_salary_stats = create_city_comparison_analysis(df)
    
    # 基于行业映射的城市对比分析
    print("\n2. 基于行业映射的城市对比分析...")
    heatmap_data = create_industry_based_city_comparison(df)
    
    # 经验与学历城市对比分析
    print("\n3. 经验与学历城市对比分析...")
    create_experience_education_city_comparison(df)
    
    # 保存结果
    print("\n4. 保存分析结果...")
    save_city_comparison_results(df, city_salary_stats, heatmap_data)
    
    print("\n" + "=" * 60)
    print("城市对比分析完成!")
    print("所有图片已保存到 'city' 文件夹")
    print("=" * 60)

# 如果独立运行
if __name__ == "__main__":
    # 指定清洗后的数据文件路径
    matplotlib.rc("font", family='FangSong')
    cleaned_file = "cleaned_recruitment_data_all.csv"
    
    # 运行城市对比分析
    run_city_comparison_analysis(cleaned_file)