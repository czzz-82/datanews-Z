# ==================== 大规模数据处理系统 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')
from map import MajorIndustryMapper
import matplotlib
import os

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试多种字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'simhei.ttf',  # 当前目录下的字体文件
        ]
        
        for path in font_paths:
            try:
                font_manager.fontManager.addfont(path)
                font_name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"成功设置字体: {font_name}")
                return font_name
            except:
                continue
        
        # 回退方案
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return "SimHei"
    
    except Exception as e:
        print(f"字体设置失败: {e}")
        return None

setup_chinese_font()
sns.set_style("whitegrid")

class UniversityDataProcessor:
    """高校数据处理类"""
    
    def __init__(self, data_paths=None):
        """初始化处理器"""
        self.data_paths = data_paths or []
        self.merged_df = None
        self.summary_stats = {}
        
    def load_multiple_files(self, file_patterns=None):
        """加载多个数据文件"""
        import glob
        
        all_data = []
        
        if file_patterns:
            for pattern in file_patterns:
                files = glob.glob(pattern)
                for file in files:
                    try:
                        if file.endswith('.xlsx'):
                            # 读取所有sheet
                            xls = pd.ExcelFile(file)
                            for sheet_name in xls.sheet_names:
                                if '就业率' in sheet_name or 'employment' in sheet_name.lower():
                                    df = pd.read_excel(file, sheet_name=sheet_name)
                                    df['source_file'] = file
                                    all_data.append(df)
                        elif file.endswith('.csv'):
                            df = pd.read_csv(file, encoding='utf-8-sig')
                            df['source_file'] = file
                            all_data.append(df)
                    except Exception as e:
                        print(f"加载文件失败 {file}: {e}")
        
        if all_data:
            self.merged_df = pd.concat(all_data, ignore_index=True)
            print(f"成功加载 {len(all_data)} 个文件，总数据量: {len(self.merged_df):,}")
            return True
        else:
            print("未找到数据文件")
            return False
    
    def clean_data(self):
        """清洗数据"""
        if self.merged_df is None:
            print("请先加载数据")
            return
        
        print("开始数据清洗...")
        
        # 重命名列（统一格式）
        column_mapping = {
            'school_name': '学校名称',
            'year': '年份',
            'data_type': '数据类型',
            'education': '学历',
            'major': '专业名称',
            'graduate_number': '毕业生人数',
            'employment_number': '就业人数',
            'employment_rate': '就业率',
            'trade_name': '行业名称',
            'flow_number': '流向人数',
            'flow_rate': '流向比例',
        }
        
        # 应用列名映射
        for eng, chi in column_mapping.items():
            if chi in self.merged_df.columns:
                self.merged_df[eng] = self.merged_df[chi]
        
        # 确保必要列存在
        required_cols = ['school_name', 'major', 'graduate_number', 'employment_number']
        missing_cols = [col for col in required_cols if col not in self.merged_df.columns]
        
        if missing_cols:
            print(f"警告: 缺少必要的列: {missing_cols}")
        
        # 数据清洗
        original_size = len(self.merged_df)
        
        # 1. 去除重复值
        self.merged_df = self.merged_df.drop_duplicates()
        
        # 2. 处理缺失值
        if 'employment_rate' not in self.merged_df.columns:
            # 计算就业率
            self.merged_df['employment_rate'] = (
                self.merged_df['employment_number'] / 
                self.merged_df['graduate_number']
            ) * 100
        
        # 3. 处理异常值
        self.merged_df = self.merged_df[
            (self.merged_df['graduate_number'] > 0) &
            (self.merged_df['employment_rate'] >= 0) &
            (self.merged_df['employment_rate'] <= 100)
        ]
        
        # 4. 标准化学历字段
        if 'education' in self.merged_df.columns:
            education_mapping = {
                '本科': '本科',
                '学士': '本科',
                '本科毕业生': '本科',
                '本科毕业': '本科',
                '硕士': '硕士',
                '硕士研究生': '硕士',
                '硕士毕业': '硕士',
                '博士': '博士',
                '博士研究生': '博士',
                '专科': '专科',
                '高职': '专科',
            }
            self.merged_df['education'] = self.merged_df['education'].replace(education_mapping)
        
        cleaned_size = len(self.merged_df)
        print(f"数据清洗完成: 原始 {original_size:,} 条 -> 清洗后 {cleaned_size:,} 条")
        
        return self.merged_df
    
    def analyze_by_school(self):
        """按学校分析"""
        if self.merged_df is None:
            print("请先加载数据")
            return
        
        print("\n按学校分析:")
        
        school_stats = self.merged_df.groupby('school_name').agg({
            'graduate_number': 'sum',
            'employment_number': 'sum',
            'major': 'nunique'
        }).reset_index()
        
        school_stats['就业率'] = (school_stats['employment_number'] / school_stats['graduate_number']) * 100
        school_stats = school_stats.sort_values('就业率', ascending=False)
        
        # 保存结果
        school_stats.to_csv('学校就业率排名.csv', index=False, encoding='utf-8-sig')
        print("✓ 学校就业率排名已保存")
        
        return school_stats
    
    def analyze_by_region(self):
        """按地区分析（需要学校地区信息）"""
        if self.merged_df is None:
            print("请先加载数据")
            return
        
        # 学校地区映射（示例，需要实际数据）
        school_region_map = {
            # 北京地区
            '北京大学': '北京', '清华大学': '北京', '中国人民大学': '北京',
            '北京师范大学': '北京', '北京航空航天大学': '北京', '北京理工大学': '北京',
            # 上海地区
            '复旦大学': '上海', '上海交通大学': '上海', '同济大学': '上海',
            # 江苏地区
            '南京大学': '江苏', '东南大学': '江苏', '南京理工大学': '江苏',
            # 湖北地区
            '武汉大学': '湖北', '华中科技大学': '湖北',
            # 广东地区
            '中山大学': '广东', '华南理工大学': '广东', '暨南大学': '广东',
        }
        
        if 'school_name' in self.merged_df.columns:
            self.merged_df['region'] = self.merged_df['school_name'].map(school_region_map)
            
            region_stats = self.merged_df.groupby('region').agg({
                'graduate_number': 'sum',
                'employment_number': 'sum',
                'school_name': 'nunique'
            }).reset_index()
            
            region_stats['就业率'] = (region_stats['employment_number'] / region_stats['graduate_number']) * 100
            region_stats = region_stats.sort_values('就业率', ascending=False)
            
            region_stats.to_csv('地区就业率统计.csv', index=False, encoding='utf-8-sig')
            print("✓ 地区就业率统计已保存")
            
            return region_stats
        
        return None
    
    def generate_comprehensive_report(self, mapper=None):
        """生成综合分析报告"""
        print("\n生成综合分析报告...")
        
        if self.merged_df is None:
            print("请先加载数据")
            return
        
        # 基本统计
        total_stats = {
            '总毕业生数': self.merged_df['graduate_number'].sum(),
            '总就业人数': self.merged_df['employment_number'].sum(),
            '平均就业率': (self.merged_df['employment_number'].sum() / 
                        self.merged_df['graduate_number'].sum()) * 100,
            '涉及学校数': self.merged_df['school_name'].nunique(),
            '涉及专业数': self.merged_df['major'].nunique(),
        }
        
        # 学历分布
        if 'education' in self.merged_df.columns:
            edu_stats = self.merged_df.groupby('education').agg({
                'graduate_number': 'sum',
                'employment_number': 'sum'
            })
            edu_stats['就业率'] = (edu_stats['employment_number'] / edu_stats['graduate_number']) * 100
        
        # 热门专业Top 20
        major_stats = self.merged_df.groupby('major').agg({
            'graduate_number': 'sum',
            'employment_rate': 'mean',
            'school_name': 'nunique'
        }).sort_values('graduate_number', ascending=False).head(20)
        
        # 高就业率专业Top 20（毕业生>100人）
        high_rate_majors = self.merged_df[self.merged_df['graduate_number'] >= 100]
        high_rate_stats = high_rate_majors.groupby('major').agg({
            'graduate_number': 'sum',
            'employment_rate': 'mean'
        }).sort_values('employment_rate', ascending=False).head(20)
        
        # 保存报告
        with open('综合分析报告.txt', 'w', encoding='utf-8') as f:
            f.write("高校就业数据综合分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("一、总体概况\n")
            for key, value in total_stats.items():
                if '率' in key:
                    f.write(f"{key}: {value:.1f}%\n")
                else:
                    f.write(f"{key}: {value:,}\n")
            
            f.write("\n二、热门专业Top 20（按毕业生人数）\n")
            f.write("-"*60 + "\n")
            for i, (major, row) in enumerate(major_stats.iterrows(), 1):
                f.write(f"{i:2d}. {major:<25} {row['graduate_number']:>6,}人  就业率:{row['employment_rate']:5.1f}%  涉及学校:{int(row['school_name'])}\n")
            
            f.write("\n三、高就业率专业Top 20（毕业生≥100人）\n")
            f.write("-"*60 + "\n")
            for i, (major, row) in enumerate(high_rate_stats.iterrows(), 1):
                f.write(f"{i:2d}. {major:<25} {row['graduate_number']:>6,}人  就业率:{row['employment_rate']:5.1f}%\n")
        
        print("✓ 综合分析报告已保存到: 综合分析报告.txt")
        
        return total_stats, major_stats, high_rate_stats

def create_separate_visualizations(employment_rate_df, mapper=None):
    """创建分开的可视化图表，每张图单独保存"""
    print("\n创建分开的可视化图表...")
    
    # 创建可视化输出目录
    output_dir = "可视化图表"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 就业率分布直方图
    plt.figure(figsize=(10, 6))
    valid_rates = employment_rate_df['employment_rate'].dropna()
    plt.hist(valid_rates, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.title('1. 就业率分布直方图', fontsize=14, fontweight='bold')
    plt.xlabel('就业率 (%)')
    plt.ylabel('专业数量')
    plt.grid(True, alpha=0.3)
    
    # 添加统计线
    mean_rate = valid_rates.mean()
    median_rate = valid_rates.median()
    plt.axvline(mean_rate, color='red', linestyle='--', label=f'平均: {mean_rate:.1f}%')
    plt.axvline(median_rate, color='green', linestyle='--', label=f'中位数: {median_rate:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_就业率分布直方图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 1/9: 就业率分布直方图已保存")
    
    # 2. 不同学历就业率对比
    if 'education' in employment_rate_df.columns:
        plt.figure(figsize=(10, 6))
        edu_groups = employment_rate_df.groupby('education')['employment_rate'].mean().sort_values()
        edu_groups.plot(kind='bar', color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
        plt.title('2. 不同学历平均就业率', fontsize=14, fontweight='bold')
        plt.xlabel('学历')
        plt.ylabel('平均就业率 (%)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(edu_groups.values):
            plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/2_不同学历平均就业率.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 2/9: 不同学历平均就业率已保存")
    
    # 3. 专业类别就业率箱线图
    if mapper and 'major' in employment_rate_df.columns:
        plt.figure(figsize=(12, 7))
        # 获取专业类别
        employment_rate_df['major_category'] = employment_rate_df['major'].apply(mapper.infer_major_category)
        
        # 选择主要类别
        major_categories = employment_rate_df['major_category'].value_counts().head(6).index.tolist()
        filtered_df = employment_rate_df[employment_rate_df['major_category'].isin(major_categories)]
        
        # 创建箱线图
        category_data = []
        category_labels = []
        for category in major_categories:
            rates = filtered_df[filtered_df['major_category'] == category]['employment_rate'].dropna().values
            if len(rates) > 0:
                category_data.append(rates)
                category_labels.append(category)
        
        if category_data:
            box = plt.boxplot(category_data, labels=category_labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_data)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            plt.title('3. 不同学科门类就业率分布', fontsize=14, fontweight='bold')
            plt.xlabel('学科门类')
            plt.ylabel('就业率 (%)')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/3_不同学科门类就业率分布.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 3/9: 不同学科门类就业率分布已保存")
    
    # 4. 毕业生人数与就业率关系
    plt.figure(figsize=(10, 6))
    plt.scatter(employment_rate_df['graduate_number'], 
                employment_rate_df['employment_rate'],
                alpha=0.5, s=30, c='blue')
    plt.title('4. 毕业生人数与就业率关系', fontsize=14, fontweight='bold')
    plt.xlabel('毕业生人数')
    plt.ylabel('就业率 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_毕业生人数与就业率关系.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 4/9: 毕业生人数与就业率关系已保存")
    
    # 5. 热门专业就业率（Top 15）
    plt.figure(figsize=(12, 8))
    top_majors = employment_rate_df.groupby('major').agg({
        'graduate_number': 'sum',
        'employment_rate': 'mean'
    }).sort_values('graduate_number', ascending=False).head(15)
    
    y_pos = np.arange(len(top_majors))
    bars = plt.barh(y_pos, top_majors['employment_rate'])
    plt.yticks(y_pos, top_majors.index, fontsize=10)
    plt.title('5. 热门专业就业率 (Top 15)', fontsize=14, fontweight='bold')
    plt.xlabel('就业率 (%)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        grad_count = top_majors.iloc[i]['graduate_number']
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}% (n={grad_count})', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/5_热门专业就业率Top15.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 5/9: 热门专业就业率Top15已保存")
    
    # 6. 高就业率专业（毕业生≥50人）
    plt.figure(figsize=(12, 8))
    filtered = employment_rate_df[employment_rate_df['graduate_number'] >= 50]
    high_rate = filtered.groupby('major')['employment_rate'].mean().sort_values(ascending=False).head(15)
    
    y_pos = np.arange(len(high_rate))
    bars = plt.barh(y_pos, high_rate.values)
    plt.yticks(y_pos, high_rate.index, fontsize=10)
    plt.title('6. 高就业率专业 (毕业生≥50人)', fontsize=14, fontweight='bold')
    plt.xlabel('就业率 (%)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_高就业率专业毕业生≥50人.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 6/9: 高就业率专业毕业生≥50人已保存")
    
    # 7. 行业就业率分析（如果已映射）
    if 'industry' in employment_rate_df.columns:
        plt.figure(figsize=(12, 8))
        industry_stats = employment_rate_df.groupby('industry').agg({
            'graduate_number': 'sum',
            'employment_rate': 'mean'
        }).sort_values('graduate_number', ascending=False).head(10)
        
        y_pos = np.arange(len(industry_stats))
        width = 0.35
        
        bars1 = plt.barh(y_pos - width/2, industry_stats['graduate_number'], 
                        width, label='毕业生人数', alpha=0.7)
        bars2 = plt.barh(y_pos + width/2, industry_stats['employment_rate'], 
                        width, label='就业率', alpha=0.7)
        
        plt.yticks(y_pos, industry_stats.index, fontsize=9)
        plt.title('7. 主要行业就业情况', fontsize=14, fontweight='bold')
        plt.xlabel('数量 / 百分比')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/7_主要行业就业情况.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 7/9: 主要行业就业情况已保存")
    
    # 8. 就业率随时间变化（如果有多年数据）
    if 'year' in employment_rate_df.columns and employment_rate_df['year'].nunique() > 1:
        plt.figure(figsize=(10, 6))
        yearly_stats = employment_rate_df.groupby('year').agg({
            'graduate_number': 'sum',
            'employment_rate': 'mean'
        }).reset_index()
        
        plt.plot(yearly_stats['year'], yearly_stats['employment_rate'], 
                marker='o', linewidth=2, markersize=8)
        plt.title('8. 就业率年度变化趋势', fontsize=14, fontweight='bold')
        plt.xlabel('年份')
        plt.ylabel('平均就业率 (%)')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for x, y in zip(yearly_stats['year'], yearly_stats['employment_rate']):
            plt.text(x, y + 0.5, f'{y:.1f}%', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/8_就业率年度变化趋势.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 8/9: 就业率年度变化趋势已保存")
    
    # 9. 学校就业率分布
    if 'school_name' in employment_rate_df.columns:
        plt.figure(figsize=(12, 8))
        school_stats = employment_rate_df.groupby('school_name').agg({
            'graduate_number': 'sum',
            'employment_rate': 'mean'
        }).sort_values('employment_rate', ascending=False).head(10)
        
        y_pos = np.arange(len(school_stats))
        bars = plt.barh(y_pos, school_stats['employment_rate'])
        plt.yticks(y_pos, school_stats.index, fontsize=9)
        plt.title('9. 高就业率学校Top 10', fontsize=14, fontweight='bold')
        plt.xlabel('就业率 (%)')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            grad_count = school_stats.iloc[i]['graduate_number']
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}% (n={grad_count})', 
                    ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/9_高就业率学校Top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 9/9: 高就业率学校Top10已保存")
    
    print(f"\n✓ 所有可视化图表已保存到 '{output_dir}' 目录下")

# 主程序 - 大规模数据处理
if __name__ == "__main__":
    print("="*80)
    print("大规模高校就业数据处理系统")
    print("="*80)
    matplotlib.rc("font",family='FangSong')
    # 1. 创建处理器
    processor = UniversityDataProcessor()
    
    # 2. 加载数据文件（支持通配符）
    # 示例：加载所有Excel和CSV文件
    file_patterns = [
        '高校就业详细数据.xlsx',
    ]
    
    success = processor.load_multiple_files(file_patterns)
    
    if success:
        # 3. 清洗数据
        cleaned_df = processor.clean_data()
        
        # 4. 创建专业映射器
        mapper = MajorIndustryMapper()
        
        # 应用专业映射
        if 'major' in cleaned_df.columns:
            cleaned_df['industry'] = cleaned_df['major'].apply(mapper.map_major)
        
        # 5. 按学校分析
        school_stats = processor.analyze_by_school()
        
        # 6. 按地区分析
        region_stats = processor.analyze_by_region()
        
        # 7. 生成报告
        total_stats, major_stats, high_rate_stats = processor.generate_comprehensive_report(mapper)
        
        # 8. 创建分开的可视化
        create_separate_visualizations(cleaned_df, mapper)
        
        # 9. 保存处理后的数据
        cleaned_df.to_csv('高校就业数据_清洗后.csv', index=False, encoding='utf-8-sig')
        print("✓ 清洗后的数据已保存到: 高校就业数据_清洗后.csv")
        
        print("\n" + "="*80)
        print("大规模数据处理完成！")
        print("="*80)
        print("\n生成的文件列表:")
        print("1. 高校就业数据_清洗后.csv - 清洗合并后的完整数据")
        print("2. 学校就业率排名.csv - 按学校排名的就业率")
        print("3. 地区就业率统计.csv - 按地区统计的就业率")
        print("4. 综合分析报告.txt - 文本分析报告")
        print("5. 可视化图表/ - 包含9张单独的可视化图表")
        print("   1_就业率分布直方图.png")
        print("   2_不同学历平均就业率.png")
        print("   3_不同学科门类就业率分布.png")
        print("   4_毕业生人数与就业率关系.png")
        print("   5_热门专业就业率Top15.png")
        print("   6_高就业率专业毕业生≥50人.png")
        print("   7_主要行业就业情况.png")
        print("   8_就业率年度变化趋势.png")
        print("   9_高就业率学校Top10.png")
        print("6. 专业行业映射表.json - 专业-行业映射关系")
    else:
        print("请将数据文件放在当前目录或指定目录中")