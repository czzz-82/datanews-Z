# ==================== 修复版：数据集成与高级分析 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')
import matplotlib
from map import MajorIndustryMapper
from industry_mapper import RecruitmentIndustryMapper  # 导入新的映射器
import os

# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    try:
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'simhei.ttf',
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
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return "SimHei"
    
    except Exception as e:
        print(f"字体设置失败: {e}")
        return None

setup_chinese_font()
sns.set_style("whitegrid")

class AdvancedIntegrationAnalyzer:
    """高级数据集成分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.university_df = None
        self.market_df = None
        self.integrated_df = None
        self.industry_mapper = RecruitmentIndustryMapper()  # 新增行业映射器
        
    def load_data(self, university_file, market_file):
        """加载数据"""
        print("正在加载数据...")
        
        try:
            # 加载高校数据
            if university_file.endswith('.xlsx'):
                # 尝试读取所有sheet
                xls = pd.ExcelFile(university_file)
                sheet_names = xls.sheet_names
                print(f"可用sheet: {sheet_names}")
                
                # 查找包含就业数据的sheet
                for sheet in sheet_names:
                    if any(keyword in sheet for keyword in ['就业', 'employment', '毕业生']):
                        self.university_df = pd.read_excel(university_file, sheet_name=sheet)
                        print(f"使用sheet: {sheet}")
                        break
                
                # 如果没找到，使用第一个sheet
                if self.university_df is None:
                    self.university_df = pd.read_excel(university_file, sheet_name=0)
                    
            elif university_file.endswith('.csv'):
                self.university_df = pd.read_csv(university_file, encoding='utf-8-sig')
            
            print(f"高校数据: {self.university_df.shape}")
            print(f"高校数据列: {self.university_df.columns.tolist()}")
            
            # 加载市场数据
            self.market_df = pd.read_csv(market_file, encoding='utf-8-sig')
            print(f"市场数据: {self.market_df.shape}")
            print(f"市场数据列: {self.market_df.columns.tolist()}")
            
            # 显示市场数据前几行以了解数据结构
            print("\n市场数据预览:")
            print(self.market_df.head())
            
            return self.university_df, self.market_df
            
        except Exception as e:
            print(f"加载数据出错: {e}")
            return None, None
    
    def clean_salary_data(self, salary_data):
        """清洗薪资数据，转换为千元单位"""
        cleaned_salaries = []
        
        for salary in salary_data:
            if pd.isna(salary):
                cleaned_salaries.append(0)
                continue
                
            # 如果是字符串，提取数字
            if isinstance(salary, str):
                import re
                
                # 移除空格和特殊字符
                salary_str = str(salary).replace(' ', '').replace(',', '').replace('，', '')
                
                # 检查是否是范围格式（如"5-10K"）
                if '-' in salary_str and ('K' in salary_str.upper() or '千' in salary_str):
                    # 提取范围中的数字
                    numbers = re.findall(r'\d+\.?\d*', salary_str)
                    if len(numbers) >= 2:
                        # 计算平均值
                        avg_salary = (float(numbers[0]) + float(numbers[1])) / 2
                        cleaned_salaries.append(avg_salary)
                        continue
                
                # 检查是否是月薪格式（带K或千）
                if 'K' in salary_str.upper() or '千' in salary_str:
                    # 提取数字
                    numbers = re.findall(r'\d+\.?\d*', salary_str)
                    if numbers:
                        cleaned_salaries.append(float(numbers[0]))
                        continue
                
                # 检查是否是月薪格式（带万）
                if '万' in salary_str:
                    numbers = re.findall(r'\d+\.?\d*', salary_str)
                    if numbers:
                        # 万转换为K（1万=10K）
                        cleaned_salaries.append(float(numbers[0]) * 10)
                        continue
                
                # 检查是否是年薪格式
                if '年' in salary_str:
                    numbers = re.findall(r'\d+\.?\d*', salary_str)
                    if numbers:
                        # 年薪转换为月薪K（年薪/12）
                        yearly_salary = float(numbers[0])
                        # 如果年薪单位是万，先转换为K
                        if '万' in salary_str:
                            yearly_salary = yearly_salary * 10  # 万转换为K
                        monthly_salary = yearly_salary / 12
                        cleaned_salaries.append(monthly_salary)
                        continue
                
                # 尝试直接提取数字
                numbers = re.findall(r'\d+\.?\d*', salary_str)
                if numbers:
                    # 假设是月薪，检查数字是否合理
                    salary_value = float(numbers[0])
                    if salary_value < 1:  # 小于1可能是万为单位
                        cleaned_salaries.append(salary_value * 10)  # 万转K
                    elif salary_value > 100:  # 大于100可能是元为单位
                        cleaned_salaries.append(salary_value / 1000)  # 元转K
                    else:
                        cleaned_salaries.append(salary_value)
                else:
                    cleaned_salaries.append(0)
                    
            # 如果是数字
            elif isinstance(salary, (int, float)):
                # 检查数字范围是否合理
                if salary < 1:  # 小于1可能是万为单位
                    cleaned_salaries.append(salary * 10)  # 万转K
                elif salary > 100:  # 大于100可能是元为单位
                    cleaned_salaries.append(salary / 1000)  # 元转K
                else:
                    cleaned_salaries.append(salary)
            else:
                cleaned_salaries.append(0)
        
        return cleaned_salaries
    
    def prepare_university_data(self, mapper):
        """准备高校数据（使用比例方法）"""
        print("\n准备高校数据（比例方法）...")
        
        if self.university_df is None:
            print("高校数据为空")
            return None
        
        # 首先查看数据结构
        print("高校数据结构:")
        print(self.university_df.head())
        
        # 尝试查找专业列（可能有不同的列名）
        major_col = None
        for col in self.university_df.columns:
            if any(keyword in str(col) for keyword in ['major', '专业', '专业名称']):
                major_col = col
                break
        
        if major_col is None:
            print("未找到专业列，尝试使用第一列")
            major_col = self.university_df.columns[0]
        
        print(f"使用专业列: {major_col}")
        
        # 应用专业映射
        self.university_df['industry'] = self.university_df[major_col].apply(
            lambda x: mapper.map_major(x) if pd.notna(x) else '其他'
        )
        
        # 查找毕业生人数和就业人数列
        graduate_col = None
        employment_col = None
        
        for col in self.university_df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['graduate', '毕业生', '毕业人数']):
                graduate_col = col
            elif any(keyword in col_lower for keyword in ['employment', '就业', '就业人数']):
                employment_col = col
        
        if graduate_col is None or employment_col is None:
            print("警告: 未找到毕业生人数或就业人数列")
            # 尝试使用数值列
            numeric_cols = self.university_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                graduate_col = numeric_cols[0]
                employment_col = numeric_cols[1]
                print(f"使用数值列: {graduate_col}, {employment_col}")
            else:
                print("无法准备高校数据")
                return None
        
        # 按行业汇总
        university_by_industry = self.university_df.groupby('industry').agg({
            graduate_col: 'sum',
            employment_col: 'sum'
        }).reset_index()
        
        university_by_industry = university_by_industry.rename(columns={
            graduate_col: 'graduate_number',
            employment_col: 'employment_number'
        })
        
        # 计算总毕业生数
        total_graduates = university_by_industry['graduate_number'].sum()
        
        if total_graduates > 0:
            # 计算比例指标
            university_by_industry['graduate_ratio'] = (
                university_by_industry['graduate_number'] / total_graduates * 100
            )
            
            # 计算就业率
            university_by_industry['employment_rate'] = (
                university_by_industry['employment_number'] / 
                university_by_industry['graduate_number'] * 100
            )
            
            # 计算就业人数比例
            total_employed = university_by_industry['employment_number'].sum()
            university_by_industry['employed_ratio'] = (
                university_by_industry['employment_number'] / total_employed * 100
            )
            
            # 按毕业生比例排序
            university_by_industry = university_by_industry.sort_values('graduate_ratio', ascending=False)
            
            print(f"高校数据涉及 {len(university_by_industry)} 个行业")
            print("\n高校行业分布Top 10:")
            print(university_by_industry[['industry', 'graduate_ratio', 'employment_rate']].head(10))
            
            return university_by_industry
        else:
            print("总毕业生数为0，无法计算比例")
            return None
    
    def prepare_market_data(self):
        """准备市场数据（统一行业分类）"""
        print("\n准备市场数据（统一行业分类）...")
        
        if self.market_df is None:
            print("市场数据为空")
            return None
        
        # 查找行业列
        industry_col = None
        for col in self.market_df.columns:
            if any(keyword in str(col) for keyword in ['行业', 'industry', 'trade']):
                industry_col = col
                break
        
        if industry_col is None:
            print("警告: 未找到行业列")
            return None
        
        print(f"使用行业列: {industry_col}")
        
        # 应用行业映射，将招聘行业映射到标准行业
        self.market_df['standard_industry'] = self.industry_mapper.batch_map(self.market_df[industry_col])
        
        # 分析映射覆盖率
        coverage = self.industry_mapper.analyze_coverage(self.market_df[industry_col])
        print(f"\n行业映射覆盖率: {coverage['coverage_rate']:.1f}%")
        print(f"映射分布: {coverage['distribution'].to_dict()}")
        
        # 查找薪资列
        salary_col = None
        for col in self.market_df.columns:
            if any(keyword in str(col).lower() for keyword in ['薪资', 'salary', '工资', '月薪', '薪酬']):
                salary_col = col
                break
        
        if salary_col is None:
            print("警告: 未找到薪资列，检查所有列:")
            for col in self.market_df.columns:
                print(f"  - {col}")
            # 尝试使用可能的薪资列名
            possible_cols = [col for col in self.market_df.columns if any(keyword in col.lower() for keyword in ['pay', 'wage', '收入', '待遇'])]
            if possible_cols:
                salary_col = possible_cols[0]
                print(f"尝试使用列: {salary_col}")
        
        if salary_col:
            print(f"使用薪资列: {salary_col}")
            print(f"薪资数据类型: {self.market_df[salary_col].dtype}")
            print(f"薪资数据样例: {self.market_df[salary_col].head(10).tolist()}")
            
            # 清洗薪资数据
            print("\n清洗薪资数据...")
            cleaned_salaries = self.clean_salary_data(self.market_df[salary_col])
            self.market_df['薪资_月平均K'] = cleaned_salaries
            
            # 统计清洗后的薪资数据
            valid_salaries = self.market_df[self.market_df['薪资_月平均K'] > 0]['薪资_月平均K']
            print(f"有效薪资数据: {len(valid_salaries)} 条")
            print(f"薪资范围: {valid_salaries.min():.1f}K - {valid_salaries.max():.1f}K")
            print(f"平均薪资: {valid_salaries.mean():.1f}K")
            print(f"中位数薪资: {valid_salaries.median():.1f}K")
            
            # 显示薪资分布
            print("\n薪资分布:")
            for percentile in [10, 25, 50, 75, 90]:
                value = valid_salaries.quantile(percentile/100)
                print(f"  {percentile}% 分位数: {value:.1f}K")
        else:
            print("警告: 未找到薪资列，使用默认值5K")
            self.market_df['薪资_月平均K'] = 5.0  # 默认值
        
        # 按标准行业汇总
        print("\n按行业汇总数据...")
        market_by_industry = self.market_df.groupby('standard_industry').agg({
            'standard_industry': 'count',  # 岗位数量
            '薪资_月平均K': ['mean', 'median', 'std', 'count']  # 薪资统计
        }).reset_index()
        
        # 扁平化列名
        market_by_industry.columns = ['industry', 'job_count', 'avg_salary', 'median_salary', 'salary_std', 'salary_count']
        
        # 处理可能的NaN值
        market_by_industry = market_by_industry.fillna(0)
        
        # 计算总岗位数
        total_jobs = market_by_industry['job_count'].sum()
        
        if total_jobs > 0:
            # 计算比例指标
            market_by_industry['job_ratio'] = (
                market_by_industry['job_count'] / total_jobs * 100
            )
            
            # 按岗位比例排序
            market_by_industry = market_by_industry.sort_values('job_ratio', ascending=False)
            
            print(f"\n市场数据涉及 {len(market_by_industry)} 个标准行业")
            print("\n市场行业分布Top 15:")
            for i, row in market_by_industry.head(15).iterrows():
                print(f"  {row['industry']:<25} 岗位:{row['job_count']:>4}个 ({row['job_ratio']:>5.1f}%) 平均薪资:{row['avg_salary']:>5.1f}K")
            
            return market_by_industry
        else:
            print("总岗位数为0，无法计算比例")
            return None
    
    def calculate_competition_index(self, integrated_data):
        """计算竞争指数（基于高校就业数据的行业比例）"""
        print("\n计算竞争指数（基于高校就业数据）...")
        
        if integrated_data is None or 'graduate_ratio' not in integrated_data.columns:
            print("缺少高校数据或毕业生比例信息")
            return integrated_data
        
        try:
            # 方法1：毕业生比例越高，竞争越激烈
            max_grad_ratio = integrated_data['graduate_ratio'].max()
            if max_grad_ratio > 0:
                # 归一化到0-10范围
                integrated_data['competition_index_raw'] = (
                    integrated_data['graduate_ratio'] / max_grad_ratio * 10
                )
            else:
                integrated_data['competition_index_raw'] = 0
            
            # 方法2：结合岗位比例进行修正
            # 竞争指数 = 毕业生比例 / (岗位比例 + 0.01) 避免除零
            integrated_data['competition_index'] = np.where(
                integrated_data['job_ratio'] > 0,
                integrated_data['graduate_ratio'] / integrated_data['job_ratio'],
                integrated_data['graduate_ratio'] * 10  # 如果没有岗位，竞争指数更高
            )
            
            # 处理无穷大值
            max_comp = integrated_data['competition_index'].max()
            if max_comp == np.inf:
                # 找到有限的最大值
                finite_max = integrated_data[integrated_data['competition_index'] < np.inf]['competition_index'].max()
                integrated_data['competition_index'] = integrated_data['competition_index'].replace(
                    np.inf, finite_max * 2
                )
            
            # 归一化到0-10范围
            max_comp = integrated_data['competition_index'].max()
            if max_comp > 0:
                integrated_data['competition_index'] = (
                    integrated_data['competition_index'] / max_comp * 10
                )
            
            print(f"竞争指数计算完成，范围: [{integrated_data['competition_index'].min():.2f}, {integrated_data['competition_index'].max():.2f}]")
            
            # 添加竞争程度分类
            def classify_competition(score):
                if score < 2:
                    return '低竞争'
                elif score < 5:
                    return '中等竞争'
                elif score < 8:
                    return '高竞争'
                else:
                    return '极高竞争'
            
            integrated_data['competition_level'] = integrated_data['competition_index'].apply(classify_competition)
            
            # 显示竞争指数分布
            print("\n竞争指数分布:")
            comp_dist = integrated_data['competition_level'].value_counts()
            for level, count in comp_dist.items():
                print(f"  {level}: {count}个行业")
            
            return integrated_data
            
        except Exception as e:
            print(f"计算竞争指数出错: {e}")
            import traceback
            traceback.print_exc()
            return integrated_data
    
    def integrate_data(self, university_data, market_data):
        """集成高校和市场数据（基于比例）"""
        print("\n集成高校和市场数据...")
        
        if university_data is None or market_data is None:
            print("高校或市场数据为空，无法集成")
            return None
        
        print(f"高校数据行业数: {len(university_data)}")
        print(f"市场数据行业数: {len(market_data)}")
        
        # 显示高校数据
        print("\n高校数据前10个行业:")
        print(university_data[['industry', 'graduate_ratio', 'employment_rate']].head(10))
        
        print("\n市场数据前10个行业:")
        print(market_data[['industry', 'job_ratio', 'avg_salary']].head(10))
        
        # 合并数据 - 使用outer join确保所有行业都在
        try:
            integrated = pd.merge(
                university_data,
                market_data,
                left_on='industry',
                right_on='industry',
                how='outer',
                suffixes=('_uni', '_market')
            )
        except Exception as e:
            print(f"数据合并出错: {e}")
            # 尝试不同的合并策略
            integrated = pd.merge(
                university_data,
                market_data,
                left_on='industry',
                right_on='industry',
                how='outer'
            )
        
        print(f"合并后数据行业数: {len(integrated)}")
        
        # 确保所有列都是合适的类型
        for col in integrated.columns:
            if integrated[col].dtype.name == 'category':
                integrated[col] = integrated[col].astype(str)
        
        # 列出所有列
        print("合并后数据列:", integrated.columns.tolist())
        
        # 填充缺失值 - 分别处理数值列和非数值列
        numeric_cols = integrated.select_dtypes(include=[np.number]).columns
        non_numeric_cols = integrated.select_dtypes(exclude=[np.number]).columns
        
        print(f"数值列: {numeric_cols.tolist()}")
        print(f"非数值列: {non_numeric_cols.tolist()}")
        
        # 数值列填充0
        if len(numeric_cols) > 0:
            integrated[numeric_cols] = integrated[numeric_cols].fillna(0)
        
        # 非数值列填充空字符串
        if len(non_numeric_cols) > 0:
            integrated[non_numeric_cols] = integrated[non_numeric_cols].fillna('')
        
        # 计算供需匹配指标（基于比例）
        # 确保有足够的岗位比例数据
        if 'graduate_ratio' in integrated.columns and 'job_ratio' in integrated.columns:
            # 避免除以0
            integrated['supply_demand_ratio'] = np.where(
                integrated['job_ratio'] > 0,
                integrated['graduate_ratio'] / integrated['job_ratio'],
                0
            )
            
            # 处理无穷大
            integrated['supply_demand_ratio'] = integrated['supply_demand_ratio'].replace(
                [np.inf, -np.inf], 0
            )
            
            # 标准化供需比（0-100）
            max_ratio = integrated['supply_demand_ratio'].max()
            if max_ratio > 0:
                integrated['supply_demand_ratio_norm'] = (
                    integrated['supply_demand_ratio'] / max_ratio * 100
                )
            else:
                integrated['supply_demand_ratio_norm'] = 0
        
        # 计算综合匹配度（基于比例）
        def calculate_proportional_match_score(row):
            """基于比例计算匹配度分数"""
            score = 0
            
            # 就业率得分（0-40分）
            if 'employment_rate' in row and pd.notna(row['employment_rate']):
                employment_score = min(row['employment_rate'] * 0.4, 40)
                score += employment_score
            
            # 供需平衡得分（0-30分）
            if 'job_ratio' in row and 'graduate_ratio' in row:
                if row['job_ratio'] > 0 and row['graduate_ratio'] > 0:
                    # 理想状态是毕业生比例与岗位比例接近
                    balance_ratio = min(row['graduate_ratio'] / max(row['job_ratio'], 0.01), 3)
                    balance_score = 30 * np.exp(-abs(np.log(max(balance_ratio, 0.01))))
                    score += balance_score
            
            # 薪资水平得分（0-30分）
            if 'avg_salary' in row and pd.notna(row['avg_salary']) and row['avg_salary'] > 0:
                # 薪资得分：假设20K为满分
                salary_score = min(row['avg_salary'] / 20 * 30, 30)
                score += salary_score
            
            return min(score, 100)
        
        integrated['match_score'] = integrated.apply(calculate_proportional_match_score, axis=1)
        
        # 分类供需状态
        def classify_supply_demand(row):
            if 'job_ratio' not in row or row['job_ratio'] == 0:
                return '无市场需求'
            elif 'supply_demand_ratio' not in row or row['supply_demand_ratio'] == 0:
                return '数据不足'
            elif row['supply_demand_ratio'] < 0.3:
                return '严重供不应求'
            elif row['supply_demand_ratio'] < 0.7:
                return '供不应求'
            elif row['supply_demand_ratio'] < 1.5:
                return '供需平衡'
            elif row['supply_demand_ratio'] < 3:
                return '供过于求'
            else:
                return '严重供过于求'
        
        integrated['supply_demand_status'] = integrated.apply(classify_supply_demand, axis=1)
        
        # 计算竞争指数（基于高校就业数据）
        integrated = self.calculate_competition_index(integrated)
        
        # 显示集成结果
        print("\n集成结果Top 20:")
        print(integrated[['industry', 'graduate_ratio', 'job_ratio', 'avg_salary', 
                          'competition_index', 'match_score', 'supply_demand_status']]
              .sort_values('match_score', ascending=False).head(20))
        
        self.integrated_df = integrated
        print(f"\n集成完成，共 {len(integrated)} 个行业")
        
        return integrated
    
    def create_four_quadrant_chart(self, integrated_data):
        """创建改进后的高薪高竞争四象限图（基于高校数据的竞争指数）"""
        print("\n创建改进后的四象限图（薪资 vs 高校竞争指数）...")
        
        if integrated_data is None or len(integrated_data) == 0:
            print("没有集成数据，无法创建四象限图")
            return None
        
        try:
            # 创建可视化输出目录
            output_dir = "集成可视化图表"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 准备数据 - 过滤有有效数据的数据
            df = integrated_data[
                (integrated_data['avg_salary'] > 0) & 
                (integrated_data['competition_index'] > 0) &
                (integrated_data['graduate_ratio'] > 0)
            ].copy()
            
            if len(df) == 0:
                print("没有足够的有效数据创建四象限图")
                return None
            
            print(f"用于四象限图的数据: {len(df)} 个行业")
            print(f"薪资范围: {df['avg_salary'].min():.1f}K - {df['avg_salary'].max():.1f}K")
            print(f"竞争指数范围: {df['competition_index'].min():.2f} - {df['competition_index'].max():.2f}")
            
            # 计算中位数作为分界点
            salary_median = df['avg_salary'].median()
            competition_median = df['competition_index'].median()
            
            print(f"薪资中位数: {salary_median:.2f}K")
            print(f"竞争指数中位数: {competition_median:.2f}")
            
            # 创建四象限图 - 使用统一的图形大小
            plt.figure(figsize=(14, 10))
            
            # 将毕业生比例归一化到0-1范围，用于颜色映射
            grad_ratio_min = df['graduate_ratio'].min()
            grad_ratio_max = df['graduate_ratio'].max()
            grad_ratio_range = grad_ratio_max - grad_ratio_min
            
            if grad_ratio_range > 0:
                # 归一化毕业生比例，用于颜色深度
                color_values = (df['graduate_ratio'] - grad_ratio_min) / grad_ratio_range
            else:
                color_values = np.ones(len(df)) * 0.5
            
            # 创建散点图 - 使用统一大小，颜色表示毕业生比例
            scatter = plt.scatter(
                df['avg_salary'], 
                df['competition_index'],
                s=100,  # 统一大小
                c=color_values,  # 颜色表示毕业生比例
                cmap='viridis',  # 使用viridis颜色映射
                alpha=0.7,
                edgecolors='black',
                linewidth=1
            )
            
            # 添加四象限分界线
            plt.axhline(y=competition_median, color='gray', linestyle='--', alpha=0.7, linewidth=2)
            plt.axvline(x=salary_median, color='gray', linestyle='--', alpha=0.7, linewidth=2)
            
            # 添加象限标签 - 放在四角
            plt.text(plt.xlim()[1] * 0.85, plt.ylim()[1] * 0.9, '高薪高竞争', 
                    fontsize=14, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(plt.xlim()[1] * 0.85, plt.ylim()[0] * 1.1, '高薪低竞争', 
                    fontsize=14, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(plt.xlim()[0] * 1.1, plt.ylim()[1] * 0.9, '低薪高竞争', 
                    fontsize=14, fontweight='bold', color='orange',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.text(plt.xlim()[0] * 1.1, plt.ylim()[0] * 1.1, '低薪低竞争', 
                    fontsize=14, fontweight='bold', color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 设置图表属性
            plt.title('行业四象限分析：薪资 vs 高校竞争指数', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('平均月薪 (K)', fontsize=14)
            plt.ylabel('高校竞争指数（基于毕业生比例）', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('毕业生比例（颜色越深比例越高）', fontsize=12)
            
            # 创建图例 - 显示重要行业
            from matplotlib.patches import Patch
            
            # 识别各象限的重要行业（前3名）
            quadrants_data = {
                '高薪高竞争': df[(df['avg_salary'] >= salary_median) & (df['competition_index'] >= competition_median)],
                '高薪低竞争': df[(df['avg_salary'] >= salary_median) & (df['competition_index'] < competition_median)],
                '低薪高竞争': df[(df['avg_salary'] < salary_median) & (df['competition_index'] >= competition_median)],
                '低薪低竞争': df[(df['avg_salary'] < salary_median) & (df['competition_index'] < competition_median)],
            }
            
            # 创建自定义图例
            legend_elements = []
            quadrant_colors = ['red', 'green', 'orange', 'blue']
            
            for i, (quadrant_name, quadrant_df) in enumerate(quadrants_data.items()):
                if len(quadrant_df) > 0:
                    # 按毕业生比例排序，取前3名
                    top_industries = quadrant_df.nlargest(3, 'graduate_ratio')
                    
                    if len(top_industries) > 0:
                        # 添加象限标签
                        legend_elements.append(Patch(
                            facecolor=quadrant_colors[i], 
                            alpha=0.3, 
                            label=f'{quadrant_name} ({len(quadrant_df)}个行业)'
                        ))
                        
                        # 添加该象限的重要行业
                        for j, (_, row) in enumerate(top_industries.iterrows()):
                            industry_name = row['industry'][:15] + '...' if len(row['industry']) > 15 else row['industry']
                            legend_elements.append(Patch(
                                facecolor='white',
                                edgecolor=quadrant_colors[i],
                                label=f"  {industry_name}: {row['avg_salary']:.1f}K/{row['competition_index']:.2f}"
                            ))
            
            # 添加图例
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper left', fontsize=9, 
                          framealpha=0.9, frameon=True)
            
            # 添加统计信息框
            stats_text = f"统计信息:\n"
            stats_text += f"• 总行业数: {len(df)}\n"
            stats_text += f"• 薪资中位数: {salary_median:.1f}K\n"
            stats_text += f"• 竞争指数中位数: {competition_median:.2f}\n"
            stats_text += f"• 薪资范围: {df['avg_salary'].min():.1f}K - {df['avg_salary'].max():.1f}K\n"
            stats_text += f"• 毕业生比例范围: {df['graduate_ratio'].min():.1f}% - {df['graduate_ratio'].max():.1f}%"
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # 添加一些关键行业的标注（避免重叠）
            # 选择毕业生比例最高的5个行业进行标注
            top_grad_ratio = df.nlargest(5, 'graduate_ratio')
            for _, row in top_grad_ratio.iterrows():
                plt.annotate(
                    row['industry'][:10] if len(row['industry']) > 10 else row['industry'],
                    (row['avg_salary'], row['competition_index']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/改进版_行业四象限分析图.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 改进版四象限图已保存到: {output_dir}/改进版_行业四象限分析图.png")
            
            # 返回各象限的数据
            quadrants = {
                '高薪高竞争': df[(df['avg_salary'] >= salary_median) & (df['competition_index'] >= competition_median)],
                '高薪低竞争': df[(df['avg_salary'] >= salary_median) & (df['competition_index'] < competition_median)],
                '低薪高竞争': df[(df['avg_salary'] < salary_median) & (df['competition_index'] >= competition_median)],
                '低薪低竞争': df[(df['avg_salary'] < salary_median) & (df['competition_index'] < competition_median)],
            }
            
            # 打印各象限统计信息
            print("\n各象限统计信息:")
            for quadrant_name, quadrant_data in quadrants.items():
                if len(quadrant_data) > 0:
                    print(f"  {quadrant_name}: {len(quadrant_data)}个行业")
                    print(f"    平均薪资: {quadrant_data['avg_salary'].mean():.2f}K")
                    print(f"    平均竞争指数: {quadrant_data['competition_index'].mean():.2f}")
                    print(f"    平均匹配度: {quadrant_data['match_score'].mean():.2f}")
            
            return quadrants
            
        except Exception as e:
            print(f"创建四象限图出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_separate_competition_charts(self, integrated_data):
        """创建分开的竞争指数分析图表（4张独立图表）"""
        print("\n创建分开的竞争指数分析图表...")
        
        if integrated_data is None or len(integrated_data) == 0:
            print("没有集成数据，无法创建竞争分析图表")
            return
        
        try:
            # 创建可视化输出目录
            output_dir = "集成可视化图表"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 过滤有效数据
            df = integrated_data[
                (integrated_data['competition_index'] > 0) & 
                (integrated_data['avg_salary'] > 0)
            ].copy()
            
            if len(df) == 0:
                print("没有足够的有效数据进行分析")
                return
            
            print(f"用于竞争分析的数据: {len(df)} 个行业")
            
            # 1. 创建竞争指数Top 15行业图
            plt.figure(figsize=(14, 8))
            top_competition = df.sort_values('competition_index', ascending=False).head(15)
            
            y_pos = np.arange(len(top_competition))
            bars = plt.barh(y_pos, top_competition['competition_index'])
            plt.yticks(y_pos, top_competition['industry'], fontsize=10)
            plt.title('1. 竞争指数Top 15行业', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('竞争指数（基于毕业生比例）', fontsize=14)
            plt.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                grad_ratio = top_competition.iloc[i]['graduate_ratio']
                salary = top_competition.iloc[i]['avg_salary']
                plt.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                        f'{width:.2f}\n({grad_ratio:.1f}%/{salary:.1f}K)', 
                        ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/1_竞争指数Top15行业.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 1/4: 竞争指数Top15行业图已保存")
            
            # 2. 创建竞争指数与薪资关系图
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(df['avg_salary'], df['competition_index'],
                                 s=df['graduate_ratio']*50,
                                 c=df['match_score'], cmap='RdYlGn',
                                 alpha=0.7, edgecolors='black')
            
            plt.title('2. 竞争指数 vs 平均薪资', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('平均月薪 (K)', fontsize=14)
            plt.ylabel('竞争指数', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter)
            cbar.set_label('匹配度分数', fontsize=12)
            
            # 添加趋势线
            try:
                z = np.polyfit(df['avg_salary'], df['competition_index'], 1)
                p = np.poly1d(z)
                plt.plot(df['avg_salary'], p(df['avg_salary']), "r--", alpha=0.5)
                plt.text(0.05, 0.95, f'趋势: y={z[0]:.3f}x+{z[1]:.3f}', 
                        transform=plt.gca().transAxes, fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            except:
                pass
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/2_竞争指数与平均薪资关系.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 2/4: 竞争指数与平均薪资关系图已保存")
            
            # 3. 创建行业供需状态分布图
            plt.figure(figsize=(12, 10))
            
            if 'supply_demand_status' in df.columns:
                status_counts = df['supply_demand_status'].value_counts()
                
                colors = ['green', 'lightgreen', 'yellow', 'orange', 'red', 'gray', 'lightgray']
                status_colors = {
                    '供需平衡': 'green',
                    '供不应求': 'lightgreen',
                    '严重供不应求': 'yellow',
                    '供过于求': 'orange',
                    '严重供过于求': 'red',
                    '无市场需求': 'gray',
                    '数据不足': 'lightgray'
                }
                
                pie_colors = [status_colors.get(status, 'gray') for status in status_counts.index]
                
                wedges, texts, autotexts = plt.pie(
                    status_counts.values,
                    labels=status_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=pie_colors,
                    wedgeprops=dict(edgecolor='white', linewidth=2)
                )
                
                # 美化百分比标签
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
                
                plt.title('3. 行业供需状态分布', fontsize=16, fontweight='bold', pad=20)
            else:
                plt.text(0.5, 0.5, '无供需状态数据', ha='center', va='center', fontsize=16)
                plt.title('3. 行业供需状态分布', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/3_行业供需状态分布.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 3/4: 行业供需状态分布图已保存")
            
            # 4. 创建竞争程度分布图
            plt.figure(figsize=(12, 10))
            
            if 'competition_level' in df.columns:
                level_counts = df['competition_level'].value_counts()
                colors = ['green', 'yellow', 'orange', 'red']
                
                wedges, texts, autotexts = plt.pie(
                    level_counts.values,
                    labels=level_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors[:len(level_counts)],
                    wedgeprops=dict(edgecolor='white', linewidth=2)
                )
                
                # 美化百分比标签
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)
                
                plt.title('4. 行业竞争程度分布', fontsize=16, fontweight='bold', pad=20)
            else:
                plt.text(0.5, 0.5, '无竞争程度分类数据', ha='center', va='center', fontsize=16)
                plt.title('4. 行业竞争程度分布', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/4_行业竞争程度分布.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 4/4: 行业竞争程度分布图已保存")
            
            print(f"\n✓ 所有竞争指数分析图表已保存到 '{output_dir}' 目录下")
            
        except Exception as e:
            print(f"创建竞争分析图表出错: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_comprehensive_text_report(self, integrated_data, quadrants=None):
        """生成综合文字分析报告，整合所有文字分析结果"""
        print("\n生成综合文字分析报告...")
        
        if integrated_data is None or len(integrated_data) == 0:
            print("没有集成数据，无法生成报告")
            return
        
        try:
            # 过滤有效数据
            df = integrated_data[
                (integrated_data['avg_salary'] > 0) & 
                (integrated_data['competition_index'] > 0)
            ].copy()
            
            if len(df) == 0:
                print("没有足够的有效数据生成报告")
                return
            
            # 开始构建报告内容
            report_content = []
            report_content.append("=" * 100)
            report_content.append("高校-市场数据集成分析综合报告")
            report_content.append("=" * 100)
            report_content.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"分析行业数: {len(df)}")
            report_content.append("")
            
            # 第一部分：总体概况
            report_content.append("一、总体概况")
            report_content.append("-" * 50)
            
            total_graduates = df['graduate_number'].sum()
            total_jobs = df['job_count'].sum()
            
            report_content.append(f"1. 总毕业生数: {total_graduates:,}")
            report_content.append(f"2. 总岗位数: {total_jobs:,}")
            
            if 'employment_rate' in df.columns:
                avg_employment = df['employment_rate'].mean()
                report_content.append(f"3. 平均就业率: {avg_employment:.1f}%")
            
            if 'avg_salary' in df.columns:
                avg_salary = df['avg_salary'].mean()
                median_salary = df['avg_salary'].median()
                report_content.append(f"4. 平均月薪: {avg_salary:.1f}K (中位数: {median_salary:.1f}K)")
            
            if 'competition_index' in df.columns:
                avg_comp = df['competition_index'].mean()
                median_comp = df['competition_index'].median()
                report_content.append(f"5. 平均竞争指数: {avg_comp:.2f} (中位数: {median_comp:.2f})")
            report_content.append("")
            
            # 第二部分：竞争指数分析
            report_content.append("二、基于高校数据的竞争指数分析")
            report_content.append("-" * 50)
            
            if 'competition_index' in df.columns:
                # 竞争指数Top 10
                top_competition = df.nlargest(10, 'competition_index')
                report_content.append("1. 竞争最激烈的Top 10行业:")
                report_content.append(f"{'排名':<5} {'行业':<25} {'竞争指数':<12} {'毕业生比例':<12} {'平均薪资':<12} {'匹配度':<12}")
                report_content.append("-" * 90)
                for i, (_, row) in enumerate(top_competition.iterrows(), 1):
                    report_content.append(
                        f"{i:<5} {str(row.get('industry', '未知'))[:24]:<25} "
                        f"{row.get('competition_index', 0):<12.2f} "
                        f"{row.get('graduate_ratio', 0):<12.1f}% "
                        f"{row.get('avg_salary', 0):<12.1f}K "
                        f"{row.get('match_score', 0):<12.1f}"
                    )
                
                # 竞争指数最低的Top 10
                bottom_competition = df.nsmallest(10, 'competition_index')
                report_content.append("\n2. 竞争最不激烈的Top 10行业:")
                report_content.append(f"{'排名':<5} {'行业':<25} {'竞争指数':<12} {'毕业生比例':<12} {'平均薪资':<12} {'匹配度':<12}")
                report_content.append("-" * 90)
                for i, (_, row) in enumerate(bottom_competition.iterrows(), 1):
                    report_content.append(
                        f"{i:<5} {str(row.get('industry', '未知'))[:24]:<25} "
                        f"{row.get('competition_index', 0):<12.2f} "
                        f"{row.get('graduate_ratio', 0):<12.1f}% "
                        f"{row.get('avg_salary', 0):<12.1f}K "
                        f"{row.get('match_score', 0):<12.1f}"
                    )
            
            # 第三部分：高薪资行业分析
            report_content.append("\n三、高薪资行业分析")
            report_content.append("-" * 50)
            
            if 'avg_salary' in df.columns:
                high_salary = df.nlargest(10, 'avg_salary')
                report_content.append("1. 高薪资行业Top 10:")
                report_content.append(f"{'排名':<5} {'行业':<25} {'平均薪资':<12} {'竞争指数':<12} {'毕业生比例':<12} {'匹配度':<12}")
                report_content.append("-" * 90)
                for i, (_, row) in enumerate(high_salary.iterrows(), 1):
                    report_content.append(
                        f"{i:<5} {str(row.get('industry', '未知'))[:24]:<25} "
                        f"{row.get('avg_salary', 0):<12.1f}K "
                        f"{row.get('competition_index', 0):<12.2f} "
                        f"{row.get('graduate_ratio', 0):<12.1f}% "
                        f"{row.get('match_score', 0):<12.1f}"
                    )
            
            # 第四部分：高匹配度行业分析
            report_content.append("\n四、高匹配度行业分析（培养质量高）")
            report_content.append("-" * 50)
            
            if 'match_score' in df.columns:
                high_match = df.nlargest(10, 'match_score')
                report_content.append("1. 高匹配度行业Top 10:")
                report_content.append(f"{'排名':<5} {'行业':<25} {'匹配度':<12} {'竞争指数':<12} {'就业率':<12} {'平均薪资':<12}")
                report_content.append("-" * 90)
                for i, (_, row) in enumerate(high_match.iterrows(), 1):
                    report_content.append(
                        f"{i:<5} {str(row.get('industry', '未知'))[:24]:<25} "
                        f"{row.get('match_score', 0):<12.1f} "
                        f"{row.get('competition_index', 0):<12.2f} "
                        f"{row.get('employment_rate', 0):<12.1f}% "
                        f"{row.get('avg_salary', 0):<12.1f}K"
                    )
            
            # 第五部分：四象限分析结果
            report_content.append("\n五、四象限分析结果（基于高校竞争指数）")
            report_content.append("-" * 50)
            
            if quadrants:
                for quadrant_name, quadrant_data in quadrants.items():
                    if len(quadrant_data) > 0:
                        report_content.append(f"1. {quadrant_name}象限:")
                        report_content.append(f"   行业数量: {len(quadrant_data)}个")
                        report_content.append(f"   平均薪资: {quadrant_data['avg_salary'].mean():.2f}K")
                        report_content.append(f"   平均竞争指数: {quadrant_data['competition_index'].mean():.2f}")
                        report_content.append(f"   平均匹配度: {quadrant_data['match_score'].mean():.2f}")
                        
                        # 列出前5个主要行业
                        report_content.append(f"   主要行业（前5个）:")
                        for i, (_, row) in enumerate(quadrant_data.head(5).iterrows(), 1):
                            report_content.append(
                                f"     {i}. {str(row.get('industry', '未知'))[:20]:<20} "
                                f"薪资:{row.get('avg_salary', 0):>5.1f}K "
                                f"竞争:{row.get('competition_index', 0):>5.2f} "
                                f"匹配:{row.get('match_score', 0):>5.1f}"
                            )
                        report_content.append("")
            
            # 第六部分：供需状态分析
            report_content.append("六、行业供需状态分析")
            report_content.append("-" * 50)
            
            if 'supply_demand_status' in df.columns:
                status_counts = df['supply_demand_status'].value_counts()
                report_content.append("1. 供需状态分布:")
                for status, count in status_counts.items():
                    percentage = count / len(df) * 100
                    report_content.append(f"   {status}: {count}个行业 ({percentage:.1f}%)")
            
            # 第七部分：政策建议
            report_content.append("\n七、政策建议（基于高校竞争指数分析）")
            report_content.append("-" * 50)
            
            report_content.append("1. 高校专业设置建议:")
            report_content.append("   • 降低高竞争行业相关专业的招生规模，避免毕业生供过于求")
            report_content.append("   • 扩大低竞争、高薪资行业的专业培养，满足市场需求")
            report_content.append("   • 关注供需平衡行业，保持适度培养规模")
            report_content.append("   • 加强新兴行业和紧缺人才领域的专业建设")
            report_content.append("")
            
            report_content.append("2. 学生择业指导建议:")
            report_content.append("   • 高竞争行业: 建议学生提升专业技能，增加竞争力，考虑差异化发展")
            report_content.append("   • 低竞争行业: 适合希望避免激烈竞争、寻求稳定发展的学生")
            report_content.append("   • 高薪高竞争: 适合能力强、追求高回报、愿意接受挑战的学生")
            report_content.append("   • 低薪低竞争: 适合追求工作稳定性、工作生活平衡的学生")
            report_content.append("   • 供需平衡行业: 风险较低，适合大多数学生选择")
            report_content.append("")
            
            report_content.append("3. 人才培养模式建议:")
            report_content.append("   • 加强校企合作，提高人才培养的市场适应性")
            report_content.append("   • 建立动态调整机制，根据市场需求调整专业设置")
            report_content.append("   • 加强就业指导，帮助学生了解不同行业的竞争状况")
            report_content.append("   • 建立毕业生就业跟踪机制，持续优化人才培养方案")
            
            # 第八部分：关键发现
            report_content.append("\n八、关键发现")
            report_content.append("-" * 50)
            
            # 竞争最激烈的行业
            if 'competition_index' in df.columns and len(df) > 0:
                max_comp_industry = df.loc[df['competition_index'].idxmax()]
                report_content.append(f"1. 竞争最激烈的行业: {max_comp_industry.get('industry', '未知')}")
                report_content.append(f"   竞争指数: {max_comp_industry.get('competition_index', 0):.2f}")
                report_content.append(f"   毕业生比例: {max_comp_industry.get('graduate_ratio', 0):.1f}%")
                report_content.append(f"   平均薪资: {max_comp_industry.get('avg_salary', 0):.1f}K")
            
            # 薪资最高的行业
            if 'avg_salary' in df.columns and len(df) > 0:
                max_salary_industry = df.loc[df['avg_salary'].idxmax()]
                report_content.append(f"\n2. 薪资最高的行业: {max_salary_industry.get('industry', '未知')}")
                report_content.append(f"   平均薪资: {max_salary_industry.get('avg_salary', 0):.1f}K")
                report_content.append(f"   竞争指数: {max_salary_industry.get('competition_index', 0):.2f}")
                report_content.append(f"   匹配度: {max_salary_industry.get('match_score', 0):.1f}")
            
            # 匹配度最高的行业
            if 'match_score' in df.columns and len(df) > 0:
                max_match_industry = df.loc[df['match_score'].idxmax()]
                report_content.append(f"\n3. 匹配度最高的行业: {max_match_industry.get('industry', '未知')}")
                report_content.append(f"   匹配度: {max_match_industry.get('match_score', 0):.1f}")
                report_content.append(f"   就业率: {max_match_industry.get('employment_rate', 0):.1f}%")
                report_content.append(f"   平均薪资: {max_match_industry.get('avg_salary', 0):.1f}K")
            
            # 第九部分：数据统计
            report_content.append("\n九、数据统计摘要")
            report_content.append("-" * 50)
            
            report_content.append("1. 薪资统计:")
            report_content.append(f"   平均薪资: {df['avg_salary'].mean():.1f}K")
            report_content.append(f"   中位数薪资: {df['avg_salary'].median():.1f}K")
            report_content.append(f"   最高薪资: {df['avg_salary'].max():.1f}K")
            report_content.append(f"   最低薪资: {df['avg_salary'].min():.1f}K")
            
            report_content.append("\n2. 竞争指数统计:")
            report_content.append(f"   平均竞争指数: {df['competition_index'].mean():.2f}")
            report_content.append(f"   中位数竞争指数: {df['competition_index'].median():.2f}")
            report_content.append(f"   最高竞争指数: {df['competition_index'].max():.2f}")
            report_content.append(f"   最低竞争指数: {df['competition_index'].min():.2f}")
            
            if 'match_score' in df.columns:
                report_content.append("\n3. 匹配度统计:")
                report_content.append(f"   平均匹配度: {df['match_score'].mean():.1f}")
                report_content.append(f"   中位数匹配度: {df['match_score'].median():.1f}")
                report_content.append(f"   最高匹配度: {df['match_score'].max():.1f}")
                report_content.append(f"   最低匹配度: {df['match_score'].min():.1f}")
            
            # 结束语
            report_content.append("\n" + "=" * 100)
            report_content.append("报告结束")
            report_content.append("=" * 100)
            
            # 保存报告到文件
            with open('综合文字分析报告.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            print("✓ 综合文字分析报告已保存到: 综合文字分析报告.txt")
            
            return report_content
            
        except Exception as e:
            print(f"生成综合报告出错: {e}")
            import traceback
            traceback.print_exc()
            return []

# 主程序
if __name__ == "__main__":
    print("="*80)
    print("高级数据集成与四象限分析系统（基于高校竞争指数）")
    print("="*80)
    matplotlib.rc("font",family='FangSong')
    try:
        # 1. 初始化分析器
        analyzer = AdvancedIntegrationAnalyzer()
        
        # 2. 加载数据
        university_file = "高校就业详细数据.xlsx"
        market_file = "cleaned_recruitment_data_all.csv"
        
        university_df, market_df = analyzer.load_data(university_file, market_file)
        
        if university_df is None or market_df is None:
            print("数据加载失败，请检查文件路径和格式")
            exit(1)
        
        # 3. 准备专业映射器
        print("\n初始化专业映射器...")
        mapper = MajorIndustryMapper()
        
        # 4. 准备数据（基于比例）
        print("\n" + "="*80)
        print("基于比例的数据准备")
        print("="*80)
        
        university_data = analyzer.prepare_university_data(mapper)
        market_data = analyzer.prepare_market_data()
        
        if university_data is None or market_data is None:
            print("数据准备失败，请检查数据内容")
            exit(1)
        
        # 5. 数据集成分析
        print("\n" + "="*80)
        print("数据集成分析（计算高校竞争指数）")
        print("="*80)
        
        integrated_data = analyzer.integrate_data(university_data, market_data)
        
        if integrated_data is None:
            print("数据集成失败")
            exit(1)
        
        # 6. 创建四象限图（基于高校竞争指数）
        print("\n" + "="*80)
        print("四象限分析（基于高校竞争指数）")
        print("="*80)
        
        quadrants = analyzer.create_four_quadrant_chart(integrated_data)
        
        # 7. 创建分开的竞争指数分析图表
        print("\n" + "="*80)
        print("创建分开的竞争指数分析图表")
        print("="*80)
        
        analyzer.create_separate_competition_charts(integrated_data)
        
        # 8. 生成综合文字分析报告（整合所有文字分析结果）
        print("\n" + "="*80)
        print("生成综合文字分析报告")
        print("="*80)
        
        analyzer.generate_comprehensive_text_report(integrated_data, quadrants)
        
        # 9. 保存集成数据
        if integrated_data is not None:
            # 确保所有列都是可序列化的
            for col in integrated_data.columns:
                if integrated_data[col].dtype.name == 'category':
                    integrated_data[col] = integrated_data[col].astype(str)
            
            # 过滤有效数据保存
            valid_data = integrated_data[
                (integrated_data['avg_salary'] > 0) & 
                (integrated_data['competition_index'] > 0)
            ]
            
            if len(valid_data) > 0:
                valid_data.to_csv('高级集成分析数据_高校竞争指数.csv', index=False, encoding='utf-8-sig')
                print("✓ 集成分析数据已保存到: 高级集成分析数据_高校竞争指数.csv")
            else:
                print("警告: 没有有效的集成数据可保存")
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print("\n生成的主要文件:")
        print("1. 高级集成分析数据_高校竞争指数.csv - 完整的集成分析数据")
        print("2. 综合文字分析报告.txt - 整合所有文字分析结果的综合报告")
        print("3. 集成可视化图表/ - 包含所有可视化图表")
        print("   行业四象限分析图_高校竞争指数.png - 薪资vs高校竞争指数四象限图")
        print("   1_竞争指数Top15行业.png - 竞争指数Top15行业图")
        print("   2_竞争指数与平均薪资关系.png - 竞争指数与薪资关系图")
        print("   3_行业供需状态分布.png - 行业供需状态分布图")
        print("   4_行业竞争程度分布.png - 行业竞争程度分布图")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件存在:")
        print("1. 高校就业详细数据.xlsx (或您的实际高校数据文件)")
        print("2. cleaned_recruitment_data_all.csv")
    except Exception as e:
        print(f"分析出错: {e}")
        import traceback
        traceback.print_exc()