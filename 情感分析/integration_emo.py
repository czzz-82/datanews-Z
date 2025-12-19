# ==================== 精简版：职场焦虑核心分析系统 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import os
import glob

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

class AnxietyCoreAnalyzer:
    """职场焦虑核心分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.recruitment_df = None  # 招聘市场数据
        self.anxiety_df = None      # 职场焦虑数据
        self.university_df = None   # 高校培养数据
        self.integrated_df = None   # 整合后的数据
        
    def load_recruitment_data(self, file_path: str):
        """加载招聘市场数据"""
        print(f"加载招聘市场数据: {file_path}")
        
        try:
            # 读取CSV文件
            self.recruitment_df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            print(f"招聘数据加载成功: {self.recruitment_df.shape}")
            
            # 标准化行业分类
            self._prepare_recruitment_data()
            
            return True
            
        except Exception as e:
            print(f"加载招聘数据失败: {e}")
            return False
    
    def _prepare_recruitment_data(self):
        """准备招聘市场数据"""
        # 标准化行业分类
        if '行业' in self.recruitment_df.columns:
            # 简化行业分类
            self.recruitment_df['行业_标准化'] = self.recruitment_df['行业'].apply(self._standardize_industry)
        
        # 按标准化行业汇总数据
        if '行业_标准化' in self.recruitment_df.columns:
            industry_stats = self.recruitment_df.groupby('行业_标准化').agg({
                '薪资_月平均K': ['mean', 'count']
            }).reset_index()
            
            # 扁平化列名
            industry_stats.columns = ['industry', 'recruit_avg_salary', 'recruit_job_count']
            
            self.recruitment_industry_stats = industry_stats
    
    def _standardize_industry(self, industry_str):
        """标准化行业分类"""
        if pd.isna(industry_str):
            return '其他'
        
        industry_lower = str(industry_str).lower()
        
        # 互联网相关
        if any(keyword in industry_lower for keyword in ['互联网', 'it', '计算机', '软件', '游戏', '电子商务', 
                                                        '人工智能', '大数据', '云计算', '物联网', '移动互联网',
                                                        '信息传输', '信息技术']):
            return '互联网'
        
        # 金融相关
        elif any(keyword in industry_lower for keyword in ['金融', '银行', '证券', '基金', '保险', '投资', 
                                                          '理财', '信贷', '信托', '互联网金融', '金融科技']):
            return '金融'
        
        # 体制内相关
        elif any(keyword in industry_lower for keyword in ['政府', '事业单位', '公共管理', '社会保障', '社会组织', 
                                                          '非营利', '公共服务', '机关', '行政', 
                                                          '教育', '培训', '学校', '学院', '教师',
                                                          '医疗', '医药', '医院', '健康', '生物', '医生', '护士',
                                                          '卫生', '社会工作']):
            return '体制内'
        
        else:
            return '其他'
    
    def load_anxiety_data(self, file_patterns: list):
        """加载职场焦虑数据"""
        print(f"\n加载职场焦虑数据: {file_patterns}")
        
        all_anxiety_data = []
        
        for pattern in file_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    # 从文件名提取行业信息
                    filename = os.path.basename(file)
                    industry = '未知'
                    
                    if '互联网' in filename:
                        industry = '互联网'
                    elif '金融' in filename:
                        industry = '金融'
                    elif '体制内' in filename:
                        industry = '体制内'
                    
                    # 读取CSV文件
                    df = pd.read_csv(file, encoding='utf-8-sig')
                    df['industry'] = industry
                    
                    all_anxiety_data.append(df)
                    print(f"✓ 加载文件: {file} ({len(df)} 条记录, 行业: {industry})")
                    
                except Exception as e:
                    print(f"✗ 加载文件失败 {file}: {e}")
        
        if all_anxiety_data:
            self.anxiety_df = pd.concat(all_anxiety_data, ignore_index=True)
            print(f"\n焦虑数据合并成功: {self.anxiety_df.shape}")
            
            # 计算各行业焦虑统计
            self._calculate_anxiety_stats()
            
            return True
        else:
            print("没有找到焦虑数据文件")
            return False
    
    def _calculate_anxiety_stats(self):
        """计算各行业焦虑统计"""
        if self.anxiety_df is None or len(self.anxiety_df) == 0:
            print("没有焦虑数据")
            return
        
        # 按行业统计焦虑指数
        anxiety_stats = self.anxiety_df.groupby('industry').agg({
            'overall_anxiety': 'mean',
            'career_path_confusion': 'mean',
            'salary_gap': 'mean',
            'workplace_involution': 'mean',
            'identity_crisis': 'mean',
            'sentiment': lambda x: x.mode()[0] if not x.mode().empty else '未知'
        }).reset_index()
        
        anxiety_stats.columns = ['industry', 'avg_anxiety', 'avg_career_confusion', 
                                'avg_salary_gap', 'avg_involution', 'avg_identity_crisis', 'dominant_sentiment']
        
        self.anxiety_industry_stats = anxiety_stats
    
    def load_university_data(self, file_path: str):
        """加载高校培养数据"""
        print(f"\n加载高校培养数据: {file_path}")
        
        try:
            # 读取CSV文件
            self.university_df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            print(f"高校数据加载成功: {self.university_df.shape}")
            
            # 标准化行业分类
            self.university_df['industry_std'] = self.university_df['industry'].apply(self._standardize_university_industry)
            
            # 按标准化行业汇总
            university_stats = self.university_df.groupby('industry_std').agg({
                'avg_salary': 'mean',
                'match_score': 'mean',
                'competition_index': 'mean'
            }).reset_index()
            
            university_stats.columns = ['industry', 'uni_avg_salary', 'uni_match_score', 'uni_competition_index']
            
            self.university_industry_stats = university_stats
            
            return True
            
        except Exception as e:
            print(f"加载高校数据失败: {e}")
            return False
    
    def _standardize_university_industry(self, industry_str):
        """标准化高校数据行业分类"""
        if pd.isna(industry_str):
            return '其他'
        
        industry_str = str(industry_str).strip()
        
        # 互联网相关
        if '信息传输、软件和信息技术服务业' in industry_str:
            return '互联网'
        
        # 金融相关
        elif '金融业' in industry_str or '金融' in industry_str:
            return '金融'
        
        # 体制内相关
        elif any(keyword in industry_str for keyword in ['公共管理、社会保障和社会组织', '教育', '卫生和社会工作']):
            return '体制内'
        
        else:
            return '其他'
    
    def integrate_all_data(self):
        """整合三个数据源"""
        print("\n整合三个数据源...")
        
        # 检查数据是否加载
        if not hasattr(self, 'recruitment_industry_stats'):
            print("没有招聘市场数据")
            return None
        
        if not hasattr(self, 'anxiety_industry_stats'):
            print("没有焦虑数据")
            return None
        
        if not hasattr(self, 'university_industry_stats'):
            print("没有高校培养数据")
            return None
        
        # 合并招聘和高校数据
        merged_data = pd.merge(
            self.recruitment_industry_stats,
            self.university_industry_stats,
            on='industry',
            how='inner'
        )
        
        # 合并焦虑数据
        integrated_data = pd.merge(
            merged_data,
            self.anxiety_industry_stats,
            on='industry',
            how='inner'
        )
        
        print(f"\n整合后数据: {len(integrated_data)} 个行业")
        print("整合数据预览:")
        print(integrated_data[['industry', 'recruit_avg_salary', 'avg_anxiety', 
                              'uni_match_score', 'uni_competition_index']])
        
        self.integrated_df = integrated_data
        return integrated_data
    
    
    def create_competition_anxiety_chart(self, output_dir: str = "焦虑核心分析"):
        """3. 创建竞争指数和焦虑关系图"""
        if self.integrated_df is None or len(self.integrated_df) == 0:
            print("没有整合数据")
            return
        
        print("\n3. 创建竞争指数和焦虑关系图...")
        
        # 创建图形
        plt.figure(figsize=(14, 10))
        
        # 确保数据是数值型
        self.integrated_df['uni_competition_index'] = pd.to_numeric(self.integrated_df['uni_competition_index'], errors='coerce')
        self.integrated_df['avg_anxiety'] = pd.to_numeric(self.integrated_df['avg_anxiety'], errors='coerce')
        
        # 创建散点图
        scatter = plt.scatter(
            self.integrated_df['uni_competition_index'],
            self.integrated_df['avg_anxiety'],
            s=self.integrated_df['recruit_avg_salary'].fillna(20) * 10,  # 点大小表示薪资水平
            c=self.integrated_df['uni_match_score'].fillna(0.5),  # 颜色表示匹配度
            cmap='RdYlGn',  # 红绿颜色映射
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5
        )
        
        # 添加行业标签
        for i, row in self.integrated_df.iterrows():
            plt.annotate(
                row['industry'],
                (row['uni_competition_index'], row['avg_anxiety']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        # 添加回归线
        try:
            # 移除NaN值
            valid_data = self.integrated_df[['uni_competition_index', 'avg_anxiety']].dropna()
            if len(valid_data) >= 2:
                x = valid_data['uni_competition_index'].values
                y = valid_data['avg_anxiety'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "k--", alpha=0.7, linewidth=2, label=f'趋势线: y={z[0]:.3f}x+{z[1]:.3f}')
                
                # 计算相关系数
                correlation = np.corrcoef(x, y)[0, 1]
                plt.text(0.3, 0.95, f'相关系数: r = {correlation:.3f}', 
                        transform=plt.gca().transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"添加回归线失败: {e}")
        
        # 设置图表属性
        plt.title('竞争指数与职场焦虑关系分析', fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('竞争指数', fontsize=14)
        plt.ylabel('平均焦虑指数', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('高校-市场匹配度', fontsize=12)
        
        # 添加图例说明
        from matplotlib.patches import Patch
        
        # 创建大小图例（薪资水平）
        legend_elements = [
            plt.scatter([], [], s=200, facecolor='gray', edgecolor='black', alpha=0.6, 
                       label='薪资 ≈ 10K'),
            plt.scatter([], [], s=400, facecolor='gray', edgecolor='black', alpha=0.6, 
                       label='薪资 ≈ 20K'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', title='薪资水平（点大小）')
        
        # 添加象限分界线
        competition_median = self.integrated_df['uni_competition_index'].median()
        anxiety_median = self.integrated_df['avg_anxiety'].median()
        
        plt.axhline(y=anxiety_median, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        plt.axvline(x=competition_median, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3_竞争指数与焦虑关系图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 3/4: 竞争指数与焦虑关系图已保存")
    
    def create_anxiety_dimensions_chart(self, output_dir: str = "焦虑核心分析"):
        """4. 创建行业焦虑多维度对比条形图"""
        if self.integrated_df is None or len(self.integrated_df) == 0:
            print("没有整合数据")
            return
        
        print("\n4. 创建行业焦虑多维度对比条形图...")
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 需要对比的指标
        metrics = [
            ('recruit_avg_salary', '平均薪资(K)', 'steelblue'),
            ('avg_anxiety', '整体焦虑指数', 'coral'),
            ('avg_career_confusion', '发展迷茫焦虑', 'gold'),
            ('avg_salary_gap', '薪酬落差焦虑', 'lightcoral'),
            ('avg_involution', '职场内卷焦虑', 'lightgreen'),
            ('avg_identity_crisis', '身份认同焦虑', 'lightblue'),
            ('uni_match_score', '匹配度', 'purple'),
            ('uni_competition_index', '竞争指数', 'orange')
        ]
        
        # 只显示前6个指标（2x3布局）
        for i, (metric, name, color) in enumerate(metrics[:6]):
            if i >= len(axes):
                break
                
            # 筛选有数据的行业
            valid_data = self.integrated_df.dropna(subset=[metric])
            if len(valid_data) == 0:
                axes[i].text(0.5, 0.5, f'无{name}数据', ha='center', va='center', fontsize=12)
                axes[i].set_title(name, fontsize=12, fontweight='bold')
                continue
            
            # 排序
            sorted_data = valid_data.sort_values(metric, ascending=False)
            
            # 创建条形图
            bars = axes[i].bar(range(len(sorted_data)), sorted_data[metric], color=color, alpha=0.7)
            axes[i].set_xticks(range(len(sorted_data)))
            axes[i].set_xticklabels(sorted_data['industry'], rotation=45, ha='right', fontsize=10)
            axes[i].set_title(f'{name}对比', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}' if metric in ['uni_match_score', 'uni_competition_index'] else f'{height:.1f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 隐藏多余的子图
        for i in range(len(metrics[:6]), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('行业焦虑多维度对比分析', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/4_行业焦虑多维度对比图.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建补充图表：薪资、匹配度、竞争指数的对比图
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        
        supplement_metrics = [
            ('recruit_avg_salary', '平均薪资(K)', 'steelblue'),
            ('uni_match_score', '匹配度', 'purple'),
            ('uni_competition_index', '竞争指数', 'orange')
        ]
        
        for i, (metric, name, color) in enumerate(supplement_metrics):
            # 筛选有数据的行业
            valid_data = self.integrated_df.dropna(subset=[metric])
            if len(valid_data) == 0:
                continue
            
            # 排序
            sorted_data = valid_data.sort_values(metric, ascending=False)
            
            # 创建条形图
            bars = axes2[i].bar(range(len(sorted_data)), sorted_data[metric], color=color, alpha=0.7)
            axes2[i].set_xticks(range(len(sorted_data)))
            axes2[i].set_xticklabels(sorted_data['industry'], rotation=45, ha='right', fontsize=10)
            axes2[i].set_title(f'{name}对比', fontsize=12, fontweight='bold')
            axes2[i].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes2[i].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}' if metric in ['uni_match_score', 'uni_competition_index'] else f'{height:.1f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.suptitle('行业薪资、匹配度、竞争指数对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/4_补充_行业核心指标对比图.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 4/4: 行业焦虑多维度对比图已保存（共2张）")
    
    def generate_core_analysis_report(self, output_dir: str = "焦虑核心分析"):
        """生成核心分析报告"""
        print("\n生成核心分析报告...")
        
        if self.integrated_df is None or len(self.integrated_df) == 0:
            print("没有整合数据")
            return
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 开始构建报告内容
        report_content = []
        report_content.append("=" * 100)
        report_content.append("职场焦虑核心分析报告")
        report_content.append("=" * 100)
        report_content.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # 1. 数据概况
        report_content.append("一、数据概况")
        report_content.append("-" * 50)
        
        report_content.append(f"1. 整合行业数: {len(self.integrated_df)}")
        report_content.append(f"2. 主要分析指标:")
        report_content.append(f"   • 行业薪资水平 (招聘市场平均月薪)")
        report_content.append(f"   • 职场焦虑指数 (整体焦虑及四个维度)")
        report_content.append(f"   • 高校-市场匹配度")
        report_content.append(f"   • 高校竞争指数")
        report_content.append("")
        
        # 2. 核心发现
        report_content.append("二、核心发现")
        report_content.append("-" * 50)
        
        # 2.1 薪资与焦虑关系分析
        if 'recruit_avg_salary' in self.integrated_df.columns and 'avg_anxiety' in self.integrated_df.columns:
            # 计算相关性
            salary_anxiety_corr = self.integrated_df['recruit_avg_salary'].corr(self.integrated_df['avg_anxiety'])
            
            report_content.append(f"1. 薪资与焦虑关系 (图1):")
            report_content.append(f"   • 相关系数: {salary_anxiety_corr:.3f}")
            
            if salary_anxiety_corr > 0.3:
                report_content.append(f"   • 发现: 薪资越高，焦虑程度也越高")
                report_content.append(f"   • 解读: 高薪行业可能伴随更高的工作压力和期望")
            elif salary_anxiety_corr < -0.3:
                report_content.append(f"   • 发现: 薪资越高，焦虑程度越低")
                report_content.append(f"   • 解读: 高薪行业可能提供更好的工作保障和满意度")
            else:
                report_content.append(f"   • 发现: 薪资与焦虑无明显线性关系")
                report_content.append(f"   • 解读: 薪资不是影响职场焦虑的主要因素")
            
            # 识别象限分布
            salary_median = self.integrated_df['recruit_avg_salary'].median()
            anxiety_median = self.integrated_df['avg_anxiety'].median()
            
            high_salary_high_anxiety = self.integrated_df[
                (self.integrated_df['recruit_avg_salary'] >= salary_median) &
                (self.integrated_df['avg_anxiety'] >= anxiety_median)
            ]
            
            if len(high_salary_high_anxiety) > 0:
                report_content.append(f"   • 高薪高焦虑行业: {', '.join(high_salary_high_anxiety['industry'].tolist())}")
            
            high_salary_low_anxiety = self.integrated_df[
                (self.integrated_df['recruit_avg_salary'] >= salary_median) &
                (self.integrated_df['avg_anxiety'] < anxiety_median)
            ]
            
            if len(high_salary_low_anxiety) > 0:
                report_content.append(f"   • 高薪低焦虑行业: {', '.join(high_salary_low_anxiety['industry'].tolist())}")
        
        # 2.2 匹配度与焦虑关系分析
        if 'uni_match_score' in self.integrated_df.columns and 'avg_anxiety' in self.integrated_df.columns:
            match_anxiety_corr = self.integrated_df['uni_match_score'].corr(self.integrated_df['avg_anxiety'])
            
            report_content.append(f"\n2. 匹配度与焦虑关系 (图2):")
            report_content.append(f"   • 相关系数: {match_anxiety_corr:.3f}")
            
            if match_anxiety_corr > 0.3:
                report_content.append(f"   • 发现: 匹配度越高，焦虑程度也越高")
                report_content.append(f"   • 解读: 培养质量高的行业可能对学生期望更高，导致职场焦虑")
            elif match_anxiety_corr < -0.3:
                report_content.append(f"   • 发现: 匹配度越高，焦虑程度越低")
                report_content.append(f"   • 解读: 良好的培养质量有助于学生适应职场，降低焦虑")
            else:
                report_content.append(f"   • 发现: 匹配度与焦虑无明显线性关系")
                report_content.append(f"   • 解读: 培养质量不是影响职场焦虑的主要因素")
        
        # 2.3 竞争指数与焦虑关系分析
        if 'uni_competition_index' in self.integrated_df.columns and 'avg_anxiety' in self.integrated_df.columns:
            comp_anxiety_corr = self.integrated_df['uni_competition_index'].corr(self.integrated_df['avg_anxiety'])
            
            report_content.append(f"\n3. 竞争指数与焦虑关系 (图3):")
            report_content.append(f"   • 相关系数: {comp_anxiety_corr:.3f}")
            
            if comp_anxiety_corr > 0.3:
                report_content.append(f"   • 发现: 竞争越激烈，焦虑程度越高")
                report_content.append(f"   • 解读: 高竞争行业的工作环境和压力导致更高的职场焦虑")
            elif comp_anxiety_corr < -0.3:
                report_content.append(f"   • 发现: 竞争越激烈，焦虑程度越低")
                report_content.append(f"   • 解读: 竞争激烈的行业可能吸引更有竞争力的个体，对焦虑的抵抗力更强")
            else:
                report_content.append(f"   • 发现: 竞争指数与焦虑无明显线性关系")
                report_content.append(f"   • 解读: 竞争程度不是影响职场焦虑的主要因素")
        
        # 3. 行业对比分析
        report_content.append("\n三、行业对比分析 (图4)")
        report_content.append("-" * 50)
        
        # 各维度排名
        dimensions = {
            'recruit_avg_salary': '薪资水平',
            'avg_anxiety': '整体焦虑',
            'avg_career_confusion': '发展迷茫焦虑',
            'avg_salary_gap': '薪酬落差焦虑',
            'avg_involution': '职场内卷焦虑',
            'avg_identity_crisis': '身份认同焦虑',
            'uni_match_score': '匹配度',
            'uni_competition_index': '竞争指数'
        }
        
        for dim, name in dimensions.items():
            if dim in self.integrated_df.columns:
                # 找出最高和最低的行业
                if not self.integrated_df[dim].isnull().all():
                    max_idx = self.integrated_df[dim].idxmax()
                    min_idx = self.integrated_df[dim].idxmin()
                    
                    max_value = self.integrated_df.loc[max_idx, dim]
                    min_value = self.integrated_df.loc[min_idx, dim]
                    max_industry = self.integrated_df.loc[max_idx, 'industry']
                    min_industry = self.integrated_df.loc[min_idx, 'industry']
                    
                    report_content.append(f"• {name}:")
                    report_content.append(f"  最高: {max_industry} ({max_value:.2f})")
                    report_content.append(f"  最低: {min_industry} ({min_value:.2f})")
        
        # 4. 重点行业深度分析
        report_content.append("\n四、重点行业深度分析")
        report_content.append("-" * 50)
        
        key_industries = ['互联网', '金融', '体制内']
        for industry in key_industries:
            if industry in self.integrated_df['industry'].values:
                row = self.integrated_df[self.integrated_df['industry'] == industry].iloc[0]
                report_content.append(f"\n{industry}行业:")
                report_content.append(f"  • 薪资水平: {row.get('recruit_avg_salary', 'N/A'):.1f}K")
                report_content.append(f"  • 整体焦虑: {row.get('avg_anxiety', 'N/A'):.3f}")
                report_content.append(f"  • 匹配度: {row.get('uni_match_score', 'N/A'):.2f}")
                report_content.append(f"  • 竞争指数: {row.get('uni_competition_index', 'N/A'):.2f}")
                
                # 主要焦虑维度
                anxiety_dims = ['avg_career_confusion', 'avg_salary_gap', 'avg_involution', 'avg_identity_crisis']
                dim_names = ['发展迷茫', '薪酬落差', '职场内卷', '身份认同']
                
                max_anxiety_value = 0
                max_anxiety_dim = ''
                for dim, name in zip(anxiety_dims, dim_names):
                    if dim in row and not pd.isna(row[dim]):
                        report_content.append(f"  • {name}焦虑: {row[dim]:.3f}")
                        if row[dim] > max_anxiety_value:
                            max_anxiety_value = row[dim]
                            max_anxiety_dim = name
                
                if max_anxiety_dim:
                    report_content.append(f"  • 主要焦虑来源: {max_anxiety_dim}")
        
        # 5. 政策建议
        report_content.append("\n五、政策建议")
        report_content.append("-" * 50)
        
        report_content.append("1. 针对高薪高焦虑行业:")
        report_content.append("   • 企业: 优化工作环境，提供心理健康支持，改善工作生活平衡")
        report_content.append("   • 高校: 加强职场适应教育，帮助学生合理设定薪资期望")
        report_content.append("   • 个人: 提升心理调适能力，寻求工作与生活的平衡")
        
        report_content.append("\n2. 针对高竞争行业:")
        report_content.append("   • 企业: 建立公平的竞争机制，提供清晰的职业发展通道")
        report_content.append("   • 高校: 加强差异化培养，避免同质化竞争")
        report_content.append("   • 个人: 发展核心竞争力，避免盲目跟风竞争")
        
        report_content.append("\n3. 针对匹配度与焦虑的矛盾:")
        report_content.append("   • 高校: 平衡专业技能培养与心理素质教育")
        report_content.append("   • 企业: 合理设定岗位要求，避免过度筛选")
        report_content.append("   • 个人: 理性看待培养质量，注重实际能力提升")
        
        # 保存报告
        report_file = f"{output_dir}/职场焦虑核心分析报告.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print(f"✓ 核心分析报告已保存: {report_file}")

# 主程序
if __name__ == "__main__":
    print("="*80)
    print("职场焦虑核心分析系统")
    print("="*80)
    matplotlib.rc("font",family='FangSong')
    try:
        # 1. 初始化分析器
        analyzer = AnxietyCoreAnalyzer()
        
        # 2. 加载招聘市场数据
        print("\n1. 加载招聘市场数据...")
        recruitment_file = "cleaned_recruitment_data_all.csv"
        if os.path.exists(recruitment_file):
            analyzer.load_recruitment_data(recruitment_file)
        else:
            print(f"错误: 找不到招聘数据文件 {recruitment_file}")
            print("请确保文件在当前目录下")
            exit(1)
        
        # 3. 加载职场焦虑数据
        print("\n2. 加载职场焦虑数据...")
        anxiety_files = [
            "互联网.csv",
            "金融.csv", 
            "体制内.csv"
        ]
        
        analyzer.load_anxiety_data(anxiety_files)
        
        # 4. 加载高校培养数据
        print("\n3. 加载高校培养数据...")
        # 请根据实际文件名修改
        university_file = "高校培养数据.csv"  # 假设这是您的第三个数据源文件
        if os.path.exists(university_file):
            analyzer.load_university_data(university_file)
        else:
            # 尝试其他可能的文件名
            university_files = ["university_industry_data.csv", "高校行业数据.csv", "高校培养分析数据.csv"]
            loaded = False
            for file in university_files:
                if os.path.exists(file):
                    analyzer.load_university_data(file)
                    loaded = True
                    break
            
            if not loaded:
                print(f"警告: 找不到高校培养数据文件")
                print("请将高校培养数据文件放在当前目录下")
                # 可以继续执行，但分析会缺少高校数据
        
        # 5. 整合数据
        print("\n4. 整合数据源...")
        integrated_data = analyzer.integrate_all_data()
        
        if integrated_data is not None and len(integrated_data) > 0:
            # 6. 创建核心分析图表
            print("\n5. 创建核心分析图表...")
            
            # 创建输出目录
            output_dir = "职场焦虑核心分析"
            
            # 创建4个核心图表
            #analyzer.create_anxiety_salary_chart(output_dir)        # 图1: 焦虑和行业薪资关系
            #analyzer.create_match_anxiety_chart(output_dir)         # 图2: 匹配度和焦虑关系
            analyzer.create_competition_anxiety_chart(output_dir)   # 图3: 竞争指数和焦虑关系
            analyzer.create_anxiety_dimensions_chart(output_dir)    # 图4: 行业焦虑多维度对比
            
            # 7. 生成分析报告
            print("\n6. 生成分析报告...")
            analyzer.generate_core_analysis_report(output_dir)
            
            # 8. 保存整合数据
            integrated_data.to_csv(f'{output_dir}/核心分析数据.csv', index=False, encoding='utf-8-sig')
            print(f"✓ 核心分析数据已保存: {output_dir}/核心分析数据.csv")
            
            print("\n" + "="*80)
            print("职场焦虑核心分析完成!")
            print("="*80)
            print("\n生成的文件:")
            print(f"1. {output_dir}/核心分析数据.csv - 整合后的分析数据")
            print(f"2. {output_dir}/职场焦虑核心分析报告.txt - 文本分析报告")
            print(f"3. {output_dir}/1_行业薪资与焦虑关系图.png - 薪资与焦虑关系分析")
            print(f"4. {output_dir}/2_匹配度与焦虑关系图.png - 匹配度与焦虑关系分析")
            print(f"5. {output_dir}/3_竞争指数与焦虑关系图.png - 竞争指数与焦虑关系分析")
            print(f"6. {output_dir}/4_行业焦虑多维度对比图.png - 多维度焦虑对比分析")
            print(f"7. {output_dir}/4_补充_行业核心指标对比图.png - 核心指标对比分析")
        else:
            print("警告: 没有足够的整合数据进行分析")
        
    except Exception as e:
        print(f"分析出错: {e}")
        import traceback
        traceback.print_exc()