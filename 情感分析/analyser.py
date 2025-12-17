"""
基于通义千问API的职场焦虑情感分析系统
生成量化职场焦虑指数：发展路径迷茫、薪酬落差、职场内卷、身份认同
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings
from tqdm import tqdm
import openai
from openai import OpenAI
import matplotlib as mpl
import time

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class WorkplaceAnxietyAnalyzer:
    """职场焦虑情感分析器"""
    
    def __init__(self, api_key: str = None, base_url: str = None, use_cache: bool = True):
        """
        初始化职场焦虑分析器
        
        Args:
            api_key: 通义千问API密钥
            base_url: API基础URL
            use_cache: 是否使用缓存
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "请设置API密钥! 方法:\n"
                "1. 传入api_key参数\n"
                "2. 设置环境变量: DASHSCOPE_API_KEY\n"
                "获取API密钥: https://dashscope.console.aliyun.com/apiKey"
            )
        
        # 通义千问API配置
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "qwen-turbo"  # 可选: qwen-turbo, qwen-plus, qwen-max
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # API调用统计
        self.api_calls = 0
        self.total_tokens = 0
        
        # 缓存设置
        self.use_cache = use_cache
        self.cache_file = './workplace_anxiety_cache.json'
        self.cache = self._load_cache() if use_cache else {}
        
        print(f"✓ 职场焦虑分析器初始化成功 (模型: {self.model})")
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"✓ 已加载缓存: {len(cache)} 条记录")
                return cache
            except Exception as e:
                print(f"⚠️ 缓存加载失败: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        if self.use_cache:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                print(f"✓ 缓存已保存: {len(self.cache)} 条记录")
            except Exception as e:
                print(f"⚠️ 缓存保存失败: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _call_api(self, messages: List[Dict], temperature: float = 0.3, 
                  max_tokens: int = 1500) -> str:
        """
        调用通义千问API
        
        Args:
            messages: 消息列表
            temperature: 温度参数(0-2), 越小越确定
            max_tokens: 最大生成token数
            
        Returns:
            API响应内容
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            self.api_calls += 1
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def analyze_workplace_anxiety(self, text: str, language: str = 'zh') -> Dict:
        """
        分析单条语料的职场焦虑指数
        
        Args:
            text: 语料文本
            language: 语言('zh'或'en')
            
        Returns:
            {
                'overall_anxiety': float (0-1, 整体焦虑指数),
                'dimensions': {
                    'career_path_confusion': float (0-1, 发展路径迷茫),
                    'salary_gap': float (0-1, 薪酬落差),
                    'workplace_involution': float (0-1, 职场内卷),
                    'identity_crisis': float (0-1, 身份认同)
                },
                'sentiment': str (positive/negative/neutral),
                'confidence': float (0-1, 分析置信度),
                'key_phrases': List[str] (关键焦虑表达),
                'explanation': str (分析解释)
            }
        """
        # 检查缓存
        cache_key = self._get_cache_key(f"anxiety_{language}_{text}")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        lang_text = "中文" if language == 'zh' else "English"
        
        prompt = f"""请分析以下{lang_text}职场相关文本中的焦虑情绪，并量化四个维度的焦虑指数（0-1，0表示无焦虑，1表示极度焦虑）：

文本内容：
"{text}"

请分析以下四个维度的焦虑程度：

1. 发展路径迷茫：对职业发展方向不明确，缺乏晋升通道的焦虑感
2. 薪酬落差：对薪酬待遇不满意，与期望或同行有差距的焦虑感
3. 职场内卷：对工作压力大、竞争激烈、过度竞争的焦虑感
4. 身份认同：对自己在职场中的角色和归属感不确定的焦虑感

要求：
1. 每个维度给出0-1之间的数值评分
2. 计算整体焦虑指数（取四个维度的平均值）
3. 分析文本的情感倾向（positive/negative/neutral）
4. 提取关键焦虑表达短语
5. 简要解释分析依据

请以JSON格式返回，格式如下：
{{
    "overall_anxiety": 0.0-1.0,
    "dimensions": {{
        "career_path_confusion": 0.0-1.0,
        "salary_gap": 0.0-1.0,
        "workplace_involution": 0.0-1.0,
        "identity_crisis": 0.0-1.0
    }},
    "sentiment": "positive/negative/neutral",
    "confidence": 0.0-1.0,
    "key_phrases": ["短语1", "短语2"],
    "explanation": "分析解释"
}}"""

        messages = [
            {"role": "system", "content": "你是一个专业的职场心理学专家，擅长分析职场焦虑情绪和量化评估。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_api(messages, temperature=0.2, max_tokens=1000)
        
        try:
            # 提取JSON部分
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            
            # 确保数值在0-1范围内
            for key in result['dimensions']:
                result['dimensions'][key] = max(0, min(1, float(result['dimensions'][key])))
            result['overall_anxiety'] = max(0, min(1, float(result['overall_anxiety'])))
            result['confidence'] = max(0, min(1, float(result['confidence'])))
            
            # 保存到缓存
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"解析失败: {e}, 原始响应: {response[:200]}")
            # 返回默认值
            default_result = {
                'overall_anxiety': 0.5,
                'dimensions': {
                    'career_path_confusion': 0.5,
                    'salary_gap': 0.5,
                    'workplace_involution': 0.5,
                    'identity_crisis': 0.5
                },
                'sentiment': 'neutral',
                'confidence': 0.0,
                'key_phrases': [],
                'explanation': '分析失败'
            }
            self.cache[cache_key] = default_result
            return default_result
    
    def batch_analyze_anxiety(self, texts: List[str], language: str = 'zh', 
                              batch_size: int = 20, delay: float = 1.0) -> List[Dict]:
        """
        批量分析职场焦虑
        
        Args:
            texts: 语料文本列表
            language: 语言
            batch_size: 每批处理数量
            delay: 批次间延迟(秒)
            
        Returns:
            分析结果列表
        """
        print(f"\n=== 开始批量分析职场焦虑 (共{len(texts)}条语料) ===")
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch)} 条)")
            
            for j, text in enumerate(batch):
                try:
                    if not text or str(text).strip() == '':
                        results.append(None)
                        continue
                        
                    result = self.analyze_workplace_anxiety(str(text), language)
                    results.append(result)
                    
                    # 显示进度
                    progress = i + j + 1
                    print(f"  ✓ {progress}/{total} (焦虑指数: {result['overall_anxiety']:.2f})", end='\r')
                    
                except Exception as e:
                    print(f"\n  ✗ 错误 (第{i+j+1}条): {e}")
                    results.append(None)
            
            # 批次间延迟
            if i + batch_size < total:
                time.sleep(delay)
        
        # 保存缓存
        self._save_cache()
        
        print(f"\n✓ 批量分析完成! 有效分析: {len([r for r in results if r])}/{total}")
        return results
    
    def calculate_aggregate_indices(self, analysis_results: List[Dict]) -> Dict:
        """
        计算聚合焦虑指数
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            聚合指数统计
        """
        # 过滤无效结果
        valid_results = [r for r in analysis_results if r]
        if not valid_results:
            return {}
        
        # 初始化统计字典
        stats = {
            'total_samples': len(valid_results),
            'overall_mean': 0.0,
            'dimension_means': {
                'career_path_confusion': 0.0,
                'salary_gap': 0.0,
                'workplace_involution': 0.0,
                'identity_crisis': 0.0
            },
            'dimension_stds': {},
            'sentiment_distribution': {},
            'dimension_correlations': {},
            'high_anxiety_samples': []  # 高焦虑样本(整体指数>0.7)
        }
        
        # 收集数据
        overall_scores = []
        dimension_scores = {
            'career_path_confusion': [],
            'salary_gap': [],
            'workplace_involution': [],
            'identity_crisis': []
        }
        sentiments = []
        
        for i, result in enumerate(valid_results):
            overall_scores.append(result['overall_anxiety'])
            sentiments.append(result['sentiment'])
            
            for dim in dimension_scores:
                dimension_scores[dim].append(result['dimensions'][dim])
            
            # 标记高焦虑样本
            if result['overall_anxiety'] > 0.7:
                stats['high_anxiety_samples'].append({
                    'index': i,
                    'overall_anxiety': result['overall_anxiety'],
                    'dominant_dimension': max(result['dimensions'].items(), key=lambda x: x[1])[0]
                })
        
        # 计算统计量
        stats['overall_mean'] = np.mean(overall_scores)
        stats['overall_std'] = np.std(overall_scores)
        stats['overall_median'] = np.median(overall_scores)
        
        for dim in dimension_scores:
            stats['dimension_means'][dim] = np.mean(dimension_scores[dim])
            stats['dimension_stds'][dim] = np.std(dimension_scores[dim])
        
        # 情感分布
        from collections import Counter
        stats['sentiment_distribution'] = dict(Counter(sentiments))
        
        # 维度间相关性
        import pandas as pd
        df = pd.DataFrame({
            'career_path': dimension_scores['career_path_confusion'],
            'salary_gap': dimension_scores['salary_gap'],
            'workplace_involution': dimension_scores['workplace_involution'],
            'identity': dimension_scores['identity_crisis'],
            'overall': overall_scores
        })
        stats['dimension_correlations'] = df.corr().to_dict()
        
        return stats
    
    def generate_anxiety_radar_chart(self, analysis_results: List[Dict], 
                                     output_file: str = './anxiety_radar.png',
                                     show_individual: bool = False,
                                     title: str = "职场焦虑指数雷达图") -> None:
        """
        生成焦虑指数雷达图
        
        Args:
            analysis_results: 分析结果列表
            output_file: 输出文件路径
            show_individual: 是否显示个体轨迹
            title: 图表标题
        """
        # 过滤无效结果
        valid_results = [r for r in analysis_results if r]
        if not valid_results:
            print("⚠️ 没有有效分析结果可生成雷达图")
            return
        
        # 维度标签
        dimensions = ['发展路径迷茫', '薪酬落差', '职场内卷', '身份认同']
        dimensions_en = ['career_path_confusion', 'salary_gap', 'workplace_involution', 'identity_crisis']
        
        # 计算平均指数
        avg_scores = []
        for dim_en in dimensions_en:
            scores = [r['dimensions'][dim_en] for r in valid_results]
            avg_scores.append(np.mean(scores))
        
        # 计算分位数（用于显示变异范围）
        percentile_25 = []
        percentile_75 = []
        for dim_en in dimensions_en:
            scores = [r['dimensions'][dim_en] for r in valid_results]
            percentile_25.append(np.percentile(scores, 25))
            percentile_75.append(np.percentile(scores, 75))
        
        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        avg_scores += avg_scores[:1]  # 闭合图形
        percentile_25 += percentile_25[:1]
        percentile_75 += percentile_75[:1]
        angles += angles[:1]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 绘制平均指数
        ax.plot(angles, avg_scores, 'o-', linewidth=3, markersize=8, 
                color='#FF6B6B', label='平均焦虑指数')
        ax.fill(angles, avg_scores, alpha=0.25, color='#FF6B6B')
        
        # 绘制变异范围
        ax.fill_between(angles, percentile_25, percentile_75, 
                        alpha=0.1, color='#4ECDC4', label='25-75百分位区间')
        
        # 如果显示个体轨迹
        if show_individual and len(valid_results) <= 50:  # 避免过多轨迹
            for i, result in enumerate(valid_results[:20]):  # 最多显示20条
                individual_scores = [result['dimensions'][dim] for dim in dimensions_en]
                individual_scores += individual_scores[:1]
                ax.plot(angles, individual_scores, 'o-', linewidth=0.5, markersize=2, 
                       alpha=0.3, color='gray')
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=12, fontweight='bold')
        
        # 设置网格和范围
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置标题
        overall_avg = np.mean([r['overall_anxiety'] for r in valid_results])
        ax.set_title(f'{title}\n(样本数: {len(valid_results)}, 整体平均指数: {overall_avg:.3f})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        # 添加注释：最高和最低焦虑维度
        max_dim_idx = np.argmax(avg_scores[:-1])
        min_dim_idx = np.argmin(avg_scores[:-1])
        
        ax.annotate(f'最高焦虑维度:\n{dimensions[max_dim_idx]}\n({avg_scores[max_dim_idx]:.3f})',
                   xy=(angles[max_dim_idx], avg_scores[max_dim_idx]),
                   xytext=(angles[max_dim_idx]+0.3, avg_scores[max_dim_idx]+0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=10, fontweight='bold', color='red')
        
        ax.annotate(f'最低焦虑维度:\n{dimensions[min_dim_idx]}\n({avg_scores[min_dim_idx]:.3f})',
                   xy=(angles[min_dim_idx], avg_scores[min_dim_idx]),
                   xytext=(angles[min_dim_idx]-0.5, avg_scores[min_dim_idx]-0.15),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=10, fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 雷达图已保存: {output_file}")
    
    def generate_comprehensive_report(self, analysis_results: List[Dict], 
                                     stats: Dict, output_file: str = './anxiety_report.md') -> None:
        """
        生成综合分析报告
        
        Args:
            analysis_results: 分析结果列表
            stats: 聚合统计
            output_file: 输出文件路径
        """
        valid_results = [r for r in analysis_results if r]
        if not valid_results:
            print("⚠️ 没有有效分析结果可生成报告")
            return
        
        report = []
        report.append("# 职场焦虑情感分析报告\n")
        report.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        report.append(f"*分析引擎: 通义千问 ({self.model})*\n")
        report.append("---\n")
        
        # 1. 数据概况
        report.append("## 1. 数据概况\n")
        report.append(f"- **分析语料总数**: {len(analysis_results)} 条\n")
        report.append(f"- **有效分析结果**: {len(valid_results)} 条 ({len(valid_results)/len(analysis_results)*100:.1f}%)\n")
        report.append(f"- **分析时间范围**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 2. 整体焦虑指数
        report.append("\n## 2. 整体焦虑指数分析\n")
        report.append(f"- **平均焦虑指数**: {stats.get('overall_mean', 0):.3f}\n")
        report.append(f"- **标准差**: {stats.get('overall_std', 0):.3f}\n")
        report.append(f"- **中位数**: {stats.get('overall_median', 0):.3f}\n")
        
        # 焦虑等级分布
        anxiety_levels = {'低焦虑(0-0.3)': 0, '中焦虑(0.3-0.7)': 0, '高焦虑(0.7-1)': 0}
        for result in valid_results:
            anxiety = result['overall_anxiety']
            if anxiety <= 0.3:
                anxiety_levels['低焦虑(0-0.3)'] += 1
            elif anxiety <= 0.7:
                anxiety_levels['中焦虑(0.3-0.7)'] += 1
            else:
                anxiety_levels['高焦虑(0.7-1)'] += 1
        
        report.append("\n**焦虑等级分布**:\n")
        for level, count in anxiety_levels.items():
            percentage = count / len(valid_results) * 100
            report.append(f"- {level}: {count} 条 ({percentage:.1f}%)\n")
        
        # 3. 各维度焦虑分析
        report.append("\n## 3. 各维度焦虑指数分析\n")
        
        dimension_names = {
            'career_path_confusion': '发展路径迷茫',
            'salary_gap': '薪酬落差',
            'workplace_involution': '职场内卷',
            'identity_crisis': '身份认同'
        }
        
        report.append("| 维度 | 平均指数 | 标准差 | 排名 |\n")
        report.append("|------|----------|--------|------|\n")
        
        dimension_means = stats.get('dimension_means', {})
        sorted_dims = sorted(dimension_means.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (dim_en, mean_score) in enumerate(sorted_dims, 1):
            std_score = stats.get('dimension_stds', {}).get(dim_en, 0)
            dim_name = dimension_names.get(dim_en, dim_en)
            report.append(f"| {dim_name} | {mean_score:.3f} | {std_score:.3f} | {rank} |\n")
        
        # 4. 情感倾向分析
        report.append("\n## 4. 情感倾向分析\n")
        sentiment_dist = stats.get('sentiment_distribution', {})
        total_sentiments = sum(sentiment_dist.values())
        
        for sentiment, count in sentiment_dist.items():
            percentage = count / total_sentiments * 100
            report.append(f"- **{sentiment}**: {count} 条 ({percentage:.1f}%)\n")
        
        # 5. 高焦虑样本分析
        high_anxiety = stats.get('high_anxiety_samples', [])
        if high_anxiety:
            report.append("\n## 5. 高焦虑样本分析\n")
            report.append(f"- **高焦虑样本数**: {len(high_anxiety)} 条\n")
            
            # 主导焦虑维度统计
            dominant_counts = {}
            for sample in high_anxiety:
                dominant = sample['dominant_dimension']
                dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
            
            report.append("\n**主导焦虑维度分布**:\n")
            for dim_en, count in dominant_counts.items():
                dim_name = dimension_names.get(dim_en, dim_en)
                percentage = count / len(high_anxiety) * 100
                report.append(f"- {dim_name}: {count} 条 ({percentage:.1f}%)\n")
        
        # 6. 关键发现
        report.append("\n## 6. 关键发现与洞察\n")
        
        # 找到最高和最低的维度
        max_dim = max(dimension_means.items(), key=lambda x: x[1])
        min_dim = min(dimension_means.items(), key=lambda x: x[1])
        
        report.append(f"1. **最突出的焦虑维度**: {dimension_names.get(max_dim[0], max_dim[0])} "
                     f"(指数: {max_dim[1]:.3f})\n")
        report.append(f"2. **相对较低的焦虑维度**: {dimension_names.get(min_dim[0], min_dim[0])} "
                     f"(指数: {min_dim[1]:.3f})\n")
        
        # 相关性洞察
        correlations = stats.get('dimension_correlations', {})
        if correlations and 'overall' in correlations:
            overall_corr = correlations['overall']
            sorted_corr = sorted([(k, v) for k, v in overall_corr.items() if k != 'overall'], 
                               key=lambda x: abs(x[1]), reverse=True)
            
            if sorted_corr:
                strongest_dim = sorted_corr[0]
                dim_name = dimension_names.get(strongest_dim[0], strongest_dim[0])
                report.append(f"3. **与整体焦虑最相关的维度**: {dim_name} "
                             f"(相关系数: {strongest_dim[1]:.3f})\n")
        
        # 7. 建议措施
        report.append("\n## 7. 针对性建议措施\n")
        
        recommendations = {
            'career_path_confusion': [
                "开展职业规划培训和工作坊",
                "建立明确的晋升通道和发展路径",
                "提供职业导师制度",
                "定期进行职业发展评估"
            ],
            'salary_gap': [
                "进行市场薪酬调研，确保薪酬竞争力",
                "建立透明、公平的薪酬体系",
                "实施绩效与薪酬挂钩机制",
                "提供非货币性激励和福利"
            ],
            'workplace_involution': [
                "优化工作流程，减少无效竞争",
                "建立健康的企业文化，避免过度加班",
                "实施弹性工作制",
                "提供心理健康支持和压力管理培训"
            ],
            'identity_crisis': [
                "加强企业文化建设，增强员工归属感",
                "明确岗位职责和期望",
                "提供团队建设和沟通培训",
                "建立有效的反馈和认可机制"
            ]
        }
        
        for dim_en, mean_score in sorted_dims:
            if mean_score > 0.6:  # 对于高焦虑维度提供建议
                dim_name = dimension_names.get(dim_en, dim_en)
                report.append(f"\n### 针对{dim_name}的建议:\n")
                for i, rec in enumerate(recommendations.get(dim_en, []), 1):
                    report.append(f"{i}. {rec}\n")
        
        # 8. API使用统计
        report.append("\n## 8. AI分析统计\n")
        report.append(f"- **API调用次数**: {self.api_calls}\n")
        report.append(f"- **使用Token数**: {self.total_tokens:,}\n")
        report.append(f"- **预估成本**: ¥{self.estimate_cost():.4f}\n")
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✓ 分析报告已生成: {output_file}")
    
    def estimate_cost(self) -> float:
        """估算API调用成本(人民币)"""
        price_per_1k = {
            'qwen-turbo': 0.003,
            'qwen-plus': 0.004,
            'qwen-max': 0.04
        }
        
        rate = price_per_1k.get(self.model, 0.004)
        cost = (self.total_tokens / 1000) * rate
        
        return round(cost, 4)
    
    def get_usage_stats(self) -> Dict:
        """获取API使用统计"""
        return {
            'total_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'estimated_cost': self.estimate_cost()
        }


def main():
    """主函数：职场焦虑分析示例"""
    print("="*70)
    print("职场焦虑情感分析系统 (基于通义千问AI)")
    print("="*70)
    
    # 初始化分析器
    analyzer = WorkplaceAnxietyAnalyzer(
        api_key=None,  # 自动从环境变量读取
        use_cache=True
    )
    
    # 示例数据（实际应用中从文件加载）
    example_texts = [
        "每天都在重复同样的工作，看不到职业发展的方向，感觉自己被困住了。",
        "工作了三年，工资基本没涨，看到同行业的同学都比我高，心里很不是滋味。",
        "公司内卷严重，大家都在加班，不加班就像犯了错一样，压力很大。",
        "作为新人，总感觉融入不进去，找不到自己的位置和价值。",
        "领导只关心结果，不关心过程，感觉就是一颗螺丝钉，随时可以被替换。",
        "行业变化太快，学的东西很快就过时了，不知道未来该往哪个方向发展。",
        "工资涨幅跑不赢通货膨胀，实际购买力在下降，生活压力越来越大。",
        "团队内部竞争激烈，同事之间缺乏信任，每天上班都像上战场。",
        "工作内容与个人兴趣不符，每天都在做自己不擅长也不喜欢的事情。",
        "绩效考核不透明，努力工作的成果得不到认可，感觉很挫败。",
        # 可以添加更多语料...
    ]
    
    # 批量分析
    analysis_results = analyzer.batch_analyze_anxiety(
        texts=example_texts,
        language='zh',
        batch_size=5,
        delay=0.5
    )
    
    # 计算聚合指数
    stats = analyzer.calculate_aggregate_indices(analysis_results)
    
    # 生成雷达图
    analyzer.generate_anxiety_radar_chart(
        analysis_results=analysis_results,
        output_file='./workplace_anxiety_radar.png',
        show_individual=True,
        title="职场焦虑指数雷达分析图"
    )
    
    # 生成详细报告
    analyzer.generate_comprehensive_report(
        analysis_results=analysis_results,
        stats=stats,
        output_file='./workplace_anxiety_report.md'
    )
    
    # 显示分析结果摘要
    print("\n" + "="*70)
    print("分析结果摘要")
    print("="*70)
    
    if stats:
        print(f"总样本数: {stats['total_samples']}")
        print(f"整体平均焦虑指数: {stats['overall_mean']:.3f}")
        print(f"各维度平均焦虑指数:")
        for dim, score in stats['dimension_means'].items():
            dim_name = {
                'career_path_confusion': '发展路径迷茫',
                'salary_gap': '薪酬落差',
                'workplace_involution': '职场内卷',
                'identity_crisis': '身份认同'
            }.get(dim, dim)
            print(f"  - {dim_name}: {score:.3f}")
    
    # 显示API使用统计
    print("\nAPI使用统计:")
    usage_stats = analyzer.get_usage_stats()
    print(f"  调用次数: {usage_stats['total_calls']}")
    print(f"  使用Token: {usage_stats['total_tokens']:,}")
    print(f"  预估成本: ¥{usage_stats['estimated_cost']:.4f}")
    
    print("\n✓ 分析完成！")
    print(f"  雷达图已保存: ./workplace_anxiety_radar.png")
    print(f"  分析报告已保存: ./workplace_anxiety_report.md")
    print("="*70)


if __name__ == '__main__':
    main()