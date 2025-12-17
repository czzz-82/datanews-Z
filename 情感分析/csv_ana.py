"""
职场焦虑数据分析辅助模块
用于从CSV文件加载语料数据
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Optional
from datetime import datetime
from analyser import WorkplaceAnxietyAnalyzer

class WorkplaceDataLoader:
    """职场语料数据加载器"""
    
    def __init__(self, data_file: str, text_column: str = 'text'):
        """
        初始化数据加载器
        
        Args:
            data_file: 数据文件路径(CSV格式)
            text_column: 包含文本内容的列名
        """
        self.data_file = data_file
        self.text_column = text_column
        self.df = None
        
    def load_data(self, sample_size: Optional[int] = None) -> List[str]:
        """
        加载语料数据
        
        Args:
            sample_size: 采样数量(None=全部)
            
        Returns:
            文本内容列表
        """
        print(f"正在加载数据文件: {self.data_file}")
        
        try:
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_file, encoding=encoding)
                    print(f"✓ 使用编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("无法解码文件，请检查文件编码")
            
            print(f"数据加载成功: {len(self.df)} 条记录")
            
            # 检查文本列
            if self.text_column not in self.df.columns:
                available_columns = list(self.df.columns)
                print(f"警告: 未找到列 '{self.text_column}'")
                print(f"可用列: {available_columns}")
                
                # 尝试自动查找包含文本的列
                text_candidates = ['content', 'comment', 'review', 'description', 'text']
                for candidate in text_candidates:
                    if candidate in available_columns:
                        self.text_column = candidate
                        print(f"自动选择列: {self.text_column}")
                        break
                
                if self.text_column not in self.df.columns:
                    # 让用户选择或使用第一列
                    self.text_column = available_columns[0]
                    print(f"使用第一列: {self.text_column}")
            
            # 清理文本数据
            texts = self._clean_texts(self.df[self.text_column].tolist())
            
            # 采样
            if sample_size and sample_size < len(texts):
                texts = texts[:sample_size]
                print(f"采样 {sample_size} 条语料")
            
            return texts
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return []
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
        """清理文本数据"""
        cleaned = []
        for text in texts:
            if pd.isna(text):
                continue
                
            text_str = str(text).strip()
            if len(text_str) < 10:  # 过滤过短的文本
                continue
                
            # 清理常见问题
            text_str = text_str.replace('\r\n', ' ').replace('\n', ' ')
            text_str = ' '.join(text_str.split())  # 去除多余空格
            
            cleaned.append(text_str)
        
        print(f"文本清理完成: {len(cleaned)}/{len(texts)} 条有效")
        return cleaned
    
    def save_analysis_results(self, texts: List[str], results: List[Dict], 
                             output_file: str = './anxiety_analysis_results.csv'):
        """
        保存分析结果
        
        Args:
            texts: 原始文本列表
            results: 分析结果列表
            output_file: 输出文件路径
        """
        # 准备数据
        data = []
        for i, (text, result) in enumerate(zip(texts, results)):
            if result is None:
                continue
                
            row = {
                'id': i + 1,
                'text': text[:200] + '...' if len(text) > 200 else text,
                'overall_anxiety': result['overall_anxiety'],
                'career_path_confusion': result['dimensions']['career_path_confusion'],
                'salary_gap': result['dimensions']['salary_gap'],
                'workplace_involution': result['dimensions']['workplace_involution'],
                'identity_crisis': result['dimensions']['identity_crisis'],
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
                'key_phrases': '; '.join(result.get('key_phrases', [])),
                'explanation': result.get('explanation', '')[:100]
            }
            data.append(row)
        
        # 创建DataFrame并保存
        df_results = pd.DataFrame(data)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 分析结果已保存: {output_file}")
        print(f"  包含 {len(df_results)} 条有效记录")


# 实际使用示例
if __name__ == '__main__':
    # 1. 加载数据
    data_loader = WorkplaceDataLoader(
        data_file='selenium_微博数据_我是金融人，我的职场心态_53条.csv',  # 替换为实际文件路径
        text_column='内容'  # 替换为实际文本列名
    )
    
    # 加载全部数据或采样
    texts = data_loader.load_data(sample_size=200)  # 采样100条用于测试
    
    # 2. 初始化分析器
    analyzer = WorkplaceAnxietyAnalyzer(use_cache=True)
    
    # 3. 批量分析
    results = analyzer.batch_analyze_anxiety(
        texts=texts,
        language='zh',
        batch_size=10,
        delay=0.5
    )
    
    # 4. 计算统计
    stats = analyzer.calculate_aggregate_indices(results)
    
    # 5. 生成可视化
    analyzer.generate_anxiety_radar_chart(
        results,
        output_file='./workplace_anxiety_analysis.png',
        title="职场焦虑指数分析"
    )
    
    # 6. 保存结果
    data_loader.save_analysis_results(texts, results)