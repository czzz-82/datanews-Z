# ==================== 数据清洗代码 ====================
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# 读取原始数据
def load_and_clean_data(file_path):
    """
    加载并清洗招聘数据
    
    参数:
    file_path: CSV文件路径
    
    返回:
    清洗后的DataFrame
    """
    # 读取数据
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"原始数据形状: {df.shape}")
    
    # 解析薪资字段 - 将薪资转换为数值
    def parse_salary(salary_str):
        """
        解析薪资字符串，返回平均薪资(单位: K/月)
        支持格式: '5-8K', '6-7.5K', '300-3000元/时', '220-230元/天'
        """
        if pd.isna(salary_str):
            return np.nan
        
        # 统一为字符串
        salary_str = str(salary_str).strip()
        
        # 处理时薪和日薪（转换为月薪，假设每月工作22天，每天8小时）
        if '元/时' in salary_str:
            numbers = re.findall(r'(\d+\.?\d*)', salary_str)
            if len(numbers) >= 2:
                avg = (float(numbers[0]) + float(numbers[1])) / 2
                # 时薪转月薪：时薪 × 8小时 × 22天 / 1000
                return avg * 3 * 8 / 1000
            elif len(numbers) == 1:
                return float(numbers[0]) * 8 * 22 / 1000
        
        if '元/天' in salary_str:
            numbers = re.findall(r'(\d+\.?\d*)', salary_str)
            if len(numbers) >= 2:
                avg = (float(numbers[0]) + float(numbers[1])) / 2
                # 日薪转月薪：日薪 × 22天 / 1000
                return avg * 22 / 1000
            elif len(numbers) == 1:
                return float(numbers[0]) * 22 / 1000
        
        # 处理月薪
        if 'K' in salary_str:
            numbers = re.findall(r'(\d+\.?\d*)', salary_str)
            if len(numbers) >= 2:
                return (float(numbers[0]) + float(numbers[1])) / 2
            elif len(numbers) == 1:
                return float(numbers[0])
        
        return np.nan
    
    # 应用薪资解析函数
    df['薪资_月平均K'] = df['薪资'].apply(parse_salary)
    print(f"成功解析薪资的记录数: {df['薪资_月平均K'].notnull().sum()}/{len(df)}")
    
    # 处理经验要求 - 统一分类
    def categorize_experience(exp):
        if pd.isna(exp):
            return '经验不限'
        exp_str = str(exp)
        if '经验不限' in exp_str or '不限' in exp_str:
            return '经验不限'
        elif '1年以内' in exp_str or '不到1年' in exp_str:
            return '1年以内'
        elif '1-3年' in exp_str:
            return '1-3年'
        elif '3-5年' in exp_str:
            return '3-5年'
        elif '5-10年' in exp_str:
            return '5-10年'
        elif '10年以上' in exp_str:
            return '10年以上'
        else:
            return '经验不限'
    
    df['经验要求'] = df['经验'].apply(categorize_experience)
    
    # 处理学历要求 - 统一分类
    def categorize_education(edu):
        if pd.isna(edu):
            return '学历不限'
        edu_str = str(edu)
        if '不限' in edu_str:
            return '学历不限'
        elif '初中' in edu_str:
            return '初中及以下'
        elif '高中' in edu_str:
            return '高中'
        elif '本科' in edu_str:
            return '本科'
        elif '硕士' in edu_str:
            return '硕士'
        elif '博士' in edu_str:
            return '博士'
        else:
            return '学历不限'
    
    df['学历要求'] = df['学历'].apply(categorize_education)
    
    # 清理公司规模 - 提取数字
    def extract_company_size(size_str):
        if pd.isna(size_str):
            return np.nan
        numbers = re.findall(r'\d+', str(size_str))
        if numbers:
            return int(numbers[0])
        return np.nan
    
    df['公司规模_数值'] = df['公司规模'].apply(extract_company_size)
    
    # 清理行业字段
    df['行业'] = df['行业'].fillna('其他')
    
    # 创建技能标签列表
    def extract_tags(tags):
        if pd.isna(tags):
            return []
        if isinstance(tags, str):
            # 按逗号分割，去除空格
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        return []
    
    df['技能标签列表'] = df['技能标签'].apply(extract_tags)
    df['福利标签列表'] = df['福利标签'].apply(extract_tags)
    
    # 添加薪资范围字段
    def extract_salary_range(salary_str):
        """提取薪资范围的最低和最高值"""
        if pd.isna(salary_str):
            return np.nan, np.nan
        
        salary_str = str(salary_str).strip()
        
        # 处理月薪
        if 'K' in salary_str:
            numbers = re.findall(r'(\d+\.?\d*)', salary_str)
            if len(numbers) >= 2:
                return float(numbers[0]), float(numbers[1])
            elif len(numbers) == 1:
                return float(numbers[0]), float(numbers[0])
        
        return np.nan, np.nan
    
    # 添加薪资范围字段
    df['薪资_最低K'], df['薪资_最高K'] = zip(*df['薪资'].apply(extract_salary_range))
    
    print(f"数据清洗完成!")
    print(f"清洗后数据形状: {df.shape}")
    
    return df

# 主程序
if __name__ == "__main__":
    # 设置文件路径
    input_file = "全国岗位数据.csv"
    output_file = "cleaned_recruitment_data_all.csv"
    
    print("=== 开始数据清洗 ===")
    
    # 加载并清洗数据
    cleaned_df = load_and_clean_data(input_file)
    
    # 保存清洗后的数据
    cleaned_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"清洗后的数据已保存到: {output_file}")
    
    # 输出清洗后的基本信息
    print("\n=== 清洗后数据基本信息 ===")
    print(f"总记录数: {len(cleaned_df)}")
    print(f"列名: {cleaned_df.columns.tolist()}")
    print(f"\n前3行数据:")
    print(cleaned_df[['职位', '公司', '薪资', '薪资_月平均K', '行业', '经验要求', '学历要求']].head(3))
    
    # 统计信息
    print(f"\n=== 关键统计信息 ===")
    print(f"平均月薪: {cleaned_df['薪资_月平均K'].mean():.1f}K")
    print(f"月薪中位数: {cleaned_df['薪资_月平均K'].median():.1f}K")
    print(f"最高月薪: {cleaned_df['薪资_月平均K'].max():.1f}K")
    print(f"最低月薪: {cleaned_df['薪资_月平均K'].min():.1f}K")
    
    print(f"\n经验要求分布:")
    print(cleaned_df['经验要求'].value_counts())
    
    print(f"\n学历要求分布:")
    print(cleaned_df['学历要求'].value_counts())