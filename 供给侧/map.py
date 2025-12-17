# ==================== 专业映射系统代码 ====================
import pandas as pd
import numpy as np
import re
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MajorIndustryMapper:
    """智能专业-行业映射系统"""
    
    def __init__(self, mapping_file=None):
        """初始化映射系统"""
        # 一级学科分类（教育部标准）
        self.discipline_categories = {
            '哲学': '人文社科',
            '经济学': '经管法',
            '法学': '经管法',
            '教育学': '人文社科',
            '文学': '人文社科',
            '历史学': '人文社科',
            '理学': '理学',
            '工学': '工学',
            '农学': '农学',
            '医学': '医学',
            '军事学': '特殊',
            '管理学': '经管法',
            '艺术学': '艺术'
        }
        
        # 二级学科到行业映射（基础映射）
        self.major_to_industry_base = {
            # 工学类
            '计算机科学与技术': '信息传输、软件和信息技术服务业',
            '软件工程': '信息传输、软件和信息技术服务业',
            '电子信息工程': '信息传输、软件和信息技术服务业',
            '通信工程': '信息传输、软件和信息技术服务业',
            '人工智能': '信息传输、软件和信息技术服务业',
            '数据科学与大数据技术': '信息传输、软件和信息技术服务业',
            '物联网工程': '信息传输、软件和信息技术服务业',
            '网络工程': '信息传输、软件和信息技术服务业',
            '信息安全': '信息传输、软件和信息技术服务业',
            
            '机械工程': '制造业',
            '机械设计制造及其自动化': '制造业',
            '车辆工程': '制造业',
            '智能制造工程': '制造业',
            '材料科学与工程': '制造业',
            '材料成型及控制工程': '制造业',
            '高分子材料与工程': '制造业',
            '金属材料工程': '制造业',
            '新能源材料与器件': '制造业',
            
            '电气工程及其自动化': '电力、热力、燃气及水生产和供应业',
            '能源与动力工程': '电力、热力、燃气及水生产和供应业',
            '新能源科学与工程': '电力、热力、燃气及水生产和供应业',
            
            '土木工程': '建筑业',
            '建筑学': '建筑业',
            '城乡规划': '建筑业',
            '风景园林': '建筑业',
            '给排水科学与工程': '建筑业',
            '建筑环境与能源应用工程': '建筑业',
            
            '化学工程与工艺': '制造业',
            '制药工程': '制造业',
            '环境工程': '水利、环境和公共设施管理业',
            '环境科学': '水利、环境和公共设施管理业',
            '生物工程': '科学研究和技术服务业',
            '食品科学与工程': '制造业',
            
            '纺织工程': '制造业',
            '服装设计与工程': '纺织服装、服饰业',
            '轻化工程': '制造业',
            
            # 理学类
            '数学与应用数学': '科学研究和技术服务业',
            '信息与计算科学': '信息传输、软件和信息技术服务业',
            '物理学': '科学研究和技术服务业',
            '应用物理学': '科学研究和技术服务业',
            '化学': '科学研究和技术服务业',
            '应用化学': '科学研究和技术服务业',
            '生物科学': '科学研究和技术服务业',
            '生物技术': '科学研究和技术服务业',
            '统计学': '科学研究和技术服务业',
            '应用统计学': '科学研究和技术服务业',
            '心理学': '卫生和社会工作',
            
            # 经管法类
            '经济学': '金融业',
            '金融学': '金融业',
            '国际经济与贸易': '金融业',
            '财政学': '金融业',
            '税收学': '金融业',
            '保险学': '金融业',
            '投资学': '金融业',
            
            '工商管理': '批发和零售业',
            '市场营销': '批发和零售业',
            '会计学': '金融业',
            '财务管理': '金融业',
            '人力资源管理': '公共管理、社会保障和社会组织',
            '审计学': '金融业',
            '资产评估': '金融业',
            '物业管理': '房地产业',
            
            '行政管理': '公共管理、社会保障和社会组织',
            '公共事业管理': '公共管理、社会保障和社会组织',
            '劳动与社会保障': '公共管理、社会保障和社会组织',
            '土地资源管理': '公共管理、社会保障和社会组织',
            
            '法学': '公共管理、社会保障和社会组织',
            '社会学': '公共管理、社会保障和社会组织',
            '政治学与行政学': '公共管理、社会保障和社会组织',
            
            # 人文社科类
            '汉语言文学': '文化、体育和娱乐业',
            '汉语国际教育': '教育',
            '英语': '教育',
            '日语': '教育',
            '德语': '教育',
            '法语': '教育',
            '翻译': '教育',
            '新闻学': '文化、体育和娱乐业',
            '广告学': '文化、体育和娱乐业',
            '传播学': '文化、体育和娱乐业',
            '编辑出版学': '文化、体育和娱乐业',
            '网络与新媒体': '信息传输、软件和信息技术服务业',
            
            '历史学': '教育',
            '考古学': '科学研究和技术服务业',
            '文物与博物馆学': '文化、体育和娱乐业',
            
            '哲学': '教育',
            '宗教学': '教育',
            
            '教育学': '教育',
            '学前教育': '教育',
            '小学教育': '教育',
            '特殊教育': '教育',
            '教育技术学': '教育',
            
            # 艺术类
            '音乐学': '文化、体育和娱乐业',
            '舞蹈学': '文化、体育和娱乐业',
            '戏剧影视文学': '文化、体育和娱乐业',
            '广播电视编导': '文化、体育和娱乐业',
            '播音与主持艺术': '文化、体育和娱乐业',
            '动画': '文化、体育和娱乐业',
            '美术学': '文化、体育和娱乐业',
            '绘画': '文化、体育和娱乐业',
            '雕塑': '文化、体育和娱乐业',
            '摄影': '文化、体育和娱乐业',
            '艺术设计学': '文化、体育和娱乐业',
            '视觉传达设计': '文化、体育和娱乐业',
            '环境设计': '文化、体育和娱乐业',
            '产品设计': '制造业',
            '服装与服饰设计': '纺织服装、服饰业',
            '数字媒体艺术': '信息传输、软件和信息技术服务业',
            
            # 医学类
            '临床医学': '卫生和社会工作',
            '口腔医学': '卫生和社会工作',
            '预防医学': '卫生和社会工作',
            '中医学': '卫生和社会工作',
            '针灸推拿学': '卫生和社会工作',
            '药学': '卫生和社会工作',
            '中药学': '卫生和社会工作',
            '医学检验技术': '卫生和社会工作',
            '医学影像技术': '卫生和社会工作',
            '护理学': '卫生和社会工作',
            '康复治疗学': '卫生和社会工作',
            
            # 农学类
            '农学': '农、林、牧、渔业',
            '园艺': '农、林、牧、渔业',
            '植物保护': '农、林、牧、渔业',
            '林学': '农、林、牧、渔业',
            '园林': '农、林、牧、渔业',
            '动物科学': '农、林、牧、渔业',
            '动物医学': '农、林、牧、渔业',
            '水产养殖学': '农、林、牧、渔业',
            
            # 交叉学科
            '生物医学工程': '科学研究和技术服务业',
            '机器人工程': '制造业',
            '智能科学与技术': '信息传输、软件和信息技术服务业',
            '区块链工程': '信息传输、软件和信息技术服务业',
            '虚拟现实技术': '信息传输、软件和信息技术服务业'
        }
        
        # 行业关键词映射（用于未明确映射的专业）
        self.industry_keywords = {
            '信息传输、软件和信息技术服务业': [
                '计算机', '软件', '网络', '信息', '数据', '互联网', '电子', '通信',
                '人工智能', '大数据', '云计算', '物联网', '区块链', '虚拟现实',
                '智能科学', '数字化', '信息化'
            ],
            '制造业': [
                '机械', '制造', '材料', '工程', '化工', '制药', '食品', '纺织',
                '服装', '轻工', '汽车', '船舶', '航空', '航天', '装备', '仪器',
                '自动化', '智能', '机器人', '模具', '铸造', '焊接'
            ],
            '金融业': [
                '金融', '经济', '会计', '财务', '银行', '证券', '保险', '投资',
                '税收', '财政', '审计', '资产评估', '风险管理'
            ],
            '教育': [
                '教育', '师范', '教师', '教学', '培训', '语言', '文学', '历史',
                '哲学', '德育', '心理', '幼儿', '小学', '中学', '大学'
            ],
            '文化、体育和娱乐业': [
                '艺术', '设计', '音乐', '舞蹈', '戏剧', '影视', '传媒', '新闻',
                '广告', '出版', '广播', '电视', '动画', '美术', '绘画', '雕塑',
                '摄影', '文创', '体育', '娱乐', '表演', '主持'
            ],
            '科学研究和技术服务业': [
                '科学', '研究', '技术', '实验', '测试', '检测', '分析', '统计',
                '物理', '化学', '生物', '数学', '地质', '天文', '海洋', '气象',
                '咨询', '设计', '规划', '勘察'
            ],
            '卫生和社会工作': [
                '医学', '医疗', '卫生', '健康', '护理', '康复', '保健', '药学',
                '中医', '西医', '口腔', '预防', '公共卫生', '养老', '社工'
            ],
            '建筑业': [
                '建筑', '土木', '工程', '规划', '景观', '园林', '市政', '路桥',
                '房地产', '装修', '装饰', '测绘', '勘察', '设计'
            ],
            '批发和零售业': [
                '商务', '贸易', '商业', '营销', '市场', '销售', '零售', '批发',
                '物流', '供应链', '电子商务', '跨境电商'
            ],
            '公共管理、社会保障和社会组织': [
                '管理', '行政', '公共', '社会', '保障', '劳动', '人力', '资源',
                '土地', '城市', '乡村', '社区', '组织', '政策', '法律', '法学',
                '政治', '国际关系', '外交'
            ]
        }
        
        # 反向映射：关键词到行业
        self.keyword_to_industry = {}
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                self.keyword_to_industry[keyword] = industry
        
        # 加载外部映射文件（如果有）
        if mapping_file:
            self.load_external_mapping(mapping_file)
    
    def load_external_mapping(self, file_path):
        """加载外部专业-行业映射文件"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    external_mapping = json.load(f)
            elif file_path.endswith('.csv'):
                external_mapping = pd.read_csv(file_path).set_index('major')['industry'].to_dict()
            else:
                print(f"不支持的文件格式: {file_path}")
                return
            
            self.major_to_industry_base.update(external_mapping)
            print(f"已加载外部映射: {len(external_mapping)} 条记录")
        except Exception as e:
            print(f"加载外部映射文件失败: {e}")
    
    def save_mapping(self, file_path):
        """保存当前映射到文件"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.major_to_industry_base, f, ensure_ascii=False, indent=2)
            elif file_path.endswith('.csv'):
                df = pd.DataFrame(list(self.major_to_industry_base.items()), 
                                 columns=['major', 'industry'])
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"映射已保存到: {file_path}")
        except Exception as e:
            print(f"保存映射失败: {e}")
    
    def map_major(self, major_name):
        """映射专业到行业"""
        if not major_name or pd.isna(major_name):
            return '其他'
        
        major_name = str(major_name).strip()
        
        # 1. 直接匹配
        if major_name in self.major_to_industry_base:
            return self.major_to_industry_base[major_name]
        
        # 2. 模糊匹配（包含关系）
        for mapped_major, industry in self.major_to_industry_base.items():
            if mapped_major in major_name or major_name in mapped_major:
                return industry
        
        # 3. 关键词匹配
        for keyword, industry in self.keyword_to_industry.items():
            if keyword in major_name:
                return industry
        
        # 4. 根据专业类别推断
        category = self.infer_major_category(major_name)
        if category:
            return self.map_category_to_industry(category)
        
        # 5. 使用机器学习预测（如果需要）
        # predicted = self.predict_with_ml(major_name)
        # if predicted:
        #     return predicted
        
        return '其他'
    
    def infer_major_category(self, major_name):
        """推断专业所属学科门类"""
        major_lower = major_name.lower()
        
        # 工程类专业
        if any(word in major_lower for word in ['工程', '技术', '科学', '制造', '材料', '机械', '电气', '电子', '信息', '计算机', '软件']):
            return '工学'
        
        # 理学类专业
        if any(word in major_lower for word in ['数学', '物理', '化学', '生物', '地理', '天文', '统计', '心理']):
            return '理学'
        
        # 经管法类专业
        if any(word in major_lower for word in ['经济', '金融', '管理', '会计', '财务', '营销', '行政', '法律', '法学', '政治']):
            return '经管法'
        
        # 人文社科类专业
        if any(word in major_lower for word in ['文学', '语言', '历史', '哲学', '教育', '新闻', '传播', '广告', '编辑']):
            return '人文社科'
        
        # 艺术类专业
        if any(word in major_lower for word in ['艺术', '设计', '音乐', '舞蹈', '戏剧', '影视', '美术', '绘画', '雕塑', '摄影']):
            return '艺术学'
        
        # 医学类专业
        if any(word in major_lower for word in ['医学', '医疗', '卫生', '护理', '康复', '药学', '中医']):
            return '医学'
        
        # 农学类专业
        if any(word in major_lower for word in ['农学', '园艺', '林学', '园林', '动物', '水产', '植物']):
            return '农学'
        
        return '其他'
    
    def map_category_to_industry(self, category):
        """将学科门类映射到主要行业"""
        category_industry_map = {
            '工学': '制造业',
            '理学': '科学研究和技术服务业',
            '经管法': '金融业',
            '人文社科': '教育',
            '艺术学': '文化、体育和娱乐业',
            '医学': '卫生和社会工作',
            '农学': '农、林、牧、渔业'
        }
        return category_industry_map.get(category, '其他')
    
    def map_multiple_majors(self, major_list):
        """批量映射专业列表"""
        results = {}
        for major in major_list:
            results[major] = self.map_major(major)
        return results
    
    def analyze_mapping_coverage(self, unique_majors):
        """分析映射覆盖率"""
        mapped = []
        unmapped = []
        
        for major in unique_majors:
            industry = self.map_major(major)
            if industry == '其他':
                unmapped.append(major)
            else:
                mapped.append(major)
        
        total = len(unique_majors)
        coverage_rate = len(mapped) / total * 100 if total > 0 else 0
        
        return {
            'total_majors': total,
            'mapped_count': len(mapped),
            'unmapped_count': len(unmapped),
            'coverage_rate': coverage_rate,
            'unmapped_majors': unmapped[:50]  # 只返回前50个未映射的
        }
    
    def suggest_mappings(self, unmapped_majors, limit=20):
        """为未映射的专业提供映射建议"""
        suggestions = []
        for major in unmapped_majors[:limit]:
            # 尝试分解专业名称
            words = re.split(r'[、与及和]', major)
            suggested_industry = None
            
            for word in words:
                word = word.strip()
                for keyword, industry in self.keyword_to_industry.items():
                    if keyword in word:
                        suggested_industry = industry
                        break
                if suggested_industry:
                    break
            
            suggestions.append({
                'major': major,
                'suggested_industry': suggested_industry or '需要手动映射'
            })
        
        return suggestions

def create_comprehensive_university_analysis():
    """创建全面的高校就业数据分析系统"""
    print("正在初始化高校就业数据分析系统...")
    
    # 1. 创建专业映射器
    mapper = MajorIndustryMapper()
    
    # 2. 加载高校数据
    print("正在加载高校就业数据...")
    
    try:
        # 加载示例数据（您可以用实际数据文件替换）
        employment_rate_df = pd.read_excel('高校就业详细数据.xlsx', sheet_name='就业率数据')
        employment_flow_df = pd.read_excel('高校就业详细数据.xlsx', sheet_name='就业流向数据')
        
        print(f"就业率数据: {employment_rate_df.shape}")
        print(f"就业流向数据: {employment_flow_df.shape}")
        
        # 3. 分析专业覆盖情况
        unique_majors = employment_rate_df['major'].dropna().unique()
        coverage = mapper.analyze_mapping_coverage(unique_majors)
        
        print(f"\n专业映射覆盖率分析:")
        print(f"总专业数: {coverage['total_majors']}")
        print(f"已映射: {coverage['mapped_count']}")
        print(f"未映射: {coverage['unmapped_count']}")
        print(f"覆盖率: {coverage['coverage_rate']:.1f}%")
        
        if coverage['unmapped_count'] > 0:
            print(f"\n前10个未映射的专业:")
            for major in coverage['unmapped_majors'][:10]:
                print(f"  - {major}")
            
            # 提供映射建议
            suggestions = mapper.suggest_mappings(coverage['unmapped_majors'], limit=5)
            print(f"\n映射建议:")
            for suggestion in suggestions:
                print(f"  {suggestion['major']} -> {suggestion['suggested_industry']}")
        
        # 4. 应用专业映射
        print(f"\n正在应用专业-行业映射...")
        employment_rate_df['industry'] = employment_rate_df['major'].apply(mapper.map_major)
        
        # 5. 保存映射结果
        mapper.save_mapping('专业行业映射表.json')
        
        # 6. 创建分析报告
        create_analysis_report(employment_rate_df, employment_flow_df, mapper)
        
        return employment_rate_df, employment_flow_df, mapper
        
    except Exception as e:
        print(f"数据分析出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_analysis_report(employment_rate_df, employment_flow_df, mapper):
    """创建分析报告"""
    print("\n" + "="*80)
    print("高校就业数据分析报告")
    print("="*80)
    
    # 基本统计
    print(f"\n1. 数据概览:")
    print(f"   - 数据年份: {employment_rate_df['year'].iloc[0] if 'year' in employment_rate_df.columns else '未知'}")
    print(f"   - 涉及学校: {employment_rate_df['school_name'].nunique() if 'school_name' in employment_rate_df.columns else '未知'}")
    print(f"   - 专业数量: {employment_rate_df['major'].nunique()}")
    print(f"   - 数据条数: {len(employment_rate_df):,}")
    
    # 毕业生统计
    total_graduates = employment_rate_df['graduate_number'].sum()
    total_employed = employment_rate_df['employment_number'].sum()
    overall_rate = (total_employed / total_graduates) * 100 if total_graduates > 0 else 0
    
    print(f"\n2. 毕业生总体情况:")
    print(f"   - 总毕业生数: {total_graduates:,}")
    print(f"   - 总就业人数: {total_employed:,}")
    print(f"   - 总体就业率: {overall_rate:.1f}%")
    
    # 分学历统计
    if 'education' in employment_rate_df.columns:
        print(f"\n3. 分学历就业情况:")
        edu_stats = employment_rate_df.groupby('education').agg({
            'graduate_number': 'sum',
            'employment_number': 'sum'
        })
        
        for edu, row in edu_stats.iterrows():
            edu_rate = (row['employment_number'] / row['graduate_number']) * 100 if row['graduate_number'] > 0 else 0
            print(f"   - {edu}: {row['graduate_number']:,}人, 就业率: {edu_rate:.1f}%")
    
    # 行业分布统计
    if 'industry' in employment_rate_df.columns:
        print(f"\n4. 行业分布情况:")
        industry_stats = employment_rate_df.groupby('industry').agg({
            'graduate_number': 'sum',
            'employment_number': 'sum'
        }).sort_values('graduate_number', ascending=False)
        
        for i, (industry, row) in enumerate(industry_stats.head(10).iterrows(), 1):
            rate = (row['employment_number'] / row['graduate_number']) * 100 if row['graduate_number'] > 0 else 0
            percentage = (row['graduate_number'] / total_graduates) * 100
            print(f"   {i:2d}. {industry[:20]:<20} {row['graduate_number']:>6,}人 ({percentage:5.1f}%), 就业率: {rate:5.1f}%")
    
    # 热门专业
    print(f"\n5. 热门专业Top 10:")
    major_stats = employment_rate_df.groupby('major').agg({
        'graduate_number': 'sum',
        'employment_number': 'sum',
        'employment_rate': 'mean'
    }).sort_values('graduate_number', ascending=False).head(10)
    
    for i, (major, row) in enumerate(major_stats.iterrows(), 1):
        print(f"   {i:2d}. {major:<20} {row['graduate_number']:>5,}人, 就业率: {row['employment_rate']:5.1f}%")
    
    # 高就业率专业
    print(f"\n6. 高就业率专业Top 10 (毕业生>50人):")
    high_rate_majors = employment_rate_df[employment_rate_df['graduate_number'] >= 50]
    if len(high_rate_majors) > 0:
        high_rate_stats = high_rate_majors.groupby('major').agg({
            'graduate_number': 'sum',
            'employment_rate': 'mean'
        }).sort_values('employment_rate', ascending=False).head(10)
        
        for i, (major, row) in enumerate(high_rate_stats.iterrows(), 1):
            print(f"   {i:2d}. {major:<20} {row['graduate_number']:>5,}人, 就业率: {row['employment_rate']:5.1f}%")
    
    # 就业流向分析
    if 'trade_name' in employment_flow_df.columns:
        print(f"\n7. 主要就业行业流向:")
        industry_flow = employment_flow_df[employment_flow_df['trade_name'].notna()]
        if not industry_flow.empty:
            total_flow = industry_flow['flow_number'].sum()
            top_industries = industry_flow.groupby('trade_name')['flow_number'].sum().sort_values(ascending=False).head(10)
            
            for i, (industry, count) in enumerate(top_industries.items(), 1):
                percentage = (count / total_flow) * 100
                print(f"   {i:2d}. {industry:<30} {count:>6,}人 ({percentage:5.1f}%)")
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)

def export_analysis_results(employment_rate_df, employment_flow_df, mapper):
    """导出分析结果"""
    print("\n正在导出分析结果...")
    
    # 1. 导出专业-行业映射关系
    unique_majors = employment_rate_df['major'].dropna().unique()
    mapping_results = []
    
    for major in unique_majors:
        industry = mapper.map_major(major)
        mapping_results.append({
            '专业名称': major,
            '所属行业': industry,
            '学科门类': mapper.infer_major_category(major)
        })
    
    mapping_df = pd.DataFrame(mapping_results)
    mapping_df.to_csv('专业行业映射结果.csv', index=False, encoding='utf-8-sig')
    print("✓ 专业行业映射结果已保存到: 专业行业映射结果.csv")
    
    # 2. 导出专业就业率统计
    major_summary = employment_rate_df.groupby(['major', 'education']).agg({
        'graduate_number': 'sum',
        'employment_number': 'sum',
        'employment_rate': 'mean'
    }).reset_index()
    
    # 添加行业信息
    major_summary['industry'] = major_summary['major'].apply(mapper.map_major)
    major_summary.to_csv('专业就业率统计.csv', index=False, encoding='utf-8-sig')
    print("✓ 专业就业率统计已保存到: 专业就业率统计.csv")
    
    # 3. 导出行业就业统计
    industry_summary = employment_rate_df.groupby('industry').agg({
        'graduate_number': 'sum',
        'employment_number': 'sum'
    }).reset_index()
    
    industry_summary['就业率'] = (industry_summary['employment_number'] / industry_summary['graduate_number']) * 100
    industry_summary = industry_summary.sort_values('graduate_number', ascending=False)
    industry_summary.to_csv('行业就业统计.csv', index=False, encoding='utf-8-sig')
    print("✓ 行业就业统计已保存到: 行业就业统计.csv")
    
    # 4. 导出就业流向统计
    if not employment_flow_df.empty and 'trade_name' in employment_flow_df.columns:
        flow_summary = employment_flow_df[employment_flow_df['trade_name'].notna()].copy()
        flow_summary.to_csv('就业流向统计.csv', index=False, encoding='utf-8-sig')
        print("✓ 就业流向统计已保存到: 就业流向统计.csv")
    
    # 5. 生成综合报告
    generate_comprehensive_report(employment_rate_df, employment_flow_df, mapping_df)
    
    print("\n所有分析结果已导出！")

def generate_comprehensive_report(employment_rate_df, employment_flow_df, mapping_df):
    """生成综合报告"""
    report_content = []
    
    # 报告头部
    report_content.append("高校就业数据分析综合报告")
    report_content.append("="*60)
    report_content.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"数据来源: {employment_rate_df['school_name'].nunique()} 所高校")
    report_content.append(f"数据年份: {employment_rate_df['year'].iloc[0] if 'year' in employment_rate_df.columns else '未知'}")
    report_content.append("")
    
    # 基本统计
    total_graduates = employment_rate_df['graduate_number'].sum()
    total_employed = employment_rate_df['employment_number'].sum()
    overall_rate = (total_employed / total_graduates) * 100
    
    report_content.append("一、总体概况")
    report_content.append(f"1. 总毕业生数: {total_graduates:,} 人")
    report_content.append(f"2. 总就业人数: {total_employed:,} 人")
    report_content.append(f"3. 总体就业率: {overall_rate:.1f}%")
    report_content.append(f"4. 涉及专业数: {employment_rate_df['major'].nunique()} 个")
    report_content.append("")
    
    # 行业分布
    if 'industry' in employment_rate_df.columns:
        report_content.append("二、行业分布")
        industry_stats = employment_rate_df.groupby('industry').agg({
            'graduate_number': 'sum',
            'employment_number': 'sum'
        }).sort_values('graduate_number', ascending=False)
        
        for industry, row in industry_stats.head(10).iterrows():
            rate = (row['employment_number'] / row['graduate_number']) * 100
            percentage = (row['graduate_number'] / total_graduates) * 100
            report_content.append(f"{industry:<25} {row['graduate_number']:>6,}人 ({percentage:5.1f}%) 就业率: {rate:5.1f}%")
        report_content.append("")
    
    # 热门专业
    report_content.append("三、热门专业Top 10")
    major_stats = employment_rate_df.groupby('major').agg({
        'graduate_number': 'sum',
        'employment_rate': 'mean'
    }).sort_values('graduate_number', ascending=False).head(10)
    
    for i, (major, row) in enumerate(major_stats.iterrows(), 1):
        report_content.append(f"{i:2d}. {major:<20} {row['graduate_number']:>5,}人 就业率: {row['employment_rate']:5.1f}%")
    report_content.append("")
    
    # 专业映射覆盖率
    total_majors = mapping_df['专业名称'].nunique()
    mapped_majors = mapping_df[mapping_df['所属行业'] != '其他']['专业名称'].nunique()
    coverage_rate = (mapped_majors / total_majors) * 100
    
    report_content.append("四、专业-行业映射情况")
    report_content.append(f"1. 总专业数: {total_majors}")
    report_content.append(f"2. 已映射专业数: {mapped_majors}")
    report_content.append(f"3. 映射覆盖率: {coverage_rate:.1f}%")
    report_content.append("")
    
    # 主要发现
    report_content.append("五、主要发现与建议")
    report_content.append("1. 部分专业就业率高但市场需求有限")
    report_content.append("2. 传统工科专业仍保持较高就业率")
    report_content.append("3. 新兴行业相关专业就业前景较好")
    report_content.append("4. 建议加强校企合作，提升实践能力")
    report_content.append("5. 关注行业发展趋势，调整专业设置")
    
    # 保存报告
    with open('高校就业分析报告.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print("✓ 综合报告已保存到: 高校就业分析报告.txt")

# 主程序
if __name__ == "__main__":
    print("="*80)
    print("高校就业数据分析系统")
    print("="*80)
    
    # 创建分析
    employment_rate_df, employment_flow_df, mapper = create_comprehensive_university_analysis()
    
    if employment_rate_df is not None:
        # 导出结果
        export_analysis_results(employment_rate_df, employment_flow_df, mapper)
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print("\n生成的文件:")
        print("1. 专业行业映射结果.csv - 专业到行业的映射关系")
        print("2. 专业就业率统计.csv - 各专业就业率统计")
        print("3. 行业就业统计.csv - 各行业就业情况统计")
        print("4. 专业行业映射表.json - 专业-行业映射表")
        print("5. 高校就业分析报告.txt - 综合分析报告")
        if not employment_flow_df.empty:
            print("6. 就业流向统计.csv - 就业行业流向统计")