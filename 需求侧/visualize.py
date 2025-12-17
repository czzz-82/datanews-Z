    # ==================== 可视化分析代码 (优化版 - 单独图表 + 保存功能) ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
import os
import platform
import matplotlib

warnings.filterwarnings('ignore')

# ==================== 创建输出文件夹 ====================
output_dir = "可视化分析结果"
charts_dir = os.path.join(output_dir, "图表")
os.makedirs(charts_dir, exist_ok=True)

# 创建分析结果文件
analysis_file = os.path.join(output_dir, "招聘市场分析报告.txt")

def save_analysis(text, mode='a'):
    """保存分析结果到文件"""
    with open(analysis_file, mode, encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)

# 初始化分析文件
save_analysis("=" * 60, 'w')
save_analysis("武汉招聘市场分析报告 - Z世代职业画像")
save_analysis("=" * 60)
save_analysis(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
save_analysis("=" * 60)

# ==================== 中文字体设置 ====================
def setup_chinese_font():
    """设置中文字体，解决中文显示问题"""
    import platform
    
    system = platform.system()
    font_paths = []
    
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc',
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        ]
    
    # 检查并添加找到的字体
    added_fonts = []
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                matplotlib.font_manager.fontManager.addfont(font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                added_fonts.append(font_name)
                print(f"成功添加字体: {font_name}")
            except Exception as e:
                print(f"添加字体失败 {font_path}: {e}")
    
    # 设置默认字体
    if added_fonts:
        plt.rcParams['font.sans-serif'] = added_fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"已设置字体: {added_fonts[0]}")
    else:
        print("警告: 未找到中文字体，图表可能无法正常显示中文")
    
    return added_fonts

# 设置中文字体
fonts_available = setup_chinese_font()
sns.set_style("whitegrid")

# ==================== 停用词设置（与之前相同） ====================
JOB_TYPE_STOPWORDS = {
    '全职', '兼职', '全职兼职均可', '线下办公', '办公室坐班', 
    '不接受居家办公', '接受居家办公', '坐班', '不坐班', '招学徒',
    '接受小白', '接受无经验', '无经验要求', '有经验', '就近分配',
    '就近安排', '包吃住', '包吃', '包住', '包食宿', '五险一金',
    '五险', '社保', '住宿免费', '宿舍有空调', '免费吃住', '当天安排住宿',
    '免费培训', '法定三薪', '带薪培训', '无需培训', '不加班', '正常下班',
    '早九晚六', '朝九晚六', '双休', '单休', '月休', '做六休一',
    '月休四天', '月休8-10天', '弹性工作制', '工作时间自由',
    '底薪加提成', '保底工资', '绩效奖金', '有无线网', '有WIFI',
    '环境好', '晋升透明', '晋升空间大', '团队氛围好', '员工旅游',
    '团建聚餐', '零食下午茶', '节日福利', '生日福利', '高温补贴',
    '交通补助', '餐补', '通讯补贴', '住房补贴', '夜班补助',
    '加班补助', '工龄奖', '全勤奖', '年终奖', '股票期权',
    '定期体检', '补充医疗保险', '意外险', '企业年金',"无销售性质","月结工资",
    "正常工作制","个人（To C）","不需要倒班", "不需要晚班/夜班","周末双休", "连锁店"
}

PLATFORM_STOPWORDS = {
    '抖音', '快手', '小红书', '淘宝', '京东', '拼多多', 'B站',
    '微博', '微信', '公众号', '视频号', '直播', '短视频',
    '美团', '饿了么', '滴滴', '58同城', 'BOSS直聘', '智联招聘',
    '前程无忧',"连锁餐饮店" ,"接受无采耳相关经验"
}

POSITION_STOPWORDS = {
    '招聘', '急招', '直招', '诚聘', '高薪', '岗位', '职位', '工作',
    '经验', '要求', '学历', '专业', '能力', '技能', '职责',
    '任职资格', '岗位职责', '任职要求', '职位描述', '工作内容',
    '薪资', '工资', '待遇', '福利', '补贴', '奖金', '提成',
    '公司', '企业', '单位', '机构', '门店', '店铺', '工厂',
    '车间', '办公室', '写字楼', '商城', '商场', '超市',
    '武汉', '市区', '市内', '本地', '城区', '开发区'
}

SKILL_GENERAL_STOPWORDS = {
    '掌握', '熟练', '精通', '熟悉', '了解', '具备', '拥有',
    '能够', '可以', '会', '懂', '擅长', '善于', '优秀',
    '良好', '较强', '一定', '相关', '相应', '类似'
}

ALL_STOPWORDS = (JOB_TYPE_STOPWORDS | PLATFORM_STOPWORDS | 
                  POSITION_STOPWORDS | SKILL_GENERAL_STOPWORDS)

SKILL_KEYWORDS_TO_KEEP = {
    'Python', 'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'SQL',
    '深度学习', '机器学习', '自然语言处理', '算法', '数据分析',
    '数据挖掘', '大数据', '人工智能', 'AI', '3DMAX', 'MAYA',
    'UE4', 'Unity', 'Photoshop', 'Illustrator', 'CAD', 'AutoCAD',
    'Catia', 'UG', 'Excel', 'PPT', 'Word', '办公软件',
    '电商运营', '新媒体运营', '内容运营', '用户运营', '产品运营',
    '市场营销', '品牌策划', '广告投放', 'SEO', 'SEM',
    '会计', '财务', '审计', '税务', '金融', '证券', '投资',
    '设计', '美工', 'UI', 'UX', '交互设计', '平面设计',
    '护士', '医生', '药师', '医疗', '健康', '护理',
    '教师', '教育', '培训', '教学', '课程开发',
    '销售', '营销', '推广', '商务', '客户经理', '业务',
    '客服', '售后', '售前', '技术支持', '咨询',
    '工程师', '技术', '开发', '编程', '代码', '测试',
    '项目管理', '产品经理', '项目经理', '团队管理'
}

def should_keep_keyword(keyword):
    """判断是否应该保留关键词"""
    if keyword in ALL_STOPWORDS:
        return False
    if keyword in SKILL_KEYWORDS_TO_KEEP:
        return True
    if len(keyword) <= 1:
        return False
    if any(char.isdigit() for char in keyword):
        return True
    skill_patterns = ['开发', '设计', '分析', '管理', '运营', '销售', '服务']
    if any(pattern in keyword for pattern in skill_patterns):
        if len(keyword) <= 4:
            return True
    return True

# ==================== 数据加载函数 ====================
def load_cleaned_data(file_path):
    """加载清洗后的数据"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"加载清洗后数据: {df.shape}")
    save_analysis(f"\n一、数据概览")
    save_analysis(f"   数据总量: {df.shape[0]} 行, {df.shape[1]} 列")
    save_analysis(f"   数据列: {', '.join(df.columns.tolist())}")
    return df

# ==================== 可视化函数（每个单独保存） ====================
def create_industry_heatmap(df):
    """创建行业热度分析图 - 单独保存"""
    plt.figure(figsize=(14, 10))
    
    # 统计各行业岗位数量
    industry_counts = df['行业'].value_counts().head(15)
    
    # 创建柱状图
    colors = plt.cm.Set3(np.linspace(0, 1, len(industry_counts)))
    bars = plt.bar(range(len(industry_counts)), industry_counts.values, color=colors)
    
    plt.title('1. 行业岗位热度分析 (Top 15)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('行业', fontsize=14)
    plt.ylabel('岗位数量', fontsize=14)
    plt.xticks(range(len(industry_counts)), industry_counts.index, rotation=45, ha='right', fontsize=11)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "1_行业岗位热度分析.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n二、行业热度分析 (Top 15)")
    for i, (industry, count) in enumerate(industry_counts.items(), 1):
        percentage = (count / len(df)) * 100
        save_analysis(f"   {i:2d}. {industry:20s}: {count:4d} 个岗位 ({percentage:.1f}%)")

def create_salary_by_industry(df):
    """创建各行业平均薪资分布图 - 单独保存"""
    plt.figure(figsize=(14, 10))
    
    # 按行业计算平均薪资
    industry_salary = df.groupby('行业')['薪资_月平均K'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
    industry_salary = industry_salary[industry_salary['count'] >= 2]  # 只保留至少有2个岗位的行业
    
    # 取平均薪资最高的15个行业
    top_industries = industry_salary.head(15)
    
    # 创建水平条形图
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(top_industries)))
    bars = plt.barh(range(len(top_industries)), top_industries['mean'], color=colors)
    
    plt.title('2. 各行业平均薪资分布 (Top 15)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('平均月薪 (K)', fontsize=14)
    plt.ylabel('行业', fontsize=14)
    plt.yticks(range(len(top_industries)), top_industries.index, fontsize=11)
    
    # 添加薪资数值和样本数
    for i, bar in enumerate(bars):
        width = bar.get_width()
        count = int(top_industries.iloc[i]['count'])
        std_val = top_industries.iloc[i]['std']
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}K ± {std_val:.1f} (n={count})', 
                ha='left', va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "2_各行业平均薪资分布.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n三、高薪行业分析 (Top 15)")
    for i, (industry, row) in enumerate(top_industries.iterrows(), 1):
        save_analysis(f"   {i:2d}. {industry:20s}: {row['mean']:6.1f}K/月 (±{row['std']:.1f}, n={int(row['count'])})")

def create_experience_salary_boxplot(df):
    """创建不同经验要求的薪资分布箱线图 - 单独保存（优化版）"""
    plt.figure(figsize=(14, 10))
    
    # 准备数据
    exp_order = ['经验不限', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
    exp_data = []
    exp_labels = []
    exp_counts = []
    
    for exp in exp_order:
        exp_salaries = df[df['经验要求'] == exp]['薪资_月平均K'].dropna().values
        if len(exp_salaries) > 0:
            # 过滤掉极端值（超过3个标准差）
            mean_val = np.mean(exp_salaries)
            std_val = np.std(exp_salaries)
            filtered_salaries = exp_salaries[(exp_salaries >= mean_val - 3*std_val) & 
                                             (exp_salaries <= mean_val + 3*std_val)]
            if len(filtered_salaries) > 0:
                exp_data.append(filtered_salaries)
                exp_labels.append(exp)
                exp_counts.append(len(exp_salaries))
    
    # 创建箱线图
    if exp_data:
        # 设置更宽的箱线图
        positions = np.arange(len(exp_data)) + 1
        width = 0.6  # 箱线图宽度
        
        # 创建箱线图，不使用异常点
        box = plt.boxplot(exp_data, positions=positions, widths=width, 
                          patch_artist=True, showfliers=False, whis=1.5)
        
        # 设置颜色
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(exp_data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # 设置箱线边框颜色和宽度
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for line in box[element]:
                line.set_color('black')
                line.set_linewidth(1.5)
        
        # 调整中位数线样式
        for median in box['medians']:
            median.set_color('red')
            median.set_linewidth(2.5)
        
        # 添加中位数标注
        for i, median in enumerate(box['medians']):
            x, y = median.get_xydata()[1]
            plt.text(x, y, f'{y:.1f}K', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='red')
        
        # 添加样本数标注（在顶部）
        for i, (data, count) in enumerate(zip(exp_data, exp_counts)):
            x_pos = positions[i]
            y_max = np.max(data)
            plt.text(x_pos, y_max + 0.5, f'n={count}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 设置x轴标签
        plt.xticks(positions, exp_labels, fontsize=12)
        
        # 设置y轴范围，留出一些空间
        all_data = np.concatenate(exp_data)
        y_min = np.min(all_data) - 1
        y_max = np.max(all_data) + 3
        plt.ylim(y_min, y_max)
        
        # 计算并添加统计线
        all_salaries = np.concatenate([df[df['经验要求'] == exp]['薪资_月平均K'].dropna().values for exp in exp_labels])
        overall_mean = np.mean(all_salaries)
        overall_median = np.median(all_salaries)
        
        plt.axhline(y=overall_mean, color='blue', linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'整体平均: {overall_mean:.1f}K')
        plt.axhline(y=overall_median, color='green', linestyle=':', alpha=0.7,
                   linewidth=2, label=f'整体中位: {overall_median:.1f}K')
        plt.legend(fontsize=12, loc='upper right')
    
    plt.title('3. 不同经验要求的薪资分布', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('经验要求', fontsize=14)
    plt.ylabel('月薪 (K)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "3_不同经验要求的薪资分布.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n四、经验要求与薪资关系")
    for i, (exp, data, count) in enumerate(zip(exp_labels, exp_data, exp_counts)):
        if len(data) > 0:
            median_val = np.median(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            save_analysis(f"   {exp:10s}: 样本={count}, 中位={median_val:.1f}K, 平均={mean_val:.1f}K, Q1-Q3=[{q1:.1f}-{q3:.1f}]K")

def extract_meaningful_keywords(df):
    """提取有意义的职业关键词（过滤停用词）"""
    all_keywords = []
    
    # 从职位名称中提取关键词
    for title in df['职位'].dropna():
        title_str = str(title)
        for sep in ['/', '、', '(', ')', '（', '）', '+', '，', ' ']:
            title_str = title_str.replace(sep, ' ')
        
        title_words = title_str.split()
        for word in title_words:
            if should_keep_keyword(word):
                all_keywords.append(word)
    
    # 从技能标签中提取关键词
    for skills in df['技能标签列表']:
        if isinstance(skills, str):
            if skills.startswith('[') and skills.endswith(']'):
                try:
                    import ast
                    skills_list = ast.literal_eval(skills)
                except:
                    skills_list = [s.strip().strip("'\"") 
                                  for s in skills.strip('[]').split(',') if s.strip()]
            else:
                skills_list = [s.strip() for s in skills.split(',') if s.strip()]
        else:
            skills_list = skills if isinstance(skills, list) else []
        
        for skill in skills_list:
            if should_keep_keyword(str(skill)):
                all_keywords.append(str(skill))
    
    # 从职位描述中提取关键词（如果存在）
    if '职位描述' in df.columns:
        for desc in df['职位描述'].dropna():
            desc_str = str(desc)
            import re
            tech_words = re.findall(r'[A-Za-z]+[0-9]*|[A-Za-z]+[A-Za-z0-9]*', desc_str)
            for word in tech_words:
                if len(word) > 2 and should_keep_keyword(word):
                    all_keywords.append(word)
    
    return all_keywords

def create_keyword_wordcloud(df):
    """创建关键词词云图 - 单独保存"""
    plt.figure(figsize=(16, 12))
    
    # 提取有意义的职业关键词
    all_keywords = extract_meaningful_keywords(df)
    
    if not all_keywords:
        plt.text(0.5, 0.5, '无足够关键词数据', ha='center', va='center', fontsize=14)
        plt.title('4. 职业关键词词云', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        print("警告: 没有提取到关键词")
    else:
        # 统计词频
        keyword_counts = Counter(all_keywords)
        print(f"提取到 {len(keyword_counts)} 个唯一关键词")
        
        # 获取前100个关键词
        top_keywords = dict(keyword_counts.most_common(100))
        
        # 创建词云
        try:
            # 设置中文字体路径
            font_path = None
            if fonts_available:
                for font_name in fonts_available:
                    for font_info in matplotlib.font_manager.fontManager.ttflist:
                        if font_info.name == font_name:
                            font_path = font_info.fname
                            break
                    if font_path and os.path.exists(font_path):
                        break
            
            # 如果没找到，尝试常见路径
            if not font_path or not os.path.exists(font_path):
                if platform.system() == 'Windows':
                    common_paths = ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']
                elif platform.system() == 'Darwin':
                    common_paths = ['/System/Library/Fonts/PingFang.ttc']
                else:
                    common_paths = ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']
                
                for path in common_paths:
                    if os.path.exists(path):
                        font_path = path
                        break
            
            # 创建词云
            wordcloud = WordCloud(
                width=1200, 
                height=800, 
                background_color='white',
                font_path=font_path,
                max_words=100,
                max_font_size=200,
                min_font_size=10,
                random_state=42,
                colormap='tab20c',
                collocations=False
            ).generate_from_frequencies(top_keywords)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('4. 职业关键词词云', fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            
            # 在图表下方添加关键词频率列表
            top_15 = list(keyword_counts.most_common(15))
            info_text = "Top 15关键词: "
            for i, (word, count) in enumerate(top_15, 1):
                info_text += f"{i}.{word}({count}) "
            
            plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=11, 
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
        except Exception as e:
            print(f"词云生成失败，使用条形图替代: {e}")
            # 降级方案：显示前20个关键词的条形图
            top_20 = dict(keyword_counts.most_common(20))
            if top_20:
                words = list(top_20.keys())
                counts = list(top_20.values())
                
                y_pos = np.arange(len(words))
                plt.barh(y_pos, counts, color=plt.cm.tab20c(np.linspace(0, 1, len(words))))
                plt.yticks(y_pos, words, fontsize=11)
                plt.title('4. 职业关键词Top 20', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('出现次数', fontsize=14)
                plt.grid(True, alpha=0.3, axis='x')
                
                # 添加数值标签
                for i, (word, count) in enumerate(zip(words, counts)):
                    plt.text(count + 0.1, i, f'{count}', va='center', fontsize=10)
            else:
                plt.text(0.5, 0.5, '无足够关键词数据', ha='center', va='center', fontsize=14)
                plt.title('4. 职业关键词分析', fontsize=16, fontweight='bold', pad=20)
                plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "4_职业关键词词云.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n五、职业关键词分析")
    if all_keywords:
        keyword_counts = Counter(all_keywords)
        top_10 = keyword_counts.most_common(10)
        for i, (word, count) in enumerate(top_10, 1):
            save_analysis(f"   {i:2d}. {word:20s}: {count:4d} 次")

def create_education_salary_analysis(df):
    """创建学历与薪资关系分析 - 单独保存（优化版）"""
    plt.figure(figsize=(14, 10))
    
    # 定义学历顺序
    edu_order = ['学历不限', '初中及以下', '高中', '本科', '硕士', '博士']
    
    # 收集数据，只保留有数据的学历类别
    edu_salary_data = []
    edu_labels = []
    edu_counts = []
    
    for edu in edu_order:
        edu_salary = df[df['学历要求'] == edu]['薪资_月平均K'].dropna().values
        if len(edu_salary) > 0:
            # 过滤掉极端值（超过3个标准差）
            mean_val = np.mean(edu_salary)
            std_val = np.std(edu_salary)
            filtered_salaries = edu_salary[(edu_salary >= mean_val - 3*std_val) & 
                                           (edu_salary <= mean_val + 3*std_val)]
            if len(filtered_salaries) > 0:
                edu_salary_data.append(filtered_salaries)
                edu_labels.append(edu)
                edu_counts.append(len(edu_salary))
    
    # 如果有数据，则绘制箱线图
    if edu_salary_data:
        # 设置更宽的箱线图
        positions = np.arange(len(edu_salary_data)) + 1
        width = 0.6  # 箱线图宽度
        
        # 创建箱线图，不使用异常点
        box = plt.boxplot(edu_salary_data, positions=positions, widths=width, 
                          patch_artist=True, showfliers=False, whis=1.5)
        
        colors = plt.cm.Pastel2(np.linspace(0, 1, len(edu_salary_data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # 设置箱线边框颜色和宽度
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for line in box[element]:
                line.set_color('black')
                line.set_linewidth(1.5)
        
        # 调整中位数线样式
        for median in box['medians']:
            median.set_color('red')
            median.set_linewidth(2.5)
        
        plt.title('5. 学历要求与薪资分布关系', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('学历要求', fontsize=14)
        plt.ylabel('月薪 (K)', fontsize=14)
        plt.xticks(positions, edu_labels, fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数据点数量标注和中位数
        for i, (data, count) in enumerate(zip(edu_salary_data, edu_counts)):
            median_val = np.median(data)
            mean_val = np.mean(data)
            
            # 在箱线图上方添加标注
            x_pos = positions[i]
            y_max = np.max(data)
            plt.text(x_pos, y_max + 1, f'n={count}\n中位:{median_val:.1f}K', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # 设置y轴范围，留出一些空间
        all_data = np.concatenate(edu_salary_data)
        y_min = np.min(all_data) - 1
        y_max = np.max(all_data) + 5
        plt.ylim(y_min, y_max)
        
        # 添加整体统计线
        all_salaries = np.concatenate([df[df['学历要求'] == edu]['薪资_月平均K'].dropna().values for edu in edu_labels])
        overall_mean = np.mean(all_salaries)
        overall_median = np.median(all_salaries)
        
        plt.axhline(y=overall_mean, color='blue', linestyle='--', alpha=0.7, 
                   linewidth=2, label=f'整体平均: {overall_mean:.1f}K')
        plt.axhline(y=overall_median, color='green', linestyle=':', alpha=0.7,
                   linewidth=2, label=f'整体中位: {overall_median:.1f}K')
        plt.legend(fontsize=12, loc='upper right')
        
        # 添加统计摘要
        stats_text = f'总样本数: {len(all_salaries):,}\n'
        stats_text += f'薪资范围: {np.min(all_salaries):.1f}K - {np.max(all_salaries):.1f}K'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        plt.text(0.5, 0.5, '没有足够的薪资数据进行分析', 
                ha='center', va='center', fontsize=14)
        plt.title('5. 学历要求与薪资分布关系', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "5_学历要求与薪资分布关系.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n六、学历要求与薪资关系")
    for i, (edu, data, count) in enumerate(zip(edu_labels, edu_salary_data, edu_counts)):
        if len(data) > 0:
            median_val = np.median(data)
            mean_val = np.mean(data)
            std_val = np.std(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            save_analysis(f"   {edu:10s}: 样本={count}, 中位={median_val:.1f}K, 平均={mean_val:.1f}K, Q1-Q3=[{q1:.1f}-{q3:.1f}]K")

def create_skills_analysis(df):
    """创建技能标签分析（过滤后的）- 单独保存"""
    plt.figure(figsize=(16, 12))
    
    # 提取有意义的技能关键词
    all_skills = []
    
    for skills in df['技能标签列表']:
        if isinstance(skills, str):
            if skills.startswith('[') and skills.endswith(']'):
                try:
                    import ast
                    skills_list = ast.literal_eval(skills)
                except:
                    skills_list = [s.strip().strip("'\"") 
                                  for s in skills.strip('[]').split(',') if s.strip()]
            else:
                skills_list = [s.strip() for s in skills.split(',') if s.strip()]
        else:
            skills_list = skills if isinstance(skills, list) else []
        
        for skill in skills_list:
            skill_str = str(skill)
            if should_keep_keyword(skill_str):
                all_skills.append(skill_str)
    
    # 统计词频
    skill_counts = Counter(all_skills)
    
    if not skill_counts:
        plt.text(0.5, 0.5, '没有足够的技能标签数据', 
                ha='center', va='center', fontsize=14)
        plt.title('6. 核心技能标签分析', fontsize=16, fontweight='bold')
        print("没有足够的技能标签数据")
    else:
        # 获取前25个技能并过滤
        top_skills = skill_counts.most_common(25)
        filtered_skills = []
        for skill, count in top_skills:
            if should_keep_keyword(skill) and count >= 2:
                filtered_skills.append((skill, count))
        
        # 取前20个
        filtered_skills = filtered_skills[:20]
        
        if filtered_skills:
            skills_names = [skill[0] for skill in filtered_skills]
            skills_counts = [skill[1] for skill in filtered_skills]
            
            # 创建颜色渐变
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(skills_names)))
            
            bars = plt.barh(range(len(skills_names)), skills_counts, color=colors)
            plt.title('6. Top 20 核心技能标签 (过滤后)', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('出现次数', fontsize=14)
            plt.ylabel('技能标签', fontsize=14)
            plt.yticks(range(len(skills_names)), skills_names, fontsize=12)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{width}', ha='left', va='center', fontsize=11, fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='x')
        else:
            plt.text(0.5, 0.5, '过滤后没有技能标签数据', 
                    ha='center', va='center', fontsize=14)
            plt.title('6. 核心技能标签分析', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "6_核心技能标签分析.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n七、核心技能标签分析 (Top 10)")
    if skill_counts:
        top_10_skills = skill_counts.most_common(10)
        for i, (skill, count) in enumerate(top_10_skills, 1):
            save_analysis(f"   {i:2d}. {skill:20s}: {count:4d} 次")

def create_salary_distribution(df):
    """创建薪资分布直方图 - 单独保存"""
    plt.figure(figsize=(14, 10))
    
    # 过滤掉异常薪资值
    valid_salaries = df['薪资_月平均K'].dropna()
    valid_salaries = valid_salaries[(valid_salaries > 1) & (valid_salaries < 100)]
    
    if len(valid_salaries) > 0:
        # 使用密度图+直方图
        n, bins, patches = plt.hist(valid_salaries, bins=30, edgecolor='black', 
                                   alpha=0.6, color='steelblue', density=False)
        
        # 添加KDE曲线
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(valid_salaries)
            x_range = np.linspace(valid_salaries.min(), valid_salaries.max(), 1000)
            plt.plot(x_range, kde(x_range) * len(valid_salaries) * (bins[1] - bins[0]), 
                    'r-', linewidth=2, label='密度曲线')
        except:
            pass
        
        plt.title('7. 薪资分布直方图', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('月薪 (K)', fontsize=14)
        plt.ylabel('岗位数量', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计线
        mean_salary = valid_salaries.mean()
        median_salary = valid_salaries.median()
        
        plt.axvline(mean_salary, color='red', linestyle='--', linewidth=2.5, 
                   label=f'平均薪资: {mean_salary:.1f}K')
        plt.axvline(median_salary, color='green', linestyle='--', linewidth=2.5,
                   label=f'中位薪资: {median_salary:.1f}K')
        plt.legend(fontsize=12)
        
        # 添加统计摘要
        stats_text = f'样本数: {len(valid_salaries):,}\n'
        stats_text += f'薪资范围: {valid_salaries.min():.1f}K - {valid_salaries.max():.1f}K\n'
        stats_text += f'标准差: {valid_salaries.std():.1f}K\n'
        stats_text += f'偏度: {valid_salaries.skew():.2f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # 添加百分位数标注
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(valid_salaries, p)
            plt.axvline(value, color='orange', linestyle=':', alpha=0.5)
            plt.text(value, plt.ylim()[1]*0.9, f'{p}%', 
                    ha='center', va='bottom', fontsize=10, color='orange')
    else:
        plt.text(0.5, 0.5, '没有足够的薪资数据', 
                ha='center', va='center', fontsize=14)
        plt.title('7. 薪资分布直方图', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "7_薪资分布直方图.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n八、薪资分布统计")
    if len(valid_salaries) > 0:
        save_analysis(f"   样本数量: {len(valid_salaries):,}")
        save_analysis(f"   平均月薪: {valid_salaries.mean():.1f}K")
        save_analysis(f"   中位月薪: {valid_salaries.median():.1f}K")
        save_analysis(f"   最高月薪: {valid_salaries.max():.1f}K")
        save_analysis(f"   最低月薪: {valid_salaries.min():.1f}K")
        save_analysis(f"   标准差: {valid_salaries.std():.1f}K")
        save_analysis(f"   25%分位数: {np.percentile(valid_salaries, 25):.1f}K")
        save_analysis(f"   75%分位数: {np.percentile(valid_salaries, 75):.1f}K")

def create_company_size_analysis(df):
    """创建公司规模分析 - 单独保存"""
    if '公司规模_数值' not in df.columns:
        print("没有公司规模数据")
        return
    
    plt.figure(figsize=(14, 10))
    
    # 过滤有效数据
    valid_sizes = df['公司规模_数值'].dropna()
    
    if len(valid_sizes) > 0:
        # 创建公司规模分布直方图
        plt.hist(valid_sizes, bins=20, edgecolor='black', alpha=0.7, color='teal')
        plt.title('8. 公司规模分布', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('公司规模 (人数)', fontsize=14)
        plt.ylabel('公司数量', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'公司数量: {len(valid_sizes):,}\n'
        stats_text += f'平均规模: {valid_sizes.mean():.0f}人\n'
        stats_text += f'中位规模: {valid_sizes.median():.0f}人\n'
        stats_text += f'最大规模: {valid_sizes.max():,}人'
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        plt.text(0.5, 0.5, '没有公司规模数据', 
                ha='center', va='center', fontsize=14)
        plt.title('8. 公司规模分布', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "8_公司规模分布.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n九、公司规模分析")
    if len(valid_sizes) > 0:
        save_analysis(f"   分析公司数量: {len(valid_sizes):,}")
        save_analysis(f"   平均公司规模: {valid_sizes.mean():.0f}人")
        save_analysis(f"   中位公司规模: {valid_sizes.median():.0f}人")
        save_analysis(f"   最小公司规模: {valid_sizes.min():.0f}人")
        save_analysis(f"   最大公司规模: {valid_sizes.max():,}人")

def create_experience_distribution(df):
    """创建经验要求分布图 - 单独保存"""
    plt.figure(figsize=(12, 8))
    
    # 统计经验要求分布
    exp_counts = df['经验要求'].value_counts()
    
    if len(exp_counts) > 0:
        # 创建饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_counts)))
        wedges, texts, autotexts = plt.pie(exp_counts.values, labels=exp_counts.index, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=colors, textprops={'fontsize': 12})
        
        # 设置百分比字体
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.title('9. 经验要求分布', fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        plt.legend(wedges, [f'{label}: {count}' for label, count in zip(exp_counts.index, exp_counts.values)],
                  title="经验要求", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    else:
        plt.text(0.5, 0.5, '没有经验要求数据', 
                ha='center', va='center', fontsize=14)
        plt.title('9. 经验要求分布', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "9_经验要求分布.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n十、经验要求分布")
    for exp, count in exp_counts.items():
        percentage = count / len(df) * 100
        save_analysis(f"   {exp:10s}: {count:4d} 个岗位 ({percentage:.1f}%)")

def create_education_distribution(df):
    """创建学历要求分布图 - 单独保存（已去除'学历不限'类别）"""
    plt.figure(figsize=(12, 8))
    
    # 统计学历要求分布，并过滤掉'学历不限'
    edu_counts = df['学历要求'].value_counts()
    
    # 过滤掉'学历不限'类别
    if '学历不限' in edu_counts.index:
        edu_counts = edu_counts.drop('学历不限')
    
    if len(edu_counts) > 0:
        # 创建饼图
        colors = plt.cm.Pastel2(np.linspace(0, 1, len(edu_counts)))
        wedges, texts, autotexts = plt.pie(edu_counts.values, labels=edu_counts.index, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=colors, textprops={'fontsize': 6})
        
        # 设置百分比字体
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.title('10. 学历要求分布 (已去除"学历不限"类别)', fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        plt.legend(wedges, [f'{label}: {count}' for label, count in zip(edu_counts.index, edu_counts.values)],
                  title="学历要求", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    else:
        plt.text(0.5, 0.5, '没有学历要求数据', 
                ha='center', va='center', fontsize=14)
        plt.title('10. 学历要求分布', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(charts_dir, "10_学历要求分布.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filepath}")
    plt.show()
    
    # 保存分析结果
    save_analysis(f"\n十一、学历要求分布 (已去除'学历不限')")
    for edu, count in edu_counts.items():
        percentage = count / len(df) * 100
        save_analysis(f"   {edu:10s}: {count:4d} 个岗位 ({percentage:.1f}%)")

def save_summary_data(df):
    """保存关键统计结果到CSV和Excel文件"""
    # 保存行业统计
    industry_counts = df['行业'].value_counts()
    summary_stats = pd.DataFrame({
        '行业': industry_counts.index,
        '岗位数量': industry_counts.values,
        '占比%': (industry_counts.values / len(df) * 100).round(1)
    })
    
    csv_path = os.path.join(output_dir, 'industry_summary.csv')
    summary_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"行业统计已保存到: {csv_path}")
    
    # 保存薪资统计
    if '薪资_月平均K' in df.columns:
        salary_stats = df.groupby('行业')['薪资_月平均K'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(1)
        salary_stats = salary_stats[salary_stats['count'] >= 2]  # 只保留至少有2个样本的行业
        
        salary_csv_path = os.path.join(output_dir, 'salary_by_industry.csv')
        salary_stats.to_csv(salary_csv_path, encoding='utf-8-sig')
        print(f"行业薪资统计已保存到: {salary_csv_path}")
    
    # 保存清洗后的数据为Excel格式
    try:
        excel_path = os.path.join(output_dir, 'cleaned_recruitment_data.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"清洗后的数据已保存为Excel格式: {excel_path}")
    except Exception as e:
        print(f"保存Excel失败: {e}")
    
    # 在分析报告中添加文件信息
    save_analysis(f"\n十二、输出文件说明")
    save_analysis(f"   1. 图表文件: 保存在 '{charts_dir}' 文件夹中")
    save_analysis(f"   2. 分析报告: {analysis_file}")
    save_analysis(f"   3. 行业统计: {csv_path}")
    if '薪资_月平均K' in df.columns:
        save_analysis(f"   4. 薪资统计: {salary_csv_path}")
    save_analysis(f"   5. 完整数据: {excel_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("武汉招聘市场可视化分析 - 单独图表版本")
    print("=" * 60)
    print(f"输出文件夹: {output_dir}")
    print(f"图表保存到: {charts_dir}")
    print(f"分析报告: {analysis_file}")
    print("=" * 60)
    
    # 设置全局字体
    matplotlib.rc("font", family='FangSong')
    print("已设置全局字体: FangSong")
    
    # 加载清洗后的数据
    cleaned_file = "cleaned_recruitment_data_all.csv"
    if not os.path.exists(cleaned_file):
        print(f"错误: 找不到数据文件 '{cleaned_file}'")
        print("请确保数据文件在当前目录中，或修改文件路径")
        exit(1)
    
    print("正在加载数据...")
    df = load_cleaned_data(cleaned_file)
    
    # 检查数据
    print(f"\n数据预览:")
    print(df[['职位', '公司', '薪资_月平均K', '行业', '经验要求', '学历要求']].head(3))
    
    # 1. 行业岗位热度分析
    print("\n" + "=" * 60)
    print("1. 正在生成行业岗位热度分析图...")
    create_industry_heatmap(df)
    
    # 2. 各行业平均薪资分布
    print("\n2. 正在生成各行业平均薪资分布图...")
    create_salary_by_industry(df)
    
    # 3. 不同经验要求的薪资分布
    print("\n3. 正在生成不同经验要求的薪资分布图...")
    create_experience_salary_boxplot(df)
    
    # 4. 职业关键词词云
    print("\n4. 正在生成职业关键词词云图...")
    create_keyword_wordcloud(df)
    
    # 5. 学历与薪资关系分析
    print("\n5. 正在生成学历要求与薪资分布关系图...")
    create_education_salary_analysis(df)
    
    # 6. 技能标签分析
    print("\n6. 正在生成技能标签分析图...")
    create_skills_analysis(df)
    
    # 7. 薪资分布直方图
    print("\n7. 正在生成薪资分布直方图...")
    create_salary_distribution(df)
    
    # 8. 公司规模分析
    print("\n8. 正在生成公司规模分析图...")
    create_company_size_analysis(df)
    
    # 9. 经验要求分布
    print("\n9. 正在生成经验要求分布图...")
    create_experience_distribution(df)
    
    # 10. 学历要求分布
    print("\n10. 正在生成学历要求分布图...")
    create_education_distribution(df)
    
    # 保存关键统计结果
    print("\n" + "=" * 60)
    print("正在保存统计结果和详细分析报告...")
    save_summary_data(df)
    
    # 添加分析总结
    save_analysis("\n" + "=" * 60)
    save_analysis("分析总结")
    save_analysis("=" * 60)
    save_analysis(f"1. 本次分析共处理 {len(df):,} 个招聘岗位")
    
    if '薪资_月平均K' in df.columns:
        valid_salaries = df['薪资_月平均K'].dropna()
        valid_salaries = valid_salaries[(valid_salaries > 1) & (valid_salaries < 100)]
        if len(valid_salaries) > 0:
            save_analysis(f"2. 平均薪资水平: {valid_salaries.mean():.1f}K/月")
            save_analysis(f"3. 薪资中位数: {valid_salaries.median():.1f}K/月")
    
    # 统计热门行业
    industry_counts = df['行业'].value_counts()
    if len(industry_counts) > 0:
        top_industry = industry_counts.index[0]
        save_analysis(f"4. 最热门行业: {top_industry} ({industry_counts.iloc[0]}个岗位)")
    
    # 统计经验要求
    exp_counts = df['经验要求'].value_counts()
    if len(exp_counts) > 0:
        top_exp = exp_counts.index[0]
        save_analysis(f"5. 最常见经验要求: {top_exp} ({exp_counts.iloc[0]}个岗位)")
    
    # 统计学历要求
    edu_counts = df['学历要求'].value_counts()
    if len(edu_counts) > 0:
        top_edu = edu_counts.index[0]
        save_analysis(f"6. 最常见学历要求: {top_edu} ({edu_counts.iloc[0]}个岗位)")
    
    save_analysis("=" * 60)
    save_analysis("分析完成！")
    save_analysis(f"所有图表已保存到: {charts_dir}")
    save_analysis(f"详细分析报告: {analysis_file}")
    save_analysis("=" * 60)
    
    print("\n" + "=" * 60)
    print("可视化分析完成！")
    print(f"所有图表已保存到: {charts_dir}")
    print(f"详细分析报告: {analysis_file}")
    print("=" * 60)