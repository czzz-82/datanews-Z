###
需求侧：包括对boss直聘岗位爬虫及数据分析代码，如果cookie失效需要手动替换；

##
crawler.py boss直聘爬虫

cleaned.py 数据清洗预处理

visualize.py 需求侧单源数据可视化分析（5.1）

industry——mapper.py 行业-岗位映射器

 
##
供给侧：包括对高校就业报告平台爬虫代码及集成分析，爬虫需要提供用户名及密码；

crawler.py 高校就业信息平台爬虫

satis.py 供给侧单源数据可视化分析（5.2）

map.py 行业-专业映射器

Integration.py 需求侧-供给侧集成分析（6.1）


##
情感分析：包括微博爬虫及调用千问大模型情感分析，并进行集成分析，api_key被设为环境变量；

crawler.py 微博语料爬虫

analyser.py 情感分析主体代码

ana_csv.py csv.py 文件处理接口

integration_emo.py 焦虑——竞争指数综合分析
