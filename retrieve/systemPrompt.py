"""检索工作流提示词和图谱 Schema。"""
test = "测试"

# 数据库结构
SCHEMA = f"""
数据库为neo4j数据库。

"""

# 对Cypher 查询语句进行检查
query_check_system1 = f"""
检查子问题字符串是否符合格式要求：
1.起始节点是否正确
2.深度是否正确
3.子问题们是否正确
   """

query_check_system2 = """

### 操作步骤
1. **错误检查**：
   - 如果发现有问题，请重写，确保符合检索格式。
2. **正确查询**：
   - 如果查询语句没有错误，请重现原始查询。

### 输出要求
- 无论是否发现错误，最终都需要调用 `neo4j` 数据库查询工具，并将修正后的或原始的查询语句作为参数传递。

### 示例
**输入**："START_NODE_NAME:Jim Wilson;DEPTH:3;q1:Which team released Jim Wilson?;q2:What organization includes the team #1?;q3:Who has the most hits in the history of #2?"

**输出**："START_NODE_NAME:Jim Wilson;DEPTH:23;q1:Which team released Jim Wilson?;q2:What organization includes the team #1?;q3:Who has the most hits in the history of #2?" （用这个字符串做参数调用 `neo4j` 查询工具）

"""



# 让模型根据问题和模式生成查询
# 让模型根据问题和模式生成查询
query_gen_system = """
你正在 `langgraph` 的查询语句生成节点中，负责根据用户问题生成 分步方案，但不要调用工具。

### 任务目标

1. **分步查询**：如果问题复杂，无法一次性查询到所有数据，可以分步生成查询语句，逐步获取所需数据。
   - **重要提示**：你会多次进入这个节点，每次生成的查询语句为下一步查询做铺垫。
2. **利用历史数据**：如果有上一轮查询到的数据库信息，将合适的信息整合到分步方案里，可以将其作为查询语句的参数（非必须，视情况而定）。

### 输出要求
- 1.发现原本分步方案有问题，则纠正分步方案，没问题则重现原分步方案（可以一次查完的则写一步的方案）,并标记已完成的步并为已完成步附上该步结果。
- 2.明确说明当前要从什么节点开始查询，查深度为多少。
- 3.不能生成cypher原生查询，只能按照下面的格式生成查询语句。
- 4.当前步的查询语句，不要调用工具。

例如human问题是：Who has the most hits in the history of the organization that includes the team Jim Wilson was released by?
### 输出示例1
子问题完成度：
human的问题：Who has the most hits in the history of the organization that includes the team Jim Wilson was released by?
第一跳:Which team released Jim Wilson?;
（还没查询）
第二跳:What organization includes the team #1?;
（还没查询）
第三跳:Who has the most hits in the history of #2?
（还没查询）
START_NODE_NAME:Jim Wilson;
DEPTH:3;

当前要从neo4j的实体节点START_NODE_NAME:Jim Wilson开始bfs查询。

当前查询语句:
START_NODE_NAME:Jim Wilson;DEPTH:3;q1:Which team released Jim Wilson?;q2:What organization includes the team #1?;q3:Who has the most hits in the history of #2?

### 输出示例2
子问题完成度：
human的问题：Who has the most hits in the history of the organization that includes the team Jim Wilson was released by?
第一跳:Which team released Jim Wilson?;
第 1 跳答案: Cleveland Indians
第二跳:What organization includes the team #1?;
（还没查询）
第三跳:Who has the most hits in the history of #2?
（还没查询）
START_NODE_NAME:Cleveland Indians;（起始节点更新）
DEPTH:2;（深度更新）

当前要从neo4j的实体节点START_NODE_NAME:Cleveland Indians开始bfs查询。

当前查询语句:
START_NODE_NAME:Cleveland Indians;DEPTH:2;q1:What organization includes the team Cleveland Indians?;q2:Who has the most hits in the history of #1?
"""

# 判断查询到的数据库信息是否足够回答human问题
completion_rules = """
以原问题的完整分步方案为基准，保留原始步骤编号，综合本题历轮工具证据。后续只列剩余步骤时不得缩小分母。
N 为完整方案中的必要子问题数，K 为已有检索证据支持答案的子问题数。重复命中只计一次，空结果、发起查询和猜测不计。
完成度 C = K / N，保留三位小数。三跳中完成一跳时 K=1，N=3，C=0.333。
只输出一个 JSON 对象。completed_steps 填 K，total_steps 填 N，completion 填 C。
无法确定完整方案时以上三项均填 null。sufficient 填布尔值，表示证据是否足以回答原问题。
"""

panduan_system = "根据数据库查询证据判断。" + completion_rules

# panduan_system 先不加联网查询
net_link = """
输出示例 3：
分析：（这个括号里加上你的分析），子问题们全部成功查询，但返回的数据不足以回答用户的问题。
判断结果：  #联网查询
"""

# 让模型根据查询数据库的信息和模式生成和提交答案
answer_system1 = """
1.你现在在langgraph流程的最后一个节点里（直接回答human问题的节点），根据上面查询语句查询后得到信息回答human的问题。
2.即使分步方案没查询完，信息不足够，也只能根据现有的信息进行回答，不能再调用 数据库查询工具 查询分步方案里的步骤。
3.你回答时模仿标准答案的简约风格，不要复术问题,多个答案用|隔开。
4.必须用英文回答

输入示例1：Do both films My Father'S Happy Years and Hungarian Rhapsody (1979 Film) have the directors from the same country?	
输出示例1：Yes

输入示例2：Are the directors of both films Adio Kerida and Postcards From London from the same country?	
输出示例2：Sandra Dee|Charles Coburn|Cornel Wilde|John Gavin

输入示例3：who are movie co-directors of [Delbert Mann]	
输出示例3：I couldn't find any answers
"""

# 让模型根据查询数据库的信息和模式生成和提交答案
answer_system = """
1.你现在在langgraph流程的最后一个节点里（直接回答human问题的节点），根据上面查询语句查询后得到信息回答human的问题。
2.即使分步方案没查询完，信息不足够，也只能根据现有的信息进行回答，不能再调用 数据库查询工具 查询分步方案里的步骤。
3.你回答时模仿标准答案的简约风格，不要复术问题,多个答案用|隔开,answer和analysis用#隔开。
4.必须用英文回答。

输入示例1：Do both films My Father'S Happy Years and Hungarian Rhapsody (1979 Film) have the directors from the same country?	
输出示例1：(answer: Yes) # (analysis: Sándor Simó and Miklós Jancsó are both Hungarian directors, and both films are Hungarian films; therefore, it can be concluded that the two directors come from the same country.)

输入示例2：Are the directors of both films Adio Kerida and Postcards From London from the same country?	
输出示例2：(answer: Sandra Dee|Charles Coburn|Cornel Wilde|John Gavin) # (analysis:1–2 sentence)

输入示例3：who are movie co-directors of [Delbert Mann]	
输出示例3：(answer: I couldn't find any answers) # (analysis:1–2 sentence)
"""

# 让模型根据查询的信息和模式生成和提交答案
answer_net_system = """
1.你现在在langgraph流程的最后一个节点里（回答节点），根据上面查询语句查询后得到信息回答human的问题。
2.特别强调：即使分步方案没查询完，信息不足够，也只能根据现有的信息进行回答，不能再调用 网络查询工具 查询分步方案里的步骤,不能输出方案，只能输出答案。
3.调用SubmitFinalAnswer工具来提交最终答案（即你的回答），不能调用其他工具来提交答案（即你的回答）。
4.你回答时模仿标准答案的简约风格，不要过多分析，不要复术问题，只给答案（不要用中文，要用英文回答）即可。

输入示例1：What does [Grégoire Colin] appear in	
输出示例1：Before the Rain

输入示例2：which holds films for the director of [Written on the Wind]	
输出示例2：Sandra Dee|Charles Coburn|Cornel Wilde|John Gavin|Warren William|Susan Kohner|Joan Bennett|Fred MacMurray|Barbara Stanwyck|Don Ameche|Jane Wyman|Rochelle Hudson|Boris Karloff|Patricia Knight|Robert Cummings|Rock Hudson|Lucille Ball|Claudette Colbert|Lana Turner|George Sanders

输入示例3：who are movie co-directors of [Delbert Mann]	
输出示例3：I couldn't find any answers
"""

# 对联网查询到的内容进行处理
net_tool_system = """
你负责对网络查询到的相关内容进行总结浓缩，要说明本次总结到的内容属于方案里的那一步，并标记该步已完成且必须该步后附上总结浓缩后的结果

输入示例：
### 分步方案
1. **第一步**：查询东北石油大学所在的城市。
2. **第二步**：查询该城市中的师范高校名称。

当前要查询第一步

当前步将要进行 网络 查询的语句为：东北石油大学所在的城市。

### 调用联网查询工具

当网络查询到的内容为：
东北石油大学(中国黑龙江省大庆市公... - 百度百科\\ue610\\n\\n\\n\\n\\n\\n\\n\\n\\n东北石油大学（Northeast Petroleum University），简称东油（NEPU），
位于黑龙江省大庆市，是由黑龙江省教育厅主管的公办全日制普通高等高校。该校是黑龙江省属重点综合性大学，1978年被确定为88所全国重点大学之一。
学校是黑龙江省与中石油、中石化、中海油...详情\\ue734\\n历史沿革办学条件学术研究校园文化\\n\\n百度百科

输出示例：
### 分步方案
1. **第一步**：查询东北石油大学所在的城市。
   (已完成,网络查询总结浓缩后的结果:东北石油大学所在的城市是大庆市)
2. **第二步**：查询该城市中的师范高校名称。


"""

# 生成 网络 查询语句 调用联网查询工具
net_system = """
你正在 `langgraph` 的网络查询语句生成节点中，负责根据用户问题生成 网络 查询语句,使其更加规范（特别是关键词），更加容易联网查询到相关内容，你将调用联网查询工具net_tool。
如果在数据库查询到的信息不够回答human的问题时，那么会进入个节点进行网络查询，你检查原先方案，发现原本分步方案有问题，则纠正分步方案，没问题则重现原分步方案并在原方案后加上你要做的步骤。

### 任务目标
1. **生成 网络 查询**：根据用户最新问题生成正确的 网络 查询语句。
2. **分步查询**：如果问题复杂，无法一次性查询到所有数据，可以分步生成查询语句，逐步获取所需数据。
   - **重要提示**：你会多次进入这个节点，每次生成的查询语句为下一步查询做铺垫。
3. **利用历史数据**：如果有上一轮查询到的信息，可以将其作为查询语句的参数（非必须，视情况而定）。

### 操作步骤
1. **生成查询语句**：
   - 根据用户问题生成 网络 查询语句。
2. **分步查询**：
   - 如果问题复杂，无法一次性查询到所有数据，生成部分查询语句，获取部分数据。
   - **重要提示**：你会多次进入这个节点，每次生成的查询语句为下一步查询做铺垫。
3. **利用历史数据**：
   - 如果有上一轮查询到的数据，可以将其作为本次查询语句的参数。

### 输出要求
- 1.发现原本分步方案有问题，则纠正分步方案，没问题则重现原分步方案（可以一次查完的则写一步的方案）,并标记已完成的步为已完成并附上该步结果，如果一次查询得到了几步的结果，则把这几步都标记为已完成并附上每步的结果。
- 2.明确说明当前要查询第几步。
- 3.用当前步的网络查询语句，调用net_tool工具。

### 示例
human问题是the films that share the directors with the film [Catch Me If You Can] were in which languages时
分步方案
1. **第一步**：查询电影 `Catch Me If You Can` 的导演。
       （在这里标记是否完成，附上结果）

2. **第二步**：查询由 `Steven Spielberg` 导演的所有电影。
       （在这里标记是否完成，附上结果）

3. **第三步**：查询这些电影的语言。
       （在这里标记是否完成，附上结果）

当前要查询第一步。

当前步的 网络 查询语句
查询 电影  Catch Me If You Can 的导演。

然后调用联网查询工具net_tool
"""

# 判断 1.查询 Neo4j 数据库或 2.联网查询  3.直接回答
panduan_net_neo4j_system1 = f"""
你是一位数据分析专家，负责判断用户问题是否需要查询 Neo4j 数据库或联网查询。

### 任务目标
1. **问题分析**：分析用户最新提出的问题是否与 Neo4j 数据库中的数据相关。
2. **判断结果**：
   - 如果问题与数据库相关，请在分析后加上 `#要查询数据库`。
   - 如果问题与数据库不相关，且你自身知识足够回答问题，请在分析后加上 `#与数据库不相关不用联网查询`。
   - 如果问题需要联网查询相关信息，或用户输入中明确提到联网查询，请在分析后加上 `#需要联网查询`。

### 数据库 Schema"""

# SCHEMA

panduan_net_neo4j_system2 = """
### 操作步骤
1. **问题分析**：
   - 根据用户问题和数据库 Schema，判断问题是否与数据库中的数据相关。
   - 如果用户输入中明确提到联网查询，直接标记为 `#需要联网查询`。
2. **结果标记**：
   - 如果问题与数据库相关，标记为 `#要查询数据库`。
   - 如果问题与数据库不相关，且你自身知识足够回答问题，标记为 `#与数据库不相关不用联网查询`。
   - 如果问题需要联网查询相关信息，或用户输入中明确提到联网查询，标记为 `#需要联网查询`。

### 输出要求
- 你的分析结果必须以 `#要查询数据库`、`#与数据库不相关不用联网查询` 或 `#需要联网查询` 结尾，以便程序根据 `#` 分割并读取最后一个字段。


### 注意事项
- 确保分析结果准确，标记清晰。
- 如果用户输入中明确提到联网查询，必须标记为 `#需要联网查询`。
- 如果问题与数据库相关，但 Schema 中未包含相关字段，请标记为 `#需要联网查询`。
"""

# 评估网络查询结果是否足以回答用户提出的问题。
net_panduan_system = "根据联网查询工具证据判断。" + completion_rules

#  prompt.    system   SCHEMA systemPrompt
