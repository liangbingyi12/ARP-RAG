#!/usr/bin/env python
# coding: utf-8
import os
import config
from config import (
    CHAT_API_KEY as OPENAI_API_KEY,
    CHAT_BASE_URL as base_url,
    CHAT_MODEL as model_name,
    DEEPSEEK_THINKING,
    DEEPSEEK_REASONING_EFFORT,
    MAIN_NEO4J_URI as uri,
    MAIN_NEO4J_USER as username,
    MAIN_NEO4J_PASSWORD as password,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_API_KEY,
    DASHSCOPE_EMBEDDING_MODEL,
    DASHSCOPE_API_KEY,
    DASHSCOPE_API_BASE,
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    START_LINE as start_line,
    END_LINE as end_line,
    INPUT_FILE as file_path,
    OUTPUT_FILE as output_file_path,
    ERROR_FILE as error_output_file_path,
)

os.environ["OPENAI_API_KEY"] = config.SDK_API_KEY
os.environ["OPENAI_API_BASE"] = config.SDK_BASE_URL
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

if not OPENAI_API_KEY or not base_url or not model_name:
    raise RuntimeError("OPENAI_API_KEY, base_url, and model_name must be set")



import threading

# 创建线程本地存储对象
thread_local = threading.local()
FN = []
# messages列表里只有2个类：ai类和tool类

# 历史消息
messages1 = []
stepNeo4j = 0
stepNet = 0
input_tokens = 0
output_tokens = 0
total_tokens = 0

# In[12]:


# file1.py
import systemPrompt
# test = systemPrompt.test
#
# print(test)  # 输出: Hello from file2!


# In[13]:


from neo4j import GraphDatabase

# In[14]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


# pip install langchain langchain-community langchain-openai pip install langgraph
# pip install bs4  用于从 HTML 和 XML 文件中提取数据的 Python 库。
# pip install -U duckduckgo-search 如果在jupyter运行pip 要 !pip install -U duckduckgo-search
# pip install langgraph langchain_community langchain_openai
# pip install neo4j   pip install baidusearch -i https://pypi.tuna.tsinghua.edu.cn/simple


import ast
import getpass
import os

from langchain_openai import ChatOpenAI
from openai import OpenAI


def is_deepseek_provider() -> bool:
    # 只要 model_name 里包含 deepseek，就按 DeepSeek 兼容接口处理；
    # 不再额外依赖 LLM_PROVIDER、base_url 或环境变量来判断模型厂商。
    return "deepseek" in model_name.lower()


def make_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)


def add_provider_options(kwargs, *, thinking=None, reasoning_effort=None, raw_client=False):
    if is_deepseek_provider():
        # 判断为 DeepSeek 后，会给请求体追加 DeepSeek 专用的 thinking 参数。
        # 默认 DEEPSEEK_THINKING="disabled"，用于关闭 thinking mode，避免接口要求回传 reasoning_content 导致 400 错误。
        thinking_type = (thinking or DEEPSEEK_THINKING).strip().lower()
        kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
        if thinking_type == "enabled":
            if raw_client:
                # 原生 OpenAI SDK 调用时，reasoning_effort 直接作为顶层参数传入。
                kwargs["reasoning_effort"] = reasoning_effort or DEEPSEEK_REASONING_EFFORT
            else:
                # LangChain 的 ChatOpenAI 调用时，reasoning_effort 放进 model_kwargs 透传给底层 SDK。
                kwargs["model_kwargs"] = {
                    "reasoning_effort": reasoning_effort or DEEPSEEK_REASONING_EFFORT
                }
    return kwargs


def make_chat_model(temperature=config.TEMPERATURE, *, thinking=None, reasoning_effort=None):
    """
    LangChain -> OpenAI SDK 参数透传规则：
    - reasoning_effort 通过 model_kwargs 传递，避免被 LangChain 当成未知顶层参数报警；
    - thinking 是 DeepSeek 扩展 body，必须放到 extra_body，不能放到 model_kwargs。
    """
    kwargs = dict(
        model=model_name,
        api_key=OPENAI_API_KEY,
        base_url=base_url,
        temperature=temperature,
    )
    add_provider_options(kwargs, thinking=thinking, reasoning_effort=reasoning_effort)
    return ChatOpenAI(**kwargs)


model = make_chat_model(temperature=config.TEMPERATURE)
llm = make_chat_model(temperature=config.TEMPERATURE)

import getpass


import numpy as np
from langchain_community.embeddings import DashScopeEmbeddings  # pip install dashscope

# 初始化DashScope Embeddings
embeddings = DashScopeEmbeddings(
    model=DASHSCOPE_EMBEDDING_MODEL,
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 使用API代理服务提高访问稳定性
embeddings.client.api_base = DASHSCOPE_API_BASE


def OpenAIEmbeddings():
    return embeddings


from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)  # 联网查询最符合要求的数据，并不需要自己输入网址地址

# In[2]:


# 引入typing模块中的Any类型，用于类型提示，表示可以是任何类型
from typing import Any

# 从langchain_core.messages模块中导入ToolMessage类，用于创建工具消息
from langchain_core.messages import ToolMessage

# 从langchain_core.runnables模块中导入RunnableLambda和RunnableWithFallbacks类
# RunnableLambda用于创建可执行的lambda函数
# RunnableWithFallbacks用于创建具有回退机制的可执行对象
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks

# 从langgraph.prebuilt模块中导入ToolNode类，用于创建工具节点
from langgraph.prebuilt import ToolNode


def sanitize_state_for_gemini(state):
    safe_messages = []

    for msg in state["messages"]:
        # 复制一份，避免直接污染 LangGraph 原始状态
        msg_copy = msg.copy(deep=True) if hasattr(msg, "copy") else msg

        if hasattr(msg_copy, "content"):
            content = msg_copy.content

            if content is None or str(content).strip() == "":
                if getattr(msg_copy, "tool_calls", None):
                    msg_copy.content = " "
                else:
                    # 没内容、也不是工具调用意图，直接丢弃
                    continue

        safe_messages.append(msg_copy)

    return {"messages": safe_messages}


# 定义一个函数create_tool_node_with_fallback，接收一个工具列表作为参数，返回一个带有回退机制的RunnableWithFallbacks对象
def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    创建一个工具节点，并为其配置回退机制以处理错误并将错误信息暴露给代理。
    """
    # 返回一个ToolNode实例，该实例使用了.with_fallbacks方法来添加回退处理
    # 参数是一个包含单个RunnableLambda实例的列表，该实例指向handle_tool_error函数
    # exception_key参数指定了在异常发生时使用的键名
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


# 定义一个函数handle_tool_error，接收一个状态字典作为参数，返回一个包含错误信息的消息字典
def handle_tool_error(state) -> dict:
    # 从状态字典中获取错误信息
    error = state.get("error")

    # 获取最后一次调用工具的信息
    # print("看看回退节点里handle_tool_error的messages[-2]："+state["messages"][-2])可能报错不是print输出不是str
    print("看看回退节点里handle_tool_error的messages[-1]：" + str(state["messages"][-1]))
    tool_calls = state["messages"][-1].tool_calls

    # 构建一个包含错误信息的消息列表，每个消息都是一个ToolMessage实例
    # ToolMessage实例包含了错误内容和对应的工具调用ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # 错误内容
                tool_call_id=tc["id"],  # 工具调用ID
            )
            for tc in tool_calls  # 遍历所有工具调用
        ]
    }


# In[ ]:


# In[ ]:


# In[3]:


from langchain_core.tools import tool
from neo4j import GraphDatabase

queryTest = """
MATCH (h:油井历史数据)
WHERE h.井号 CONTAINS 'C09' AND h.生产日期 STARTS WITH '2014'
RETURN h.平均日产液量
"""

queryTest = """
MATCH (n:油井历史数据) WHERE n.井号 = "QHD32-6-B26HS" AND n.生产日期 STARTS WITH "2010" RETURN n.井号 AS 井号, n.平均日产油量 AS 平均日产油量, n.生产日期 AS 生产日期
"""


# 原始数据
def convert(records: list):
    # 转换为目标格式的字符串
    formatted_string = '['
    for record in records:
        formatted_string += '{'
        for key, value in record.items():
            formatted_string += f"'{key}': '{value}', "
        formatted_string = formatted_string.rstrip(', ')  # 去掉最后一个逗号和空格
        formatted_string += '},'
    formatted_string = formatted_string.rstrip(',')  # 去掉最后一个逗号
    formatted_string += ']'

    # 输出结果
    print("看看列表转化字符串结果：", formatted_string)
    return formatted_string


@tool
def neo4j_query_tool(query: str) -> str:
    """
    这是一个neo4j数据库查询工具：执行一条 cypher 查询并获取结果。
    """
    r = """[]"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            resultY = session.run(query)
            print("看看返回的原始数据：")
            print(resultY)
            result = list(resultY)  # 重要：将迭代器转为列表
            """

            if len(result) == 1:
                result1 = result.copy()
                result.append(result1)
                print("list长度：  ", len(result))

            """
            # r = convert(result)
            # 原始字符串


    except Exception as e:
        print("看看错误信息", e)
        return f"Error: cypher Query failed. Please rewrite your query and try again.错误信息: {str(e)}"
    finally:
        driver.close()

    # app.state.messages_resultStr = r
    # print("看看返回的图谱查询数据：",result)
    return result


# 使用三重双引号包裹多行查询字符串
# print(neo4j_query_tool.invoke(queryTest))  # 注意这里没有分号


# 使用三重双引号包裹多行查询字符串
# print(neo4j_query_tool.invoke(queryTest))  # 注意这里没有分号

from openai import OpenAI
import json

# 初始化 OpenAI 客户端
client = make_openai_client()

# 定义方案
call_FN = """
### 分步方案
1. **第一步**：查询广东海洋大学所在的城市。
2. **第二步**：查询该城市中的师范高校名称。

### 当前要执行第一步
当前步的 网络 查询语句：广东海洋大学所在的城市。

### 调用联网查询工具
-------------------------------------------------------------------------
"""

from neo4j_allSearch import neo4j_allSearch
import traceback


@tool
def neo4j_query_tool(query: str) -> str:
    """
    这是一个neo4j数据库查询工具：执行一次数据库查询并获取结果。
    """

    try:
        neo4j_allSearch_result = neo4j_allSearch(query)
        return neo4j_allSearch_result
    except Exception as e:
        # 捕获异常并打印错误信息
        print("错误信息:", e)
        print("报错的行号：")
        traceback.print_exc()
        return f"Error: neo4j Query failed. Please rewrite your query and try again. 错误信息: {str(e)}"


from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper  # API 密钥的配置（虽然 DuckDuckGo 不需要）
from langchain.prompts import ChatPromptTemplate


@tool
def net_tool_DuckDuck(query: str) -> str:
    """
    联网查询工具，可以联网查询相关信息。
    """

    # 直接使用网络搜索（无需封装工具）
    search = DuckDuckGoSearchAPIWrapper()

    """
    results = search.run(query)
    #run()：返回简洁的文本摘要（适合直接展示）
    #results()：获取结构化数据（包含链接、标题等元信息）

    num = 0
    for result in results:
        num = num + 1
    print("查询到的字数",num)
    print("联网查询到的内容： ",results)  
    """

    # 获取原始结构化数据（可指定数量）
    detailed_results = search.results(query, max_results=config.WEB_SEARCH_RESULTS)
    num = 0
    for result in detailed_results:
        num = num + 1
        # print("查询到段数：",num," , 结果：",result )
    # print("联网查询到的内容： ",detailed_results)

    return str(detailed_results)


# 定义系统消息
net_tool_system = systemPrompt.net_tool_system

from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from baidusearch import baidusearch


def search_baidu(query):
    # 执行百度搜索
    results = baidusearch.search(query)
    # 只返回前5条结果
    return results[:config.WEB_DISPLAY_RESULTS]


from openai import OpenAI
import json

# 初始化 OpenAI 客户端
client = make_openai_client()

# 定义方案
call_FN = """
### 分步方案
1. **第一步**：查询广东海洋大学所在的城市。
2. **第二步**：查询该城市中的师范高校名称。

### 当前要执行第一步
当前步的 网络 查询语句：广东海洋大学所在的城市。

### 调用联网查询工具
-------------------------------------------------------------------------
"""


@tool
def net_tool(query: str) -> str:
    """
    联网查询工具，可以联网查询相关信息。
    """
    all_result = ""
    try:
        # 百度查询
        results = search_baidu(query)

        # 解析并打印结果
        results1 = ""  # 初始化 results1
        for result in results:
            results1 = results1 + f"标题: {result['title']}\n"
            results1 = results1 + f"摘要: {result['abstract']}\n"
            results1 = results1 + f"链接: {result['url']}\n"
            results1 = results1 + f"排名: {result['rank']}\n"
            results1 = results1 + "-" * 50 + "\n"
        results = results1

        if results:
            all_result = results
    except Exception as e:
        print(f"百度 Search 查询失败: {e}")

    try:
        # 尝试使用 DuckDuckGo 查询
        search_ddg = DuckDuckGoSearchAPIWrapper()
        ddg_results = search_ddg.results(query, max_results=config.DDG_RESULTS)
        if ddg_results:
            all_result = ddg_results
    except Exception as e:
        print(f"DuckDuckGo 查询失败: {e}")

    # 如果所有查询都失败，返回错误信息
    if all_result == "":
        return "Error：所有查询工具均无法获取结果，请检查网络或重试。"

    # 提取 content 里面方案
    # call_FN = '### 分步方案\n1. **第一步**：查询广东海洋大学所在的城市。\n2. **第二步**：查询该城市中的师范高校名称。\n\n### 当前要执行第一步\n当前步的 网络 查询语句：广东海洋大学所在的城市。\n\n### 调用联网查询工具\n-------------------------------------------------------------------------'
    call_FN = FN[-1]
    # print(all_result)

    # 初始化 OpenAI 客户端
    client = make_openai_client()

    # 调用 OpenAI API
    try:
        # 确保 all_result 是字符串
        if isinstance(all_result, (list, dict)):
            all_result = json.dumps(all_result, ensure_ascii=False)

        # 确保 messages 格式正确
        messages = [
            {"role": "system", "content": net_tool_system},
            {"role": "assistant", "content": call_FN},
            {"role": "assistant", "content": "下面是网络查询到的结果：\n" + all_result},
            {"role": "user", "content": "请对网络查询到的结果进行浓缩总结"},  # "请总结广东海洋大学所在的城市。"
        ]

        # 检查 messages 是否为空
        if not all(m.get("content") for m in messages):
            return "Error：输入内容为空，请检查输入或 API 调用。"

        # 禁用 thinking mode 修复 DeepSeek 400 错误
        raw_kwargs = dict(
            model=model_name,
            messages=messages,
            stream=False,
        )
        # 这里走原生 OpenAI SDK；如果 model_name 包含 deepseek，会自动补 extra_body={"thinking": {"type": "disabled"}}。
        # 如果以后把 DEEPSEEK_THINKING 改成 enabled，也会把 reasoning_effort 一起传给 DeepSeek。
        add_provider_options(raw_kwargs, raw_client=True)
        response = client.chat.completions.create(**raw_kwargs)
        # 获取 token 使用量
        usage = response.usage
        print("网络查询里浓缩总结里的输出：")

        # 获取 input_tokens, output_tokens, total_tokens
        input_tokens1 = usage.prompt_tokens
        output_tokens1 = usage.completion_tokens
        total_tokens1 = usage.total_tokens

        global input_tokens, output_tokens, total_tokens  # 声明 tokens 相关变量为全局变量
        # input_tokens1, output_tokens1, total_tokens1 = returnTokens(panduan_result)
        input_tokens += input_tokens1
        output_tokens += output_tokens1
        total_tokens += total_tokens1
        print(f"input_tokens: {input_tokens}")
        print(f"output_tokens: {output_tokens}")
        print(f"total_tokens: {total_tokens}")

        # 判断 response.choices 的长度
        if len(response.choices) > 0:
            # print("看看长度是多少：", len(response.choices))
            print(response.choices[0].message)
            return response.choices[0].message.content
        else:
            print("未生成任何结果，请检查输入或 API 调用。")
            return "未生成任何结果，请检查输入或 API 调用。"
    except Exception as e:
        print(f"API 请求失败: {e}")
        return f"API 请求失败: {e}"


# 测试查询
# print(net_tool.invoke("查询广东海洋大学所在的城市"))


SCHEMA = systemPrompt.SCHEMA

from langchain_core.tools import tool


@tool
def get_neo4j_schema_tool() -> str:
    """
    获取neo4j数据库的schema和写cypher语句时的注意事项
    """

    return SCHEMA


# print(get_neo4j_schema_tool.invoke(""))


# In[6]:


# 下面是一个完整的call节点，调用节点
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 定义系统消息
query_check_system = systemPrompt.query_check_system1 + SCHEMA + systemPrompt.query_check_system2

# 创建 ChatPromptTemplate
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("human", "{messages}")]
)

# 创建 ChatOpenAI 实例，并绑定工具
# 禁用 thinking mode
query_check = query_check_prompt | make_chat_model(temperature=config.TEMPERATURE)

testQuery = """
MATCH (h:油井历史数据)
WHERE h.井号 CONTAINS 'C09' AND h.生产日期 STARTS WITH '2014'
RETURN h.平均日产液量
"""

# print(query_check.invoke(testQuery))


query_check = query_check_prompt | make_chat_model(temperature=config.TEMPERATURE).bind_tools(
    tools=[neo4j_query_tool], tool_choice="required"
)
"""
query_check = query_check_prompt | ChatOpenAI(model=model_name, temperature=config.TEMPERATURE).bind_tools(
    [db_query_tool], tool_choice="required"
)报错string indices must be integers
"""

# query_check.invoke({"messages": [("user",testQuery )]})

"""

# 添加一个节点用于检查查询是否正确
def model_check_query(state: State) -> dict[str, list[AIMessage]]:

    #用3引号包起描述：使用此工具在执行查询之前双重检查查询是否正确。

    print("进入了节点model_check_query")
    result =  {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}  # 用最后一条信息（查询语句生成节点的信息就是最后一条）作为检查节点的输入
    print("如果model_check_query有数据库查询操作，那么 model_check_query 的数据库查询操作会在这条提示前做了")
    return result


# 添加一个节点，让模型在执行查询前检查查询
workflow.add_node("correct_query", model_check_query)  # 添加查询校验节点
"""

# In[7]:


# 下面是一个完整的call节点，调用节点
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 生成 网络 查询语句 调用联网查询工具
net_system = systemPrompt.net_system

# 创建 ChatPromptTemplate
net_prompt = ChatPromptTemplate.from_messages(
    [("system", net_system), ("human", "{messages}")]
)

# 创建 ChatOpenAI 实例，并绑定工具
# 禁用 thinking mode
net = net_prompt | make_chat_model(temperature=config.TEMPERATURE)

# print(net.invoke("""大象资料"""))


net = net_prompt | make_chat_model(temperature=config.TEMPERATURE).bind_tools(
    tools=[net_tool], tool_choice="required"
)

# net.invoke({"messages": [("user", """大象资料""")]})


"""
# 添加一个节点用于检查查询是否正确
def net_query_call(state: State) -> dict[str, list[AIMessage]]:

    #用3引号包起描述：使用此工具在执行查询之前双重检查查询是否正确。

    print("进入了节点net_query")
    result =  {"messages": [net.invoke({"messages": [state["messages"][-1]]})]}  # 用最后一条信息（查询语句生成节点的信息就是最后一条）作为检查节点的输入
    #print("如果model_check_query有数据库查询操作，那么 model_check_query 的数据库查询操作会在这条提示前做了")
    return result

# 添加一个节点，让模型在执行查询前检查查询
#workflow.add_node("net_query_call", net_query_call)  # 添加查询校验节点
"""

# In[13]:


from typing import Annotated, Literal  # 导入类型注解和字面量类型

from langchain_core.messages import AIMessage  # 导入 AI 消息类
from langchain_openai import ChatOpenAI  # 导入 OpenAI 的聊天模型

from pydantic import BaseModel, Field  # 导入 Pydantic 的基类和字段
from typing_extensions import TypedDict  # 导入扩展类型的字典

from langgraph.graph import END, StateGraph, START  # 导入工作流图的相关常量和类
from langgraph.graph.message import AnyMessage, add_messages  # 导入消息类和消息添加函数


# 定义智能体的状态
class State(TypedDict):
    assessment: dict
    messages: Annotated[list[AnyMessage], add_messages]  # 消息列表，使用 add_messages 进行扩展


# 定义一个新的工作流图
workflow = StateGraph(State)  # 创建一个新的工作流图实例


# 手动模仿ai调用工具
# 添加第一个工具调用的节点,获取所有表
def get_neo4j_schema_tool_call(state: State) -> dict[str, list[AIMessage]]:
    # print("进入了节点first_tool_call")
    return {
        "messages": [
            AIMessage(
                content=" ",  # 消息内容为空
                tool_calls=[
                    {
                        "name": "get_neo4j_schema_tool",  # 调用的工具名称
                        "args": {},  # 工具参数
                        "id": "call_hzx8ct17jljbceu4knk0v6oj",  # 工具调用 ID
                    }
                ],
            )
        ]
    }


# 手动模仿ai调用工具
# 添加第一个工具调用的节点,获取所有表
def neo4j_query_tool_call(state: State) -> dict[str, list[AIMessage]]:
    # print("进入了节点 neo4j_query_tool_call")
    return {
        "messages": [
            AIMessage(
                content=" ",  # 消息内容为空
                tool_calls=[
                    {
                        "name": "neo4j_query_tool",  # 调用的工具名称
                        "args": {},  # 工具参数
                        "id": "call_hzx8ct17jljbceu4knk0v6oj",  # 工具调用 ID
                    }
                ],
            )
        ]
    }


def net_query_call(state: State) -> dict[str, list[AIMessage]]:
    """
    使用此工具在执行查询之前双重检查查询是否正确。
    """

    # print("进入了节点 net_query_call")
    global stepNet  # 声明 stepNeo4j 为全局变量
    stepNet = stepNet + 1

    if not hasattr(thread_local, 'net_num'):  # 线程里不存在net_num则创建一个变量 net_num
        thread_local.net_num = 0  # 每个线程独立初始化 , 记录进入查询语句生成节点次数
    thread_local.net_num += 1
    print("第", thread_local.net_num, "次进入查询语句生成节点-------------------------------------")

    # print("进入了节点net_query")
    result = {"messages": [net.invoke({"messages": [state["messages"][-1]]})]}  # 用最后一条信息作为检查节点的输入

    # 提取 content
    aimessage = result['messages'][0]  # 获取 AIMessage 对象
    content = aimessage.content  # 从 AIMessage 对象中提取 content 属性

    # 将 result字典 添加到 FN 列表中

    FN.append(content)
    # print("看看net_query_call放进FN里的是什么")
    # print(FN[-1])

    # 打印 FN 列表的最后一个元素
    # print(FN[-1])
    return result  # 这里返回的是一个裸露没包装的元组，还不是message


# 添加一个节点用于检查查询是否正确
def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    使用此工具在执行查询之前双重检查查询是否正确。
    """
    # print("进入了节点model_check_query")
    result = {
        "messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}  # 用最后一条信息（查询语句生成节点的信息就是最后一条）作为检查节点的输入
    # print("如果model_check_query有数据库查询操作，那么 model_check_query 的数据库查询操作会在这条提示前做了")
    return result


"""
# 添加一个节点，让模型在执行查询前检查查询
workflow.add_node("correct_query", model_check_query)  # 添加查询校验节点
"""

# 添加一个call节点
workflow.add_node("net_query_call", net_query_call)  # 添加查询校验节点
# 添加工具的节点
workflow.add_node(
    "net_tool", create_tool_node_with_fallback([net_tool])  # 创建列表表工具节点
)

# 将第一个工具调用节点添加到工作流中 获取所有表
workflow.add_node("get_neo4j_schema_tool_call", get_neo4j_schema_tool_call)  # 添加第一个工具调用节点

# 添加前两个工具的节点
workflow.add_node(
    "get_neo4j_schema_tool", create_tool_node_with_fallback([get_neo4j_schema_tool])  # 创建列表表工具节点
)


# 定义一个工具类表示结束状态,最终答案提交给用户（人类）最终以结构化的形式返回sql？？？？？
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""
    final_answer: str = Field(..., description="The final answer to the user")  # 最终答案字段
    """
    ... 表示使用默认的其他参数设置，这句代码就是在 SubmitFinalAnswer 类中定义了一个名为 final_answer 的字符串属性，用于存放要传达给用户的最终答案，并且通过 Field 函数给这个属性添加了一个便于理解的描述信息。
    """


# 定义生成查询的节点
def query_gen_node(state: State):
    # 添加一个节点，让模型根据问题和模式生成查询
    query_gen_system = systemPrompt.query_gen_system

    query_gen_prompt = ChatPromptTemplate.from_messages(
        [("system", query_gen_system), ("placeholder", "{messages}")]  # 系统提示和占位符
    )
    # 禁用 thinking mode
    query_gen = query_gen_prompt | make_chat_model(temperature=config.TEMPERATURE)
    """
    query_gen = query_gen_prompt | ChatOpenAI(model=model_name, temperature=config.TEMPERATURE).bind_tools( #每个节点都有自己的prompt和大模型处理自己节点的事务
        tools = [SubmitFinalAnswer]  # 绑定 SubmitFinalAnswer 工具
    ) 
    """

    # print("进入了节点 query_gen_node")
    global stepNeo4j  # 声明 stepNeo4j 为全局变量
    stepNeo4j = stepNeo4j + 1
    # 调用查询生成器，生成 SQL 查询
    message = query_gen.invoke(sanitize_state_for_gemini(state))  # 调用查询语句生成器,整个信息流给查询语句生成节点看
    # print("看看query_gen里的message是元组还是完整的aimessage")
    # print(message)

    # -----------------------------提取方案-------------------------------
    # 提取 content
    # aimessage = message['messages'][0]  # 获取 AIMessage 对象
    content = message.content  # 从 AIMessage 对象中提取 content 属性

    FN.append(content)

    # 将 result字典 添加到 FN 列表中
    print("\n看看 query_gen_node 放进FN里的是什么：")
    print(FN[-1])
    # ------------------------------------------------------------
    # print("看看生成的查询语句message是什么：", message)
    if not hasattr(thread_local, 'num'):
        thread_local.num = 0  # 每个线程独立初始化 , 记录进入查询语句生成节点次数
    thread_local.num += 1
    print("第", thread_local.num, "次进入查询语句生成节点")

    # 有时，LLM 会幻想并调用错误的工具。我们需要捕获这种情况并返回错误消息。
    tool_messages = []  # 初始化一个空的工具消息列表

    # getattr检查last_message对象是否存在tool_calls属性，若存在则返回其值，否则返回None。不会触发异常
    if getattr(message, "tool_calls", None):
        # 检查生成的消息中是否有工具调用
        if message.tool_calls:
            # 遍历所有的工具调用
            for tc in message.tool_calls:
                print("query_gen的工具调用记录：", tc["name"])

            """
            # 检查调用的工具名称是否不是 SubmitFinalAnswer
                if tc["name"] != "SubmitFinalAnswer":
                    print("调用记录：",tc["name"],"；调用记录里没有SubmitFinalAnswer，则再调用sql生成工具重写sql语句")
                    # 如果调用了错误的工具，生成一个错误消息
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called用错了工具: {tc['name']}. Please fix your mistakes.你只能生成查询语句就行，不要调用工具.",
                            tool_call_id=tc["id"],  # 设置工具调用的 ID
                        )
                    )
            """

        else:
            # 如果没有工具调用，保持工具消息列表为空
            tool_messages = []

    # 返回生成的消息和工具消息
    return {"messages": [message] + tool_messages}  # 将生成的消息和工具消息合并后返回


# 将查询生成节点添加到工作流中
workflow.add_node("query_gen", query_gen_node)  # 添加查询生成节点

# 添加一个节点，让模型在执行查询前检查查询
workflow.add_node("correct_query", model_check_query)  # 添加查询校验节点

# 添加执行查询的节点
workflow.add_node("execute_query_tool", create_tool_node_with_fallback([neo4j_query_tool]))  # 添加执行查询节点
"""
当在 execute_query_tool 节点执行SQL查询时发生错误，错误会被捕获并返回给大模型（即query_gen节点），大模型会根据错误信息重新生成SQL语句，
然后再次进入 execute_query_tool 节点执行新的SQL语句。为了避免无限循环，通常会在代码中设置一个最大重试次数，
或者在某个条件满足时停止重试（例如，错误类型不再变化或达到一定的时间限制）。
"""

# 添加一个节点，让模型根据查询数据库的信息和模式生成和提交答案
answer_system = systemPrompt.answer_system
answer_prompt = ChatPromptTemplate.from_messages(
    [("system", answer_system), ("placeholder", "{messages}")]  # 系统提示和占位符
)
# 禁用 thinking mode
answer = answer_prompt | make_chat_model(temperature=config.TEMPERATURE).bind_tools(
    tools=[SubmitFinalAnswer]  # 绑定 SubmitFinalAnswer 工具
)


# 定义提交的节点
def answer_node(state: State):
    # 安全复制并清理消息
    safe_messages = []
    for msg in state["messages"]:
        if hasattr(msg, 'content') and (msg.content is None or str(msg.content).strip() == ""):
            # 若消息有 tool_calls，则设置一个空白字符作为 content
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                msg.content = " "  # Gemini 要求非空
            else:
                continue  # 丢弃既无内容又无 tool_calls 的消息
        safe_messages.append(msg)

    # 临时构建一个只包含安全消息的 state 副本
    temp_state = {"messages": safe_messages}

    message = answer.invoke(temp_state)  # 使用清理后的消息

    # ... 后续 tool_messages 判断不变 ...
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: 答案提交节点 没有 调用答案提交工具。",
                        tool_call_id=tc["id"],
                    )
                )
    return {"messages": [message] + tool_messages}


# 添加提交答案的节点
workflow.add_node("answer_call", answer_node)  # 添加执行查询节点

# 添加一个节点，让模型根据查询的信息和模式生成和提交答案
answer_net_system = systemPrompt.answer_net_system
answer_net_prompt = ChatPromptTemplate.from_messages(
    [("system", answer_net_system), ("placeholder", "{messages}")]  # 系统提示和占位符
)
# 禁用 thinking mode
answer_net = answer_net_prompt | make_chat_model(temperature=config.TEMPERATURE).bind_tools(
    tools=[SubmitFinalAnswer]  # 绑定 SubmitFinalAnswer 工具
)


# 定义提交的节点
def answer_net_node(state: State):
    # print("进入了节点 answer_net_node")
    message = answer_net.invoke(sanitize_state_for_gemini(
        state))  # 所有信息给它？不是只给最后一条吗？#估计invoke自动提取整个流程节点的content和ai调用信息和工具的输出信息 ，有时这里调用网络查询工具，不调用答案工具，这里严重出错

    # 有时，LLM 会幻想并调用错误的工具。我们需要捕获这种情况并返回错误消息。
    tool_messages = []  # 初始化一个空的工具消息列表

    # 检查生成的消息中是否有工具调用
    if message.tool_calls:
        # 遍历所有的工具调用
        for tc in message.tool_calls:
            # print("调用记录：",tc["name"])
            # 检查调用的工具名称是否不是 SubmitFinalAnswer
            if tc["name"] != "SubmitFinalAnswer":  # 不是提交工具则报错

                """
                SubmitFinalAnswer 工具通常应该在工作流的最后阶段调用，用于提交最终的答案。
                如果在生成查询的过程中调用了 SubmitFinalAnswer，
                这可能意味着生成查询的逻辑出现了错误，因为此时查询还没有被执行，结果也没有被验证。
                """
                # 如果调用了错误的工具，生成一个错误消息
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: 答案提交节点 没有 调用答案提交工具。",
                        tool_call_id=tc["id"],  # 设置工具调用的 ID
                    )
                )
    else:
        # 如果没有工具调用，保持工具消息列表为空
        tool_messages = []

    # 返回生成的消息和工具消息
    return {"messages": [message] + tool_messages}  # 将生成的消息和工具消息合并后返回


# 添加提交答案的节点
workflow.add_node("answer_net_node_call", answer_net_node)  # 添加执行查询节点

# 添加一个节点，让模型根据查询数据库的信息和模式生成和提交答案
panduan_system = systemPrompt.panduan_system

panduan_prompt = ChatPromptTemplate.from_messages(
    [("system", panduan_system), ("placeholder", "{messages}")]  # 系统提示和占位符,到时所有节点信息整体state会放进去
)
# 禁用 thinking mode
panduan = panduan_prompt | make_chat_model(temperature=config.TEMPERATURE)


# 定义一个条件边，决定是否继续工作流 # 有3分支：1.调用记录里只有SubmitFinalAnswer 到end节点 2.query_gen; 3.correct_query
class EvidenceAssessment(BaseModel):
    completed_steps: int | None = Field(ge=0, strict=True)
    total_steps: int | None = Field(gt=0, strict=True)
    completion: float | None = Field(ge=0, le=1)
    sufficient: bool = Field(strict=True)


def assess_evidence(state, chain, attempts, limit):
    global input_tokens, output_tokens, total_tokens
    message = chain.invoke(sanitize_state_for_gemini(state))
    usage = message.usage_metadata or {}
    input_tokens += usage.get("input_tokens", 0)
    output_tokens += usage.get("output_tokens", 0)
    total_tokens += usage.get("total_tokens", 0)
    try:
        payload = message.content.strip().removeprefix("```json").removesuffix("```").strip()
        assessment = EvidenceAssessment.model_validate_json(payload).model_dump()
        k, n = assessment["completed_steps"], assessment["total_steps"]
        if (k is None) != (n is None) or (n is not None and k > n):
            raise ValueError("无效的完成步数")
        assessment["completion"] = round(k / n, 3) if n else None
    except (AttributeError, ValueError) as exc:
        print("完成度解析失败", exc)
        assessment = dict(completed_steps=None, total_steps=None, completion=None, sufficient=False)
    enough = assessment["sufficient"] and not state["messages"][-1].content.startswith("Error:")
    assessment["stop_reason"] = "sufficient" if enough else "query_limit" if attempts >= limit else "continue"
    print("检索判断", assessment)
    return {"assessment": assessment}


def graph_assessment(state: State):
    return assess_evidence(state, panduan, thread_local.num, config.GRAPH_QUERY_LIMIT)


def should_continue(state: State) -> Literal["answer_call", "query_gen"]:
    if state["assessment"]["stop_reason"] in ("sufficient", "query_limit"):
        thread_local.num = 0
        return "answer_call"
    return "query_gen"


net_panduan_system = systemPrompt.net_panduan_system

net_panduan_prompt = ChatPromptTemplate.from_messages(
    [("system", net_panduan_system), ("placeholder", "{messages}")]  # 系统提示和占位符,到时所有节点信息整体state会放进去
)
# 禁用 thinking mode
net_panduan = net_panduan_prompt | make_chat_model(temperature=config.TEMPERATURE)


# 定义一个条件边，回答or继续网络查询
def web_assessment(state: State):
    return assess_evidence(state, net_panduan, thread_local.net_num, config.WEB_QUERY_LIMIT)


def net_should_continue(state: State) -> Literal["answer_net_node_call", "net_query_call"]:
    if state["assessment"]["stop_reason"] in ("sufficient", "query_limit"):
        thread_local.net_num = 0
        return "answer_net_node_call"
    return "net_query_call"


# 添加一个节点，让模型根据查询数据库的信息和模式生成和提交答案
panduan_net_neo4j_system = systemPrompt.panduan_net_neo4j_system1 + SCHEMA + systemPrompt.panduan_net_neo4j_system2

panduan_net_neo4j_prompt = ChatPromptTemplate.from_messages(
    [("system", panduan_net_neo4j_system), ("placeholder", "{messages}")]  # 系统提示和占位符,到时所有节点信息整体state会放进去
)
# 禁用 thinking mode
panduan_net_neo4j = panduan_net_neo4j_prompt | make_chat_model(temperature=config.TEMPERATURE)


# 定义一个条件边，决定是否继续工作流 # 有3分支：1.调用记录里只有SubmitFinalAnswer 到end节点 2.query_gen; 3.correct_query
def should_net_neo4j_continue(state: State) -> Literal[
    "get_neo4j_schema_tool_call", "answer_net_node_call", "net_query_call"]:
    return "get_neo4j_schema_tool_call"
    # print("进入了判断函数 should_net_neo4j_continue")
    messages = state["messages"]  # 获取所有消息
    # 或完整对象转字符串（调试时推荐）
    # print("看看tools："+str(tools))
    # print("查看messages[-3]对象：" + str(messages[-3]))
    # print("查看messages[-2]对象：" + str(messages[-2]))
    # print("查看messages[-1]对象：" + str(messages[-1]))
    last_message = messages[-1]  # 获取最后一条消息

    panduan_result = panduan_net_neo4j.invoke(sanitize_state_for_gemini(state))

    print("panduan_result 要查询数据库 需要联网查询 判断里的输出：")

    # 获取 input_tokens, output_tokens, total_tokens
    input_tokens1 = panduan_result.usage_metadata.get("input_tokens", 0)
    output_tokens1 = panduan_result.usage_metadata.get("output_tokens", 0)
    total_tokens1 = panduan_result.usage_metadata.get("total_tokens", 0)

    global input_tokens, output_tokens, total_tokens  # 声明 tokens 相关变量为全局变量
    # input_tokens1, output_tokens1, total_tokens1 = returnTokens(panduan_result)
    input_tokens += input_tokens1
    output_tokens += output_tokens1
    total_tokens += total_tokens1
    # print(f"input_tokens: {input_tokens}")
    # print(f"output_tokens: {output_tokens}")
    print(f"total_tokens: {total_tokens}")

    # print("panduan_result看看大模型如何判断：",panduan_result.content)
    flag = panduan_result.content.split("#")[-1]

    if "要查询数据库" in flag:
        # print("下一个节点是 get_neo4j_schema_tool_call")
        return "get_neo4j_schema_tool_call"
    if "需要联网查询" in flag:
        # print("下一个节点是 net_query_call")
        return "net_query_call"
    else:
        # print("下一个节点是 answer_net_node_call")
        return "answer_net_node_call"


"""
决策节点（如first_tool_call）：生成包含工具调用描述（名称、参数、ID）的 AIMessage，但此时仅产生调用意图，尚未执行实际操作
执行节点（如list_tables_tool）：通过create_tool_node_with_fallback生成的节点，实际调用底层工具（如 SQL 数据库接口）完成操作

这种 "声明式调用→实际执行" 的二元结构是该框架的核心设计范式。
"""

# 指定节点之间的边
# workflow.add_edge(START, "get_neo4j_schema_tool_call")  # 从开始节点到第一个工具调用节点
workflow.add_conditional_edges(  # 有2分支：1.query_gen  2.answer_call
    START,
    should_net_neo4j_continue,  # 从查询生成节点到条件边
)

# 图谱回答
workflow.add_edge("get_neo4j_schema_tool_call", "get_neo4j_schema_tool")  # 从第一个工具调用节点到列表表工具节点
workflow.add_edge("get_neo4j_schema_tool", "query_gen")  # 从获取表结构工具节点到查询生成节点
workflow.add_edge("query_gen", "correct_query")  # 从语句生成节点到语句检查节点
workflow.add_edge("correct_query", "execute_query_tool")  # 从查询校验节点到执行查询节点
workflow.add_node("graph_assessment", graph_assessment)
workflow.add_edge("execute_query_tool", "graph_assessment")
workflow.add_conditional_edges(
    "graph_assessment",
    should_continue,  # 从查询生成节点到条件边
)
workflow.add_edge("answer_call", END)  # end估计就是SubmitFinalAnswer工具节点这个工具里是大模型对信息分析

# 普通回答
workflow.add_edge("answer_net_node_call", END)  # end估计就是SubmitFinalAnswer工具节点这个工具里是大模型对信息分析

# 联网查询
workflow.add_edge("net_query_call", "net_tool")  # 从第一个工具调用节点到列表表工具节点  net_should_continue
workflow.add_node("web_assessment", web_assessment)
workflow.add_edge("net_tool", "web_assessment")
workflow.add_conditional_edges(
    "web_assessment",
    net_should_continue,  # 从查询生成节点到条件边
)
# workflow.add_edge("net_tool", "answer_net_node_call")  # 从第一个工具调用节点到列表表工具节点

# 编译工作流以供运行
# graphApp = workflow.compile()  # 编译工作流


# 编译工作流以供运行
graphApp = workflow.compile()  # 编译工作流

print("编译完毕")


# In[ ]:


# In[15]:


# from IPython.display import Markdown
#
# # 直接生成并显示 Mermaid 代码（避免网络请求）
# mermaid_code = graphApp.get_graph().draw_mermaid()
# display(Markdown(f"```mermaid\n{mermaid_code}\n```"))


# In[16]:


# 这个event是state信息流里的一个元素，不是整个信息流
def returnTokens(event):
    input_tokens1 = 0
    output_tokens1 = 0
    total_tokens1 = 0

    for node_name, node_output in event.items():
        for msg in node_output.get("messages", []):
            if isinstance(msg, AIMessage):
                # 提取token使用信息
                response_metadata = msg.response_metadata
                if response_metadata and 'token_usage' in response_metadata:
                    token_data = response_metadata['token_usage']
                    # 更新全局变量
                    input_tokens1 = int(token_data.get('prompt_tokens', 0))
                    output_tokens1 = int(token_data.get('completion_tokens', 0))
                    total_tokens1 = int(token_data.get('total_tokens', 0))

    # 返回token使用信息元组
    return input_tokens1, output_tokens1, total_tokens1


# In[17]:


def returnAiAnswer(event):
    for node_name, node_output in event.items():  # 遍历事件中的每个节点
        if node_name in ["answer_call", "answer_net_node_call"]:  # 如果节点名称匹配
            for msg in node_output.get("messages", []):  # 遍历节点输出中的消息
                if isinstance(msg, AIMessage):  # 如果消息是AI生成的消息
                    final_answers = []  # 初始化最终答案列表
                    content = msg.content.strip()  # 获取消息内容并去除首尾空格

                    # 解析结构化回答
                    if msg.tool_calls:  # 如果消息中有工具调用
                        for tool_call in msg.tool_calls:  # 遍历每个工具调用
                            if tool_call['name'] == 'SubmitFinalAnswer':  # 如果工具名称匹配
                                final_answers.append(tool_call['args']['final_answer'])  # 添加最终答案
                    elif 'tool_calls' in msg.additional_kwargs:  # 如果消息的附加参数中有工具调用
                        for call in msg.additional_kwargs['tool_calls']:  # 遍历每个工具调用
                            if call['function']['name'] == 'SubmitFinalAnswer':  # 如果函数名称匹配
                                try:
                                    args = json.loads(call['function']['arguments'])  # 解析函数参数
                                    final_answers.append(args['final_answer'])  # 添加最终答案
                                except:
                                    pass  # 忽略解析错误

                    # 构建最终响应
                    if final_answers:  # 如果有最终答案
                        aiAnswer = "".join([
                            # f"{i + 1}. {ans}" for i, ans in enumerate(final_answers)  # 格式化最终答案
                            f"{ans}" for i, ans in enumerate(final_answers)  # 格式化最终答案
                        ])
                    elif content:  # 如果有直接回答内容
                        aiAnswer = f"{content}"  # 格式化直接回答
                    else:  # 抛出异常，提示未能获取有效回答
                        print("无有效回答")
                        raise ValueError("无有效回答")
                    break  # 获取第一个有效回答后退出循环

    return aiAnswer  # 返回AI回答


# In[ ]:


# In[8]:


# !pip install pandas
# !pip install openpyxl


# In[9]:

import pandas as pd
from openpyxl import load_workbook
import os
import acc
import json  # 新增导入 json 模块


def get_line_info(file_path, output_file_path, error_output_file_path, start_line=config.DEFAULT_START_LINE, end_line=config.DEFAULT_END_LINE):
    global stepNeo4j
    global stepNet
    global messages1
    global input_tokens, output_tokens, total_tokens

    # 初始化全局变量
    messages1 = []
    stepNeo4j = 0
    stepNet = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    # 定义中英文列名映射（行号放在 ID 前面）
    columns_mapping = {
        "LineNumber": "行号",  # 新增行号列
        "ID": "ID",
        "Left": "问题",
        "Accuracy": "准确率",
        "Right": "标准答案",
        "AiAnswer": "AI回答",
        "ElapsedTime": "运行时间",
        "AllStep": "总步骤",
        "StepNeo4j": "Neo4j步骤",
        "StepNet": "网络步骤",
        "InputTokens": "输入Tokens",
        "OutputTokens": "输出Tokens",
        "TotalTokens": "总Tokens"
    }

    error_columns_mapping = {
        "LineNumber": "行号",  # 新增行号列
        "ID": "ID",
        "Left": "问题",
        "Accuracy": "准确率",
        "Right": "标准答案",
        "AiAnswer": "AI回答",
        "ElapsedTime": "运行时间",
        "AllStep": "总步骤",
        "StepNeo4j": "Neo4j步骤",
        "StepNet": "网络步骤",
        "InputTokens": "输入Tokens",
        "OutputTokens": "输出Tokens",
        "TotalTokens": "总Tokens"
    }

    results = []
    error_results = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            # 重置全局变量
            messages1 = []
            stepNeo4j = 0
            stepNet = 0
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            if not hasattr(thread_local, 'num'):
                thread_local.num = 0
            thread_local.num = 0
            if not hasattr(thread_local, 'net_num'):
                thread_local.net_num = 0
            thread_local.net_num = 0

            accOne = 0

            # 行号范围过滤
            if line_number < start_line or line_number > end_line:
                continue

            if end_line == start_line:
                per = 100.0  # 只处理一行时，进度为100%
            else:
                per = (line_number - start_line) / (end_line - start_line) * 100
            per_rounded = round(per, 2)
            print("---------------------------------------------开启JSONL第" + str(
                line_number) + "行,进度" + str(per_rounded) + "%---------------------------------------------")

            # 解析 JSONL 行
            try:
                data = json.loads(line.strip())
                # 兼容 _id 和 id 字段
                current_id = data.get("_id") or data.get("id")
                if current_id is None:
                    current_id = f"missing_id_{line_number}"  # 获取 id
                question = data.get("question", "")  # 获取问题
                answer = data.get("answer", "")  # 获取标准答案
            except json.JSONDecodeError as e:
                print(f"第 {line_number} 行 JSON 解析失败: {e}")
                # 错误记录也包含行号
                error_results.append([
                    line_number,  # 行号
                    f"parse_error_{line_number}", "", 0, "", f"JSON解析错误: {e}",
                    0, 0, 0, 0, 0, 0, 0
                ])
                continue

            # 将 question 作为输入 left，answer 作为标准答案 right
            left = question
            right = answer

            try:
                import time
                start_time = time.time()

                print("input是", left)
                messages1.append(("human", left))
                event = ""
                assessment = {}
                for event in graphApp.stream(
                        {"messages": messages1, "name": "default_name"}
                ):
                    for update in event.values():
                        if isinstance(update, dict) and "assessment" in update:
                            assessment = update["assessment"]
                    input_tokens1, output_tokens1, total_tokens1 = returnTokens(event)
                    input_tokens += input_tokens1
                    output_tokens += output_tokens1
                    total_tokens += total_tokens1

                print("total_tokens:", total_tokens)
                end_time = time.time()
                elapsed_time = end_time - start_time

                allStep = stepNeo4j + stepNet

                AiAnswer = returnAiAnswer(event)
                print("标准答案是：" + right)
                print("output是：" + AiAnswer)
                accOne = acc.calculate_matching_rate(right, AiAnswer)
                print("acc是：" + str(accOne))

                # 将行号和 id 放在最前面
                stepList = [line_number, current_id, left, accOne, right, AiAnswer, elapsed_time,
                            allStep, stepNeo4j, stepNet, input_tokens, output_tokens, total_tokens]
                results.append(stepList)

                # 写入 Excel（每处理一条就追加）
                if results:
                    columns = list(columns_mapping.values())
                    if not os.path.exists(output_file_path):
                        header_df = pd.DataFrame(columns=columns)
                        header_df.to_excel(output_file_path, index=False)
                        print("Excel 文件已创建，列名已写入。")

                    df = pd.DataFrame(results, columns=list(columns_mapping.keys()))
                    df.rename(columns=columns_mapping, inplace=True)
                    for field in ("completed_steps", "total_steps", "completion", "stop_reason"):
                        df[field] = assessment.get(field)

                    with pd.ExcelWriter(output_file_path, mode='a', engine='openpyxl',
                                        if_sheet_exists='overlay') as writer:
                        sheet = writer.book['Sheet1']
                        for col, label in enumerate(df.columns, 1):
                            if sheet.cell(1, col).value is None:
                                sheet.cell(1, col, label)
                        startrow = sheet.max_row
                        df.to_excel(writer, index=False, sheet_name='Sheet1', startrow=startrow, header=False)

                    results = []

            except Exception as e:
                AiAnswer = "Error: " + str(e)
                elapsed_time = 0
                allStep = 0

                error_results.append([
                    line_number, current_id, left, accOne, right, AiAnswer, elapsed_time,
                    allStep, stepNeo4j, stepNet, input_tokens, output_tokens, total_tokens
                ])

                if error_results:
                    error_columns = list(error_columns_mapping.values())
                    if not os.path.exists(error_output_file_path):
                        header_df = pd.DataFrame(columns=error_columns)
                        header_df.to_excel(error_output_file_path, index=False)
                        print("错误信息 Excel 文件已创建，列名已写入。")

                    error_df = pd.DataFrame(error_results, columns=list(error_columns_mapping.keys()))
                    error_df.rename(columns=error_columns_mapping, inplace=True)

                    with pd.ExcelWriter(error_output_file_path, mode='a', engine='openpyxl',
                                        if_sheet_exists='overlay') as writer:
                        sheet = writer.book['Sheet1']
                        startrow = sheet.max_row
                        error_df.to_excel(writer, index=False, sheet_name='Sheet1', startrow=startrow, header=False)

                    error_results = []

                print("错误信息:", e)

        print("所有行执行结束-----------------------------------------------------")


# 运行参数已经统一放在文件开头的“实验输入输出配置”中。

if __name__ == "__main__":
    while True:
        try:
            if start_line == 0 and end_line == 0:
                print("退出程序")
                break
            if start_line <= 0 or end_line <= 0 or start_line > end_line:
                print("行号无效，请重新输入")
                continue
            # 调用处理函数
            get_line_info(file_path, output_file_path, error_output_file_path, start_line, end_line)
            start_input = input("请输入起始行号（输入0退出）: ")
            end_input = input("请输入结束行号: ")
            start_line = int(start_input)
            end_line = int(end_input)
        except ValueError:
            print("请输入有效的数字")

# # 历史消息
# messages1 = []
#
# input = "the films that share the directors with the film [Catch Me If You Can] were in which languages?"
# # "the films that share the directors with the film [Catch Me If You Can] were in which languages?"
# #联网查询广东海洋大学在那个城市，这个城市里的师范高校具体叫什么名字，这师范学校现任校长叫什么
# messages1.append(("human", input))
#
# event = ""
#
#
# for event in graphApp.stream(
#     {"messages": messages1}
# ):
#     print(event)
# messages1 = []
#
