from langchain import OpenAI, Chain, PromptTemplate
from langchain.chains import LLMChain
from serpapi import GoogleSearch

# Define the templates for each agent



planner_template = """
You are a planner. Given the following task, create a detailed plan.
Task: {task}
Plan:
"""

web_search_template = """
You are a web searcher. Given the following query, use the web to find the most relevant information.
Query: {query}
Results:
"""

executor_template = """
You are an executor. Given the following search results, generate input arguments for a Python script.
Search Results: {results}
Python Input Arguments:
"""

analyzer_template = """
You are an analyzer. Given the following data, analyze the trends and provide insights.
Data: {data}
Analysis:
"""

# Define each agent using LangChain's LLMChain
class Planner:
    def __init__(self, llm):
        self.llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(planner_template))

    def create_plan(self, task):
        return self.llm_chain.run(task=task)


class WebSearcher:
    def __init__(self, llm, serp_api_key):
        self.llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(web_search_template))
        self.api_key = serp_api_key

    def search(self, query):
        search_query = self.llm_chain.run(query=query)
        search = GoogleSearch({"q": search_query, "api_key": self.api_key})
        results = search.get_dict()['organic_results']
        return results


class Executor:
    def __init__(self, llm):
        self.llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(executor_template))

    def execute(self, search_results):
        return self.llm_chain.run(results=search_results)


class Analyzer:
    def __init__(self, llm):
        self.llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(analyzer_template))

    def analyze(self, data):
        analysis = self.llm_chain.run(data=data)
        print(analysis)
        return analysis


# Supervisor oversees all the processes
class Supervisor:
    def __init__(self, llm, serp_api_key):
        self.planner = Planner(llm)
        self.web_searcher = WebSearcher(llm, serp_api_key)
        self.executor = Executor(llm)
        self.analyzer = Analyzer(llm)

    def supervise(self, task):
        plan = self.planner.create_plan(task)
        search_results = self.web_searcher.search(plan)
        execution_results = self.executor.execute(search_results)
        analysis = self.analyzer.analyze(execution_results)
        return analysis


# Initialize the LLM (e.g., OpenAI's GPT)
llm = OpenAI(api_key="your_openai_api_key_here")

# Create the Supervisor
serp_api_key = "your_serpapi_key_here"
supervisor = Supervisor(llm, serp_api_key)

# Example usage
if __name__ == "__main__":
    task = "Optimize nitrogen reduction reaction catalysts"
    result = supervisor.supervise(task)
    print(result)
