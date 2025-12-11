from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import praw
from typing_extensions import TypedDict
from typing import Annotated
import operator
from langgraph.types import Send
from pydantic import BaseModel

class Ticker(BaseModel):
    ticker: str

class Tickers(BaseModel):
    tickers: list[Ticker]

class State(TypedDict):
    subreddit: str
    hot_posts: list[str]
    tickers: Annotated[Tickers, operator.add]

class WorkerState(TypedDict):
    post: str
    tickers: Annotated[Tickers, operator.add]


#Node 1: Fetch hottest posts from a subreddit
def get_hottest_posts_from_a_subreddit(state: State):
    """
    A tool that fetches the hottest posts from a subreddit.
    Args:
        subreddit ([str]): A subreddit name to fetch stock tickers from.
    Returns:
        list[str]: a list of the hottest posts from the subreddit.
    """
    reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="my_user_agent"
    )
    submissions = reddit.subreddit(state["subreddit"]).hot(limit=10)

    hot_posts = [title + "\n\n" + body for title, body in
                 [(submission.title, submission.selftext) for submission in submissions]]
    
    return {"hot_posts": hot_posts}

#Node 2: Extract stock tickers from a post, one node for each worker/post. 
def extract_tickers_from_post(state: WorkerState):
    """
    A tool that extracts stock tickers from a post.
    Args:
        post ([str]): A post to extract stock tickers from.
    Returns:
        list[str]: a list of stock tickers extracted from the post.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm = llm.with_structured_output(Tickers)
    tickers = structured_llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant that extracts stock tickers from social media posts."),
            HumanMessage(content=f"Extract all stock tickers from the following post:\n\n{state['post']}\n\n"),
        ]
    )
    return {"tickers": tickers}

#Edge Function: Assign a worker for each post to extract stock tickers.
def assign_workers(state: State):
    """
    A conditional edge function that creates a worker for each post to extract stock tickers.
    """
    return [Send("extract_tickers_from_post", {"post": post}) for post in state["hot_posts"]]


# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("get_hottest_posts_from_a_subreddit", get_hottest_posts_from_a_subreddit)
orchestrator_worker_builder.add_node("extract_tickers_from_post", extract_tickers_from_post)

# Add edges to connect nodes
orchestrator_worker_builder.add_edge(START, "get_hottest_posts_from_a_subreddit")
orchestrator_worker_builder.add_conditional_edges(
    "get_hottest_posts_from_a_subreddit", assign_workers, ["extract_tickers_from_post"]
)
orchestrator_worker_builder.add_edge("extract_tickers_from_post", END)

# Compile the workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# Show the workflow
png_bytes = orchestrator_worker.get_graph().draw_mermaid_png()

# Save to a file
with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)

print("Workflow graph saved as workflow_graph.png")