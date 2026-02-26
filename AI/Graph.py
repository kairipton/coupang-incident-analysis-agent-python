import os
import sys
import dotenv
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import Node



builder = StateGraph(Node.State)

# region 노드 정의
""" 그래프에 쓰일 노드들을 정의 함"""
builder.add_node( "question", Node.node_input_question )
builder.add_node( "multi_query", Node.node_multiquery_search, metadata={ "unity_label": "Mutli Querying..." } )
builder.add_node( "hybrid_search", Node.node_hybrid_search, metadata={ "unity_label": "Hybrid Searching..." } )
builder.add_node( "tool_call", Node.node_tool_call, metadata={ "unity_label": "Decision Tool Calling..." } )
builder.add_node( "route_next", Node.node_route_next, metadata={ "unity_label": "Decision Tool Calling..." } )
builder.add_node( "tools", Node.node_tools, metadata={ "unity_label": "Using Tools..." } )
builder.add_node( "final_answer", Node.node_final_answer, metadata={ "unity_label": "Generating Answer..." } )
builder.add_node( "summary", Node.node_summary_conversation, metadata={ "unity_label": "Summarying messages..." } )
builder.add_node( "evaluate", Node.node_evaluate, metadata={ "unity_label": "RAGAS Processing..." } )
builder.add_node( "graph_end", Node.node_graph_end, metadata={ "unity_label": "DONE!" } )
# endregion


# region 그래프 정의
""" 노드 연결 시작 """
builder.add_edge( START, "question" )
builder.add_edge( "question", "multi_query" )
builder.add_edge( "multi_query", "hybrid_search" )
builder.add_edge( "hybrid_search", "tool_call" )
builder.add_conditional_edges( 
    "tool_call", 
    Node.node_route_next,
    {
        "need_tools" : "tools",
        "NONE": "final_answer"
    }
)
builder.add_edge( "tools", "tool_call" )
builder.add_edge( "final_answer", "summary" )
builder.add_edge( "final_answer", "evaluate" )
builder.add_edge( "summary", "graph_end" )
builder.add_edge( "evaluate", "graph_end" )
builder.add_edge( "graph_end", END )
# endregion

memory = MemorySaver()
graph = builder.compile( checkpointer=memory )
print( graph.get_graph().draw_ascii() )

def run_test():
    config = { "configurable" : { "thread_id" : "test_uid" } }

    while True:
        question = input( "질문: ")
        result = graph.invoke({ 
            "messages" : [ 
                ( "user", question ) 
            ]
        }, config=config )

        # 0은 사용자의 입력, -1이 대체로 AI의 답변이지만, 도구사용을 마지막으로 했을 경우 메세지가 아닐 수 있다.
        #print( result["messages"][0] )
        #print( result["messages"][-1] )
        pprint( result )

        # summary = result.get( "summary", "" )
        # print( "\n[대화 요약]" )
        # print( summary )

def run_optimize():
    import optuna 

    def objective(trial: optuna.Trial):
        """ Optuna가 최적화를 목적으로 반복 호출하는 목적 함수."""

        questions = [
            "쿠팡 개인정보 유출 사고에서 최초 확인된 유출 계정 수와, 2026년 2월 5일에 추가로 유출이 신고된 계정 수는 각각 몇 개인가요?",
            # "쿠팡이 한국인터넷진흥원(KISA)에 침해사고를 처음 신고한 날짜와 시간은 언제이며, 과태료 부과 대상이 된 이유는 무엇인가요?",
            # "배송지 목록 수정 페이지 조회를 통해 유출된 개인정보 항목에는 어떤 것들이 포함되어 있나요?",
            # "과학기술정보통신부 조사 결과, 전직 개발자였던 공격자가 대규모 정보를 무단으로 유출할 수 있었던 시스템적 원인은 무엇인가요?",
            # "2026년 1월 14일, 개인정보보호위원회가 쿠팡 측에 자체조사 결과 홈페이지 공지를 즉각 중단하라고 촉구한 이유는 무엇인가요?",
            # "개인정보보호위원회가 쿠팡에게 기존의 통지문을 수정하라고 지시하면서, \"노출\"이라는 단어 대신 어떤 단어를 사용하라고 의결했나요?",
            # "개인정보보호위원회 조사 과정에서 쿠팡이 차후 제재 처분 시 '가중 요건'을 받을 수 있다고 엄중 경고를 받은 구체적인 행위는 무엇인가요?",
            # "과기정통부 조사 결과, 쿠팡이 '자료보전 명령 위반'으로 수사 의뢰를 받게 된 결정적인 이유는 무엇인가요?",
            # "공격자가 쿠팡에 보낸 이메일 원문에서 언급한 타 국가(일본, 대만) 피해 주장 내용과 실제 유출된 한국 사용자의 '주문 데이터' 규모는 메일 내용상 몇 개인가요?",
            # "과기정통부 조사단이 쿠팡에 요구한 '재발방지 대책'의 핵심 내용 3가지는 무엇인가요?"
        ]
        
        retriever_w = trial.suggest_float("retriever_w", 0.1, 0.9)
        retriever_k = trial.suggest_int( "retriever_k", 2, 15 )
        reranking_top_k = trial.suggest_int( "reranking_top_k", 2, 8 )

        i: int = 0
        all_scores = []
        for q in questions:
            config = { "configurable" : { "thread_id" : f"opt_{trial.number}_{i}" } }
            i += 1

            print( f"\n[질문] {q}" )
            result = graph.invoke({ 
                "messages" : [ 
                    ( "user", q ) 
                ],
                "opt_w" : retriever_w,
                "opt_k" : retriever_k,
                "opt_top_k" : reranking_top_k,
            }, config=config )

            scores = result.get("ragas", {})
            length = len(scores)
            score_sum: float = 0.0
            for k, _ in scores.items():
                score_sum += scores[k]

            #print( f"\n[답변] {answer}" )
            this_score = score_sum / length if length > 0 else 0.0
            all_scores.append( this_score )

            # 현재까지 누적 평균을 optuna로 보고
            curr_average = sum( all_scores ) / len( all_scores ) if len(all_scores) > 0 else 0.0
            trial.report( curr_average, step=i )

            if trial.should_prune():
                print( f"{i+1}회차 Trial은 성능이 낮으므로 중단" )
                raise optuna.exceptions.TrialPruned()

        
        return sum(all_scores) / len(all_scores) if len(all_scores) > 0 else 0.0
           
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction = "maximize",
        sampler = optuna.samplers.TPESampler(seed=42),
        study_name = "Coupang Agent Parameter Optimization",
        storage="sqlite:///optuna_results.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=4)
    )

    print( ("=" * 10) + "최적화 시작" + ("=" * 10) )
    study.optimize( objective, n_trials=10 )
    print( ("=" * 10) + "최적화 종료" + ("=" * 10) )
    print( "optuna 대쉬보드에서 결과를 확인 하셈!" )

if __name__ == "__main__":
    run_test()
    #run_optimize()



