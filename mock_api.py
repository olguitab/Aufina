import os
from unittest.mock import patch, MagicMock
from intelligence import IntelligenceLayer, AnalystOutput, StrategistOutput

# This script mocks the LLM responses so the MVP runs without a real OpenAI API key

def run_mock_mvp():
    os.environ['OPENAI_API_KEY'] = 'mock_key_for_testing'
    
    # We patch the ChatOpenAI.with_structured_output locally
    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        
        # Setup mock behavior for Analyst
        mock_analyst_llm = MagicMock()
        mock_analyst_out = AnalystOutput(
            sentiment=-0.8,
            affected_assets=["AAPL"],
            confidence=0.9
        )
        # Mocking the runnable chain's invoke method
        mock_chain_analyst = MagicMock()
        mock_chain_analyst.invoke.return_value = mock_analyst_out
        
        # Setup mock behavior for Strategist
        mock_strategist_llm = MagicMock()
        mock_strategist_out = StrategistOutput(
            signal="SELL",
            strategy_used="Trend Following (Negative Sent)",
            target_asset="AAPL",
            reasoning="Sentiment is highly negative (-0.8) indicating severe supply chain issues. Recommending SELL."
        )
        mock_chain_strategist = MagicMock()
        mock_chain_strategist.invoke.return_value = mock_strategist_out

        # Assign behaviors to IntelligenceLayer
        def patch_build_graph(self):
            # Same graph logic, but we inject our mock chains
            self.analyst_chain = mock_chain_analyst
            self.strategist_chain = mock_chain_strategist
            
            # Re-define nodes to use our mock chains instead of self.analyst_llm
            def _analyst_node(state):
                out = self.analyst_chain.invoke({"news_text": state["news_text"]})
                return {"analyst_output": out, "revision_count": state.get("revision_count", 0)}
                
            def _strategist_node(state):
                out = self.strategist_chain.invoke({"analyst_data": "mock"})
                return {"strategist_output": out}
                
            # Replace the methods
            self._analyst_node = _analyst_node
            self._strategist_node = _strategist_node
            
            from langgraph.graph import StateGraph, END
            from intelligence import AgentState
            workflow = StateGraph(AgentState)
            workflow.add_node("analyst", self._analyst_node)
            workflow.add_node("strategist", self._strategist_node)
            workflow.add_node("risk", self._risk_node)
            workflow.set_entry_point("analyst")
            workflow.add_edge("analyst", "strategist")
            workflow.add_edge("strategist", "risk")
            workflow.add_conditional_edges("risk", self._should_execute)
            return workflow.compile()
            
        with patch.object(IntelligenceLayer, '_build_graph', patch_build_graph):
            from main import main
            main()

if __name__ == '__main__':
    run_mock_mvp()
