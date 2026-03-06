import os
from dotenv import load_dotenv

from ingestion import NewsEngine
from market_data import MarketData
from execution import PortfolioManager
from intelligence import IntelligenceLayer

def main():
    load_dotenv()
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not set. The LLM nodes will fail.")
        print("Please set it in your environment or a .env file.")

    print("=== Wealth Management Agent MVP Starting ===")
    
    news_engine = NewsEngine()
    portfolio_manager = PortfolioManager(initial_balance=100000.0)
    intelligence = IntelligenceLayer()
    
    print("\n[1] Fetching News...")
    latest_news = news_engine.fetch_latest_news()
    print(f"News Alert: {latest_news}")
    
    print("\n[2] Agentic Analysis (LangGraph)...")
    try:
        result_state = intelligence.run(latest_news)
        analyst_out = result_state["analyst_output"]
        strategist_out = result_state["strategist_output"]
        
        print("\n--- Intelligence Output ---")
        print(f"Analyst: Sentiment={analyst_out.sentiment}, Assets={analyst_out.affected_assets}, Conf={analyst_out.confidence}")
        print(f"Risk Approved: {result_state['risk_approved']} (Revisions: {result_state['revision_count']})")
        print(f"Strategist Signal: {strategist_out.signal} for {strategist_out.target_asset}")
        print(f"Reasoning: {strategist_out.reasoning}")
        
        target = strategist_out.target_asset
        if not target or target == "None":
            # If no target asset is identified, we don't proceed
            target = analyst_out.affected_assets[0] if analyst_out.affected_assets else "N/A"

        if target != "N/A":
            print(f"\n[3] Fetching Market Data for {target}...")
            current_price = MarketData.get_current_price(target)
            print(f"Current Price of {target}: ${current_price:.2f}")
            
            print("\n[4] Execution Simulator...")
            portfolio_manager.execute_order(
                ticker=target,
                signal=strategist_out.signal,
                price=current_price,
                reasoning=strategist_out.reasoning
            )
        else:
            print("\n[3, 4] Skipped: No target asset identified.")
            
    except Exception as e:
        print(f"\nAgentic Process Failed: {e}")
    
    print("=== MVP Run Complete ===")

if __name__ == "__main__":
    main()
