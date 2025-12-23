import streamlit as st
import streamlit.components.v1 as components
import random
import time
import textwrap

def calculate_ai_score(ticker):
    """
    Simulates a complex multi-model AI inference.
    Returns a dictionary with the score and components.
    """
    # Simulate processing time
    time.sleep(1.5)
    
    # Mock Logic based on Ticker Name hash to keep it consistent but random-looking
    seed = sum(ord(c) for c in ticker)
    random.seed(seed)
    
    # Components (0-100)
    technicals_score = random.randint(40, 95)
    sentiment_score = random.randint(30, 90)
    fundamentals_score = random.randint(50, 98)
    
    # Weights: Tech(40%), Sent(30%), Fund(30%)
    total_score = (technicals_score * 0.4) + (sentiment_score * 0.3) + (fundamentals_score * 0.3)
    total_score = int(total_score)
    
    # Reasoning Generation
    reasons = [
        "Bullish SMA crossover detected on weekly charts.",
        "Positive sentiment spike on Twitter regarding recent earnings.",
        "RSI indicates the stock is slightly overbought but momentum is strong.",
        "Quarterly results exceeded analyst expectations by 15%.",
        "Institutional accumulation observed in the last trading session.",
        "Sector rotation favoring this industry vertical."
    ]
    selected_reasons = random.sample(reasons, k=2)
    
    return {
        "score": total_score,
        "technicals": technicals_score,
        "sentiment": sentiment_score,
        "fundamentals": fundamentals_score,
        "reasoning": selected_reasons
    }

def show_ai_insights():
    st.title("🧠 StockVision™ AI Insights")
    st.markdown("### The Unified AI Score")
    st.caption("Our proprietary engine combines Technicals, Social Sentiment, and Fundamentals into a single decision metric.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE)", "RELIANCE").upper()
        analyze_btn = st.button("Generate AI Score ✨", use_container_width=True)
    
    if analyze_btn:
        with st.spinner(f"Running Multi-Model Inference on {ticker}..."):
            data = calculate_ai_score(ticker)
        
        score = data["score"]
        
        # Color coding
        if score >= 80:
            verdict = "STRONG BUY 🚀"
            color = "#16A34A" # Green
            bg_color = "#DCFCE7"
        elif score >= 60:
            verdict = "BUY 🟢"
            color = "#16A34A"
            bg_color = "#F0FDF4"
        elif score >= 40:
            verdict = "HOLD 🟡"
            color = "#CA8A04" # Yellow
            bg_color = "#FEF9C3"
        else:
            verdict = "SELL 🔴"
            color = "#DC2626" # Red
            bg_color = "#FEE2E2"

        # --- The Super Card ---
        html_content = textwrap.dedent(f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
                body {{
                    font-family: 'Roboto', sans-serif;
                    margin: 0;
                    padding: 5px;
                }}
            </style>
            <div style="background-color: white; padding: 30px; border-radius: 15px; border: 1px solid #E2E8F0; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin:0; color: #1E293B;">StockVision Score</h2>
                        <p style="color: #64748B; margin-top: 0;">Confidence Level matching {ticker}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 3.5rem; font-weight: 800; color: {color};">{score}/100</span>
                    </div>
                </div>
                
                <div style="background-color: {bg_color}; border-left: 5px solid {color}; padding: 15px; margin: 20px 0; border-radius: 4px;">
                    <h3 style="margin: 0 0 10px 0; color: {color}; font-size: 1.5rem;">{verdict}</h3>
                    <p style="margin: 0; color: #334155; font-weight: 500;">AI Analysis:</p>
                    <ul style="margin: 5px 0 0 20px; color: #475569;">
                        <li>{data['reasoning'][0]}</li>
                        <li>{data['reasoning'][1]}</li>
                    </ul>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 30px; text-align: center;">
                    <div style="background-color: #F8FAFC; padding: 15px; border-radius: 10px;">
                        <div style="font-size: 0.9rem; color: #64748B; margin-bottom: 5px;">Technicals</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">{data['technicals']}/100</div>
                        <div style="font-size: 0.8rem; color: #22C55E;">Trend</div>
                    </div>
                    <div style="background-color: #F8FAFC; padding: 15px; border-radius: 10px;">
                        <div style="font-size: 0.9rem; color: #64748B; margin-bottom: 5px;">Sentiment</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">{data['sentiment']}/100</div>
                        <div style="font-size: 0.8rem; color: #3B82F6;">Social/News</div>
                    </div>
                    <div style="background-color: #F8FAFC; padding: 15px; border-radius: 10px;">
                        <div style="font-size: 0.9rem; color: #64748B; margin-bottom: 5px;">Fundamentals</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A;">{data['fundamentals']}/100</div>
                        <div style="font-size: 0.8rem; color: #A855F7;">Valuation</div>
                    </div>
                </div>
            </div>
        """)
        components.html(html_content, height=600, scrolling=True)
        
        st.markdown("---")
        st.info("Disclaimer: This score is generated by an AI model for educational purposes. Do not trade solely based on this output.")

if __name__ == "__main__":
    show_ai_insights()
