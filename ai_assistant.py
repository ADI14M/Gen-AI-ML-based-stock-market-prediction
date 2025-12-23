import streamlit as st
import time
import random

def get_bot_response(user_input):
    """
    Simulates a Gen-AI response based on keywords.
    """
    user_input = user_input.lower()
    
    if "hello" in user_input or "hi" in user_input:
        return "Hello! I am your StockVision Market Assistant. How can I help you optimize your portfolio today?"
    
    elif "portfolio" in user_input:
        return """
        Based on your current holdings:
        1. **Top Performer**: TATAMOTORS (+4.5% today).
        2. **Underperformer**: HDFCBANK (-0.5%).
        3. **Risk Alert**: Your portfolio is heavy on Banking stocks (HDFC, ICICI, SBI). Consider diversifying into IT or Pharma to balance the risk.
        """
        
    elif "news" in user_input or "latest" in user_input:
        return """
        **Latest Market Headlines:**
        *   **Sensex** jumps 400 points as inflation data cools down.
        *   **Tata Power** signs 500MW renewable energy deal.
        *   **IT Sector** sees a correction amid global recession fears.
        *   **Oil Prices** stabilize at $75/barrel.
        """
        
    elif "prediction" in user_input or "forecast" in user_input:
        return "My LSTM models predict a **Bullish** trend for the Nifty 50 over the next week, provided global cues remain positive. Key resistance is at 22,500."
    
    elif "thank" in user_input:
        return "You're welcome! Happy Investing. 🚀"
        
    else:
        return "I can analyze stock trends, summarize news, or review your portfolio health. Try asking: 'How is my portfolio?' or 'Latest market news'."

def show_ai_assistant():
    st.title("🤖 AI Market Assistant")
    st.caption("Your 24/7 Personal Financial Analyst. Ask me anything!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm StockVision AI. Ask me about your portfolio, specific stocks, or market trends."}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about stocks, portfolio, or news..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = get_bot_response(prompt)
            
            # Simulate stream of thought
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    show_ai_assistant()
