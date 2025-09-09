import streamlit as st
import yfinance as yf

def show_investors():
    st.title("📊 Institutional & Major Investors")

    # Get user input for stock ticker
    stock_ticker = st.text_input("Enter the Stock Ticker:", "AAPL")

    if stock_ticker:
        try:
            stock = yf.Ticker(stock_ticker)

            # Fetching major shareholders
            st.subheader("🏦 Major Shareholders")
            major_holders = stock.major_holders
            if major_holders is not None:
                st.write(major_holders)
            else:
                st.write("No data available for major shareholders.")

            # Fetching institutional holders
            st.subheader("🏢 Institutional Holders")
            institutional_holders = stock.institutional_holders
            if institutional_holders is not None:
                st.write(institutional_holders)
            else:
                st.write("No data available for institutional holders.")

            # Fetching mutual fund holders
            st.subheader("💰 Mutual Fund Holders")
            mutual_fund_holders = stock.mutualfund_holders
            if mutual_fund_holders is not None:
                st.write(mutual_fund_holders)
            else:
                st.write("No data available for mutual fund holders.")

        except Exception as e:
            st.error(f"⚠️ Error fetching data: {e}")

# Run the function if the script is executed directly
if __name__ == "__main__":
    show_investors()

