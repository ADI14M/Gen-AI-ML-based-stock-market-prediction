import streamlit as st
import pandas as pd
import random

def show_dashboard():
    # --- Custom CSS for "Premium" Look ---
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Compact Title */
        h1 {
            font-size: 2.2rem !important;
            font-weight: 700;
            color: #F8FAFC !important; /* Light text for dark mode */
            margin-bottom: 0.5rem !important;
        }
        
        /* Subheaders */
        h3 {
            font-size: 1.1rem !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8 !important; /* Lighter slate */
            margin-top: 1.5rem !important;
            margin-bottom: 0.8rem !important;
            border-bottom: 2px solid #334155; /* Darker border */
            padding-bottom: 5px;
        }

        /* Custom Class for 'Welcome' text */
        .sub-header-text { 
            font-size: 1.0rem !important; 
            color: #CBD5E1; /* Light grey */
            font-weight: 400; 
        }

        /* Styling Metrics to be compact and premium */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 600;
            color: #F1F5F9 !important; /* Almost white */
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
            color: #94A3B8 !important;
            font-weight: 500;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.8rem !important;
        }

        /* Card-like containers for data */
        .css-1r6slb0, .stDataFrame, .stTable { 
            border-radius: 8px;
            border: 1px solid #334155; /* Dark border */
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.2);
            background-color: #1E293B; /* Dark card background */
        }
        
        /* Table Headers */
        .dataframe th {
            font-size: 0.85rem !important;
            background-color: #0F172A !important; /* Very dark header */
            color: #E2E8F0 !important;
        }
        .dataframe td {
            font-size: 0.9rem !important;
            color: #CBD5E1 !important;
            border-bottom: 1px solid #334155 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # --- Header Navigation / Welcome ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Welcome back, Aditya! 👋")
        st.markdown("<p class='sub-header-text'>Your Financial Pulse at a Glance</p>", unsafe_allow_html=True)
    with col2:
        st.image("https://ui-avatars.com/api/?name=Aditya+Kumar&background=0D8ABC&color=fff&size=128", width=80)
        st.caption("Premium Member 🌟")

    st.markdown("---")

    # --- Account Details Section ---
    st.subheader("🏦 Account Overview")
    
    # Mock Account Data
    account_info = {
        "Broker": "Zerodha (Kite)",
        "Client ID": "AD123456",
        "Depository": "CDSL",
        "DP ID": "1208160012345678",
        "Status": "Active ✅"
    }

    # Display Account Info in a nice row
    cols = st.columns(len(account_info))
    for idx, (label, value) in enumerate(account_info.items()):
        with cols[idx]:
            st.metric(label=label, value=value)

    st.markdown("---")

    # --- Current Holdings Logic (Moved up for calculations) ---
    # Mock Holdings Data
    holdings_data = {
        "Instrument": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "TATAMOTORS", "ITC", "SBIN"],
        "Qty": [50, 20, 100, 150, 200, 300, 500, 400],
        "Avg. Price": [2450.00, 3200.00, 1450.00, 1500.00, 950.00, 600.00, 400.00, 550.00],
        "LTP": [2890.50, 3450.25, 1620.00, 1480.00, 1020.00, 850.00, 445.00, 620.00],
        "Changes %": ["+1.5%", "+0.8%", "+1.2%", "-0.5%", "+2.1%", "+4.5%", "+0.2%", "+1.1%"]
    }
    
    df_holdings = pd.DataFrame(holdings_data)
    
    # Calculate Current Value and P/L for display
    df_holdings["Invested Val"] = df_holdings["Qty"] * df_holdings["Avg. Price"]
    df_holdings["Current Val"] = df_holdings["Qty"] * df_holdings["LTP"]
    df_holdings["P&L"] = df_holdings["Current Val"] - df_holdings["Invested Val"]
    
    # Calculate Total Portfolio Metrics
    total_investment = df_holdings["Invested Val"].sum()
    total_current_value = df_holdings["Current Val"].sum()
    total_pnl = df_holdings["P&L"].sum()
    total_pnl_percentage = (total_pnl / total_investment) * 100
    
    # --- Portfolio Performance Section ---
    st.subheader("🚀 Portfolio Performance")
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Investment", f"₹ {total_investment:,.0f}", "Invested")
    with m2:
        st.metric("Current Value", f"₹ {total_current_value:,.0f}", f"₹ {total_pnl:,.0f} ({total_pnl_percentage:.2f}%)")
    with m3:
        st.metric("Day's Return", "₹ 15,400", "+ 1.2% 🟢")
    with m4:
        st.metric("Day's Realized P&L", "₹ 2,500", "Profit")

    # --- Current Holdings Display ---
    st.markdown("### 📊 My Holdings")
    
    # Style the dataframe (Highlight Positive P&L in Green, Negative in Red)
    def color_pnl(val):
        color = '#4ADE80' if val >= 0 else '#F87171' # Brighter green/red
        return f'color: {color}'

    # Adjust Index to start from 1
    df_display = df_holdings.copy()
    df_display.index = df_display.index + 1

    st.dataframe(
        df_display.style.applymap(color_pnl, subset=['P&L']).format({
            "Avg. Price": "₹ {:.2f}",
            "LTP": "₹ {:.2f}",
            "Current Val": "₹ {:.2f}",
            "P&L": "₹ {:.2f}"
        }),
        use_container_width=True,
        height=300
    )

    st.markdown("---")

    # --- IPO Watchlist ---
    st.subheader("📅 IPO Watchlist")
    
    col_ipo1, col_ipo2 = st.columns([2, 1])
    
    with col_ipo1:
        st.info("Upcoming & Active IPOs")
        ipo_data = {
            "IPO Name": ["Tata Technologies", "Ola Electric", "Swiggy", "FirstCry"],
            "Status": ["Open", "Upcoming", "Filed DRHP", "Upcoming"],
            "Price Band": ["₹475 - ₹500", "TBA", "TBA", "TBA"],
            "Close Date": ["24 Dec", "Jan '25", "Feb '25", "Mar '25"]
        }
        st.table(pd.DataFrame(ipo_data))
        
    with col_ipo2:
        st.success("🎉 Allotment Status")
        st.write("**Ideaforge Tech** - Allotted (1 Lot)")
        st.write("**Netweb Tech** - Not Allotted")
        st.write("**Utkarsh Small Fin** - Allotted (2 Lots)")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by Gen AI & ML • Market Data Delayed by 15 mins")

if __name__ == "__main__":
    show_dashboard()
