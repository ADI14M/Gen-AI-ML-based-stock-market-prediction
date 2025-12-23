import streamlit as st
import pandas as pd
import random

def get_ipo_data(category="Mainboard"):
    """
    Returns high-fidelity mock data mimicking Investorgain's GMP table.
    Values are hardcoded based on recent market trends (Dec 2025).
    """
    if category == "Mainboard":
        data = [
            {"IPO Name": "Tata Technologies", "Status": "Open", "Price": "₹500", "GMP": 380, "Kostak": 500, "Fire Rating": "⭐⭐⭐⭐⭐"},
            {"IPO Name": "Swiggy Limited", "Status": "Upcoming", "Price": "₹420", "GMP": 45, "Kostak": 0, "Fire Rating": "⭐⭐⭐"},
            {"IPO Name": "Ola Electric", "Status": "Upcoming", "Price": "₹76", "GMP": 8, "Kostak": 0, "Fire Rating": "⭐⭐"},
            {"IPO Name": "FirstCry (Brainbees)", "Status": "DRHP Filed", "Price": "TBA", "GMP": 0, "Kostak": 0, "Fire Rating": "NA"},
            {"IPO Name": "Innova Captab", "Status": "Closed", "Price": "₹448", "GMP": 105, "Kostak": 200, "Fire Rating": "⭐⭐⭐⭐"},
            {"IPO Name": "Azad Engineering", "Status": "Closed", "Price": "₹524", "GMP": 300, "Kostak": 0, "Fire Rating": "⭐⭐⭐⭐⭐"},
        ]
    else: # SME
        data = [
            {"IPO Name": "Kay Cee Energy & Infra", "Status": "Open", "Price": "₹54", "GMP": 60, "Kostak": 0, "Fire Rating": "⭐⭐⭐⭐⭐"},
            {"IPO Name": "AIK Pipes And Polymers", "Status": "Open", "Price": "₹89", "GMP": 12, "Kostak": 0, "Fire Rating": "⭐⭐"},
            {"IPO Name": "Akanksha Power", "Status": "Closed", "Price": "₹55", "GMP": 15, "Kostak": 0, "Fire Rating": "⭐⭐⭐"},
            {"IPO Name": "HRH Next Services", "Status": "Closed", "Price": "₹36", "GMP": 5, "Kostak": 0, "Fire Rating": "⭐"},
            {"IPO Name": "Balaji Valve Components", "Status": "Upcoming", "Price": "₹100", "GMP": 35, "Kostak": 0, "Fire Rating": "⭐⭐⭐⭐"},
            {"IPO Name": "Sameera Agro", "Status": "Listed", "Price": "₹180", "GMP": 0, "Kostak": 0, "Fire Rating": "NA"},
        ]
    
    df = pd.DataFrame(data)
    
    # Process numeric columns for calculation
    # Extract numeric part from Price string (e.g., "₹500" -> 500)
    df['Price Num'] = df['Price'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
    
    # Calculate Estimated Listing Price
    df['Est Listing'] = df['Price Num'] + df['GMP']
    
    # Calculate GMP Percentage
    df['GMP(%)'] = (df['GMP'] / df['Price Num']) * 100
    df['GMP(%)'] = df['GMP(%)'].fillna(0).round(2)
    
    return df

def show_ipo_dashboard():
    # --- Custom CSS for Table Layout ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        
        /* Table Styling matching Investorgain/Moneycontrol style */
        .stDataFrame table {
            border-collapse: collapse !important;
            width: 100%;
        }
        .stDataFrame th {
            background-color: #F8FAFC !important;
            color: #1E293B !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            text-align: left !important;
            padding: 10px !important;
        }
        .stDataFrame td {
            font-size: 0.9rem !important;
            color: #334155 !important;
            padding: 10px !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }
        
        /* Fire Rating Color */
        .fire-rating {
            color: #F59E0B;
        }
        
        /* Status Badges - Keep specific colors as they are semantic */
        .status-open { color: #15803D; font-weight: 600; }
        .status-upcoming { color: #1D4ED8; font-weight: 600; }
        .status-closed { color: #64748B; font-weight: 500; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚀 IPO GMP Dashboard")
    st.caption("Live Grey Market Premium and Subscription Status (Simulated Data)")

    # --- Metrics Row ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Hot IPO (Mainboard)", "Tata Technologies", "+76% GMP")
    c2.metric("Hot IPO (SME)", "Kay Cee Energy", "+111% GMP")
    c3.metric("Market Sentiment", "Bullish 🐂", "High Activity")

    st.markdown("---")

    # --- Controls ---
    col_filter, col_spacer = st.columns([1, 4])
    with col_filter:
        category = st.radio("Select Category:", ["Mainboard", "SME"], horizontal=True)

    # --- Data Fetching ---
    df = get_ipo_data(category)
    
    # --- Data Presentation Logic ---
    # We want to format the dataframe nicely before showing it.
    
    # 1. Format 'Est Listing' to include the price and percentage
    # Display Format: "₹880 (76.00%)"
    df["Est Listing"] = df.apply(lambda x: f"₹{x['Est Listing']:.0f} ({x['GMP(%)']}%)", axis=1)
    
    # --- Styling Function ---
    def color_gmp(val):
        if val > 0:
            return 'color: #16A34A; font-weight: bold;' # Green
        elif val < 0:
            return 'color: #DC2626; font-weight: bold;' # Red
        return 'color: #64748B;' # Grey
    
    # --- Content ---
    
    # 1. LIVE IPOs
    st.subheader("🟢 Live IPOs (Open Now)")
    df_live = df[df['Status'] == 'Open'].copy()
    if not df_live.empty:
        # Fix Index to start from 1
        df_live.reset_index(drop=True, inplace=True)
        df_live.index = df_live.index + 1
        
        st.dataframe(
            df_live.style.applymap(color_gmp, subset=['GMP']).format({"GMP": "₹ {}"}),
            use_container_width=True,
            column_config={
                "IPO Name": st.column_config.TextColumn("IPO Name", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Price": st.column_config.TextColumn("Price Band", width="small"),
                "GMP": st.column_config.NumberColumn("GMP (₹)", format="₹%d"),
                "Est Listing": st.column_config.TextColumn("Est Listing & Gain", width="medium"),
                "Fire Rating": st.column_config.TextColumn("Fire Rating", width="small"),
            }
        )
    else:
        st.info("No Live IPOs currently open for subscription.")

    st.markdown("---")

    # 2. UPCOMING IPOs
    st.subheader("📅 Upcoming IPOs")
    df_upcoming = df[df['Status'].isin(['Upcoming', 'DRHP Filed'])].copy()
    if not df_upcoming.empty:
        # Fix Index to start from 1
        df_upcoming.reset_index(drop=True, inplace=True)
        df_upcoming.index = df_upcoming.index + 1
        
        st.dataframe(
            df_upcoming.style.applymap(color_gmp, subset=['GMP']).format({"GMP": "₹ {}"}),
            use_container_width=True,
            column_config={
                "IPO Name": st.column_config.TextColumn("IPO Name", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Price": st.column_config.TextColumn("Price Band", width="small"),
                "GMP": st.column_config.NumberColumn("GMP (₹)", format="₹%d"),
                "Est Listing": st.column_config.TextColumn("Est Listing & Gain", width="medium"),
                "Fire Rating": st.column_config.TextColumn("Fire Rating", width="small"),
            }
        )
    else:
        st.info("No Upcoming IPOs announced yet.")
        
    st.markdown("---")

    # 3. CLOSED IPOs
    st.subheader("🔒 Closed IPOs")
    df_closed = df[df['Status'].isin(['Closed', 'Listed'])].copy()
    if not df_closed.empty:
        # Fix Index to start from 1
        df_closed.reset_index(drop=True, inplace=True)
        df_closed.index = df_closed.index + 1
        
        st.dataframe(
            df_closed.style.applymap(color_gmp, subset=['GMP']).format({"GMP": "₹ {}"}),
            use_container_width=True,
            column_config={
                "IPO Name": st.column_config.TextColumn("IPO Name", width="medium"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Price": st.column_config.TextColumn("Price Band", width="small"),
                "GMP": st.column_config.NumberColumn("GMP (₹)", format="₹%d"),
                "Est Listing": st.column_config.TextColumn("Est Listing & Gain", width="medium"),
                "Fire Rating": st.column_config.TextColumn("Fire Rating", width="small"),
            }
        )
    else:
        st.info("No recently closed IPOs.")
    
    st.caption("* GMP (Grey Market Premium) is unofficial and subject to high volatility.")

if __name__ == "__main__":
    show_ipo_dashboard()
