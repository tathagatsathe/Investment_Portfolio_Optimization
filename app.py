import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import expected_returns, risk_models
from pypfopt.black_litterman import BlackLittermanModel

def black_litterman(df):
    n_assets = len(df['Adj Close'].columns)
    mu = expected_returns.mean_historical_return(df['Adj Close'])
    S = risk_models.sample_cov(df['Adj Close'])

    Q = np.random.uniform(low=0, high=1, size=(n_assets,)).reshape(-1, 1)

    # Specify the link matrix
    P = np.eye(n_assets)

    # Specify the uncertainty matrix
    omega = np.diag(np.random.uniform(low=0, high=1, size=(n_assets,)))

    # Compute Black-Litterman expected returns
    bl = BlackLittermanModel(S, pi=mu,Q=Q, P=P, omega=omega)
    # bl.set_views(viewdict, confidences)
    bl_returns = bl.bl_returns()

    # Optimize the portfolio
    ef = EfficientFrontier(bl_returns, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    return pd.DataFrame(cleaned_weights,index=[0])

def mCAR(df):
    S = df['Adj Close'].cov()
    mu = mean_historical_return(df['Adj Close'])
    ef_cvar = EfficientCVaR(mu, S)
    cvar_weights = ef_cvar.min_cvar()
    cleaned_weights = ef_cvar.clean_weights()
    return pd.DataFrame(cleaned_weights, index=[0])

def HRP(df):
    daily_return_df = df['Adj Close'].pct_change().dropna()
    hrp = HRPOpt(daily_return_df)
    hrp_weights = hrp.optimize()

    return pd.DataFrame(hrp_weights,index=[0])

def mean_variance_optimization(df):
    closing_price_df = df['Adj Close']
    mu = mean_historical_return(closing_price_df)
    S = CovarianceShrinkage(closing_price_df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    return pd.DataFrame(ef.clean_weights(),index=[0])

def get_portfolio(df):
    daily_return_df = df['Adj Close'].pct_change().dropna()
    individual_rets = df['Adj Close'].resample('Y').last().pct_change().mean()

    returns = []
    volatility = []
    portfolio_weights = []

    variance_matrix = daily_return_df.cov()*252
    num_assets = len(daily_return_df.columns)
    num_portfolios = 10000

    for port in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        portfolio_weights.append(weights)

        returns.append(np.dot(weights, individual_rets))
        var = variance_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)

        ann_sd = sd*np.sqrt(250)
        volatility.append(ann_sd)

    data = {'Returns': returns, 'Volatility':volatility}

    for counter, symbol in enumerate(daily_return_df.columns.tolist()):
        data[symbol] = [w[counter] for w in portfolio_weights]

    portfolios = pd.DataFrame(data)
    return portfolios

st.set_page_config(page_title="Portfolio Optimization", page_icon=":chart_with_upwards_trend:",layout="wide")

st.title('Portfolio Optimization: Comparing Weighting Methods')

left_column, right_column = st.columns(2)
col1,col2,_,_ = st.columns(4)

options = ["AAPL", "MSFT", "AMZN", "GOOG","IBM", "META", "TSLA", "JPM", "NVDA"]
with left_column:
    st.header("Stock Chart Dashboard")
    selected_options = st.multiselect("Select stocks", options,key="ticker_key")

    start_date = st.date_input("Start date", value=date.today() - relativedelta(years=5), key="start_date")
    end_date = st.date_input("End date", value=date.today(),max_value=date.today(), key="end_date")
    
    if start_date > end_date:
        st.error("Error: End date must be after start date.")

# @st.cache(suppress_st_warning=True, allow_output_mutation=True)
def plot_stock_prices():
    if(len(selected_options)>0):
        traces = []
        for ticker in selected_options:
            df = yf.download(ticker,start=start_date,end=end_date)
            trace = go.Scatter(x=df.index, y=df['Close'], mode='lines',name=ticker)
            traces.append(trace)

        fig = go.Figure(data=traces)
        st.plotly_chart(fig)

if 'last_selected_options' not in st.session_state:
    st.session_state['last_selected_options'] = ''

if len(selected_options)>=2 and st.session_state.last_selected_options != selected_options:
    with st.spinner(text='Loading...'):
        df = yf.download(selected_options,start=start_date,end=end_date)
        portfolios = get_portfolio(df)
        st.session_state.mean_variance_optimization = mean_variance_optimization(df)
        st.session_state.HRP = HRP(df)
        st.session_state.mCAR = mCAR(df)
        st.session_state.black_litterman = black_litterman(df)
        st.session_state.portfolios = portfolios
    st.session_state.last_selected_options = selected_options

with left_column:
    plot_stock_prices()

with right_column:
    st.header('Select risk to generate weights for selected stocks')
    risk = st.slider("Risk: ",0.0,1.0,0.25)
    if st.button('Calculate weights',disabled=len(selected_options)<2):

        portfolios = st.session_state.portfolios
        # fig = go.Figure(data=go.Scatter(x=portfolios.Volatility,y=portfolios.Returns,mode='markers'))
        # st.plotly_chart(fig)
        st.subheader('Monte carlo simulation')
        st.markdown('Monte Carlo simulation is used to calculate the optimal weights for a portfolio of stocks. This simulation involves generating random values for the stock returns based on historical data, and then calculating the expected return and risk for different weight combinations of the stocks. The optimal weights are then determined based on the desired risk level and expected return.')
        min_volatility_port = portfolios.iloc[portfolios['Volatility'].idxmin()].to_frame().T
        optimal_risky_port = portfolios.iloc[((portfolios['Returns']-risk)/portfolios['Volatility']).idxmax()].to_frame().T
        st.write('Minimum volatility portfolio ', min_volatility_port)
        st.write('Optimal risky portfolio ', optimal_risky_port)
        st.write("---")

        st.subheader("Mean variance optimization")
        st.markdown("Mean Variance Optimization is a portfolio optimization technique that seeks to find the portfolio with the highest expected return for a given level of risk, or the lowest risk for a given level of expected return. MVO considers the expected returns of individual assets and their covariance matrix, and uses these inputs to mathematically optimize the portfolio.")
        st.write(st.session_state.mean_variance_optimization)
        st.write("---")

        st.subheader("Hierarchical Risk Parity (HRP)")
        st.markdown("Hierarchical Risk Parity is a portfolio optimization technique that takes into account the hierarchical structure of the market to build a diversified portfolio. The idea behind HRP is to cluster assets based on their similarity, and then allocate risk equally within each cluster, and then allocate capital to the clusters based on their total risk contribution to the portfolio.")
        st.write(st.session_state.HRP)
        st.write("---")

        st.subheader("Mean Conditional Value at Risk (mCVAR)")
        st.markdown("Mean Conditional Value at Risk is a portfolio optimization technique that considers the worst-case scenario when constructing a portfolio. CVaR seeks to minimize the expected loss of the portfolio beyond a certain level of risk. It takes into account the tail risk, which is the risk of extreme events that are unlikely to occur, but can have a significant impact on the portfolio.")
        st.write(st.session_state.mCAR)
        st.write("---")

        st.subheader("Black-Litterman Model")
        st.markdown("The Black-Litterman model is a portfolio optimization technique that combines the views of the investor with the market equilibrium to construct a portfolio. The model takes the market equilibrium as a starting point, and then adjusts the weights of the assets based on the investor's views. It takes into account the uncertainty of the views and uses a Bayesian approach to update the weights of the assets. The model also adjusts the weights of the assets to reflect the investor's risk tolerance.")
        st.write(st.session_state.black_litterman)
        # st.experimental_rerun()
