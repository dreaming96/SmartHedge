from data_handling import numerical_methods, BS_pricer, hedged_account
from SmartHedge import SmartHedge

if __name__ == "__main__":
    type = ["call"]
    #loop for train and test (2x MC)
    mc_instance = numerical_methods([])
    numerical_methods.random_numbers(mc_instance)
    simulated_prices = mc_instance.paths
    call_prices = simulated_prices * 0
    call_delta, call_gamma, call_vega, call_theta = call_prices.copy(), call_prices.copy(), call_prices.copy(), call_prices.copy()
    put_prices = simulated_prices * 0
    put_delta, put_gamma, put_vega, put_theta = put_prices.copy(), put_prices.copy(), put_prices.copy(), put_prices.copy()
    ttm = [n / len(simulated_prices) for n in range(1, len(simulated_prices))]
    ttm.reverse()
    ttm.insert(0, float(1))
    for i, row in simulated_prices.iterrows():
        for idx, j in enumerate(row):
            print("pricing option on day " + str(i) +", path " + str(idx))
            bs_instance = BS_pricer(S=j, call_delta=[], call_gamma=[], call_vega=[], call_theta=[], put_delta=[], put_gamma=[], put_vega=[], put_theta=[], ttm=ttm[i])
            for k in type:
                if k.upper() == "CALL":
                    call_price = BS_pricer.call_price(bs_instance)
                    call_prices.iloc[i, idx] = call_price
                    BS_pricer.greeks(bs_instance, k)
                    call_delta.iloc[i, idx] = bs_instance.call_delta[0]
                    call_gamma.iloc[i, idx] = bs_instance.call_gamma[0]
                    call_vega.iloc[i, idx] = bs_instance.call_vega[0]
                    call_theta.iloc[i, idx] = bs_instance.call_theta[0]
                else:
                    put_price = BS_pricer.put_price(bs_instance)
                    put_prices.iloc[i, idx] = put_price
                    BS_pricer.greeks(bs_instance, k)
                    put_delta.iloc[i, idx] = bs_instance.put_delta[0]
                    put_gamma.iloc[i, idx] = bs_instance.put_gamma[0]
                    put_vega.iloc[i, idx] = bs_instance.put_vega[0]
                    put_theta.iloc[i, idx] = bs_instance.put_theta[0]

    #calculating pnl and hedge pnl
    pnl = hedged_account.pnl(simulated_prices, call_prices)
    account_instance = hedged_account(delta=call_delta, gamma=call_gamma, spot_price=simulated_prices, option_price=call_prices, option_position=pnl)
    hedge = hedged_account.delta_hedging(account_instance)
    hedged_pnl, delta_diff, u = hedged_account.rebalance(account_instance, pnl, hedge)

    #LSTM class
    LSTM_instance = SmartHedge(M=simulated_prices[0:-1]/100, K=100, r=0.1, ttm=ttm[0:-1], call_delta=call_delta[0:-1], put_delta=put_delta[0:-1], sigma=0.2
                               ,S0=simulated_prices[0:-1], S1=simulated_prices[1:], C0=call_prices[0:-1], C1=call_prices[1:], delta_diff=delta_diff, u=hedged_pnl[0:-1])
    SmartHedge.LSTM(LSTM_instance)

    print(123)