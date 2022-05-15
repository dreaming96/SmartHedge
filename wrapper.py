from data_handling import numerical_methods, BS_pricer, hedged_account

if __name__ == "__main__":
    type = ["call", "put"]
    #loop for train and test (2x MC)
    mc_instance = numerical_methods([])
    numerical_methods.random_numbers(mc_instance)
    simulated_prices = mc_instance.paths
    call_prices = simulated_prices * 0
    call_delta, call_gamma, call_vega, call_theta = call_prices.copy(), call_prices.copy(), call_prices.copy(), call_prices.copy()
    put_prices = simulated_prices * 0
    put_delta, put_gamma, put_vega, put_theta = put_prices.copy(), put_prices.copy(), put_prices.copy(), put_prices.copy()
    for i, row in simulated_prices.iterrows():
        for idx, j in enumerate(row):
            bs_instance = BS_pricer(S=j, call_delta=[], call_gamma=[], call_vega=[], call_theta=[], put_delta=[], put_gamma=[], put_vega=[], put_theta=[])
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

    pnl = hedged_account.pnl(simulated_prices, call_prices)
    account_instance = hedged_account(delta=call_delta, gamma=call_gamma, spot_price=simulated_prices, option_position=pnl)
    hedge = hedged_account.delta_hedging(account_instance)

    print(123)