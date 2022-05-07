from data_handling import numerical_methods, BS_pricer

if __name__ == "__main__":
    #loop for train and test (2x MC)
    instance = numerical_methods([])
    numerical_methods.MC(instance)
    BS_pricer.call_price()
    BS_pricer.put_price()
    BS_pricer.greeks()
    print(123)