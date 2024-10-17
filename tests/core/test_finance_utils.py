import pytest
import numpy as np

import deeptrade.util.finance as finance_utils

def test_log_returns():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_log_returns = np.array([np.log(2.0), np.log(3.0/2.0), np.log(4.0/3.0), np.log(5.0/4.0)])
    log_returns = finance_utils.calculate_log_returns(prices)
    assert expected_log_returns == pytest.approx(log_returns)
    
def test_simple_returns():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_simple_returns = np.array([2.0 - 1.0, 3.0/2.0 - 1.0, 4.0/3.0 - 1.0, 5.0/4.0 - 1.0])
    simple_returns = finance_utils.calculate_simple_returns(prices)
    assert expected_simple_returns == pytest.approx(simple_returns)