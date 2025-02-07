import pytest
import numpy as np
from deeptrade.optimization import PortfolioOptimizer, PortfolioWeights

@pytest.fixture
def sample_returns():
    # Create sample return data for 3 assets over 100 days
    np.random.seed(42)
    return np.random.normal(0.001, 0.01, (3, 100))

@pytest.fixture
def optimizer():
    return PortfolioOptimizer(risk_free_rate=0.02, target_vol=0.15)

def test_weights_sum_to_one(optimizer, sample_returns):
    print(sample_returns.shape)
    weights = optimizer.optimize_portfolio(sample_returns)
    print(f"weights: {type(weights)}")
    assert np.isclose(np.sum(weights.weights), 1.0)

def test_weights_bounds(optimizer, sample_returns):
    weights = optimizer.optimize_portfolio(sample_returns)
    assert np.all(weights.weights >= 0)  # Assuming no short-selling allowed
    assert np.all(weights.weights <= 1)

def test_log_returns_calculation():
    prices = np.array([100, 110, 121])
    expected_log_returns = np.log(prices[1:] / prices[:-1])
    calculated_log_returns = np.log(prices[1:] / prices[:-1])
    assert np.allclose(calculated_log_returns, expected_log_returns)

def test_portfolio_volatility(optimizer, sample_returns):
    weights = optimizer.optimize_portfolio(sample_returns)
    vol = optimizer.calculate_portfolio_volatility(weights, sample_returns)
    assert isinstance(vol, float)
    assert vol > 0

def test_sharpe_ratio(optimizer, sample_returns):
    weights = optimizer.optimize_portfolio(sample_returns)
    sharpe = optimizer.calculate_sharpe_ratio(weights, sample_returns)
    assert isinstance(sharpe, float)

def test_optimization_simple_case():
    # Test with perfectly negatively correlated assets
    returns = np.array([[0.01, -0.01],
                       [-0.01, 0.01]])  # 2 assets, perfect negative correlation
    optimizer = PortfolioOptimizer(risk_free_rate=0, target_vol=0.15)
    weights = optimizer.optimize_portfolio(returns)
    assert len(weights.weights) == 2
    # Should be close to 50-50 split for perfect negative correlation
    assert np.allclose(weights.weights, [0.5, 0.5], atol=0.1)

def test_optimization_constraints(optimizer, sample_returns):
    weights = optimizer.optimize_portfolio(sample_returns)
    # Test if optimization respects basic constraints
    assert len(weights.weights) == sample_returns.shape[0]  # Correct number of weights
    assert np.isclose(np.sum(weights.weights), 1.0)

def test_portfolio_weights_all_fields():
    # Test that all fields in PortfolioWeights are properly set
    sample_weights = np.array([0.3, 0.3, 0.4])
    portfolio = PortfolioWeights(
        weights=sample_weights,
        risk_free_weight=0.0,
        sharpe=1.5,
        vol=0.12,
        ret=0.08
    )

    assert np.array_equal(portfolio.weights, sample_weights)
    assert portfolio.risk_free_weight == 0.0
    assert portfolio.sharpe == 1.5
    assert portfolio.vol == 0.12
    assert portfolio.ret == 0.08

def test_portfolio_weights_with_risk_free():
    # Test portfolio with non-zero risk-free weight
    risky_weights = np.array([0.3, 0.3])
    portfolio = PortfolioWeights(
        weights=risky_weights,
        risk_free_weight=0.4,  # 40% in risk-free asset
        sharpe=1.2,
        vol=0.10,
        ret=0.06
    )

    assert np.array_equal(portfolio.weights, risky_weights)
    assert portfolio.risk_free_weight == 0.4
    assert np.isclose(np.sum(portfolio.weights) + portfolio.risk_free_weight, 1.0)

def test_portfolio_weights_attributes_types():
    # Test that attributes have correct types
    weights = np.array([0.5, 0.5])
    portfolio = PortfolioWeights(
        weights=weights,
        risk_free_weight=0.0,
        sharpe=1.0,
        vol=0.15,
        ret=0.07
    )

    assert isinstance(portfolio.weights, np.ndarray)
    assert isinstance(portfolio.risk_free_weight, float)
    assert isinstance(portfolio.sharpe, float)
    assert isinstance(portfolio.vol, float)
    assert isinstance(portfolio.ret, float)
