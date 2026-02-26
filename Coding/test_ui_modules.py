def test_market_tab_module_exports():
    from trading_terminal.ui import (
        render_ai_ml_suite_section,
        render_ai_strategy_backtest_section,
        render_home_tab,
        render_market_scanner_simulation_section,
        render_market_signals_section,
    )

    assert callable(render_ai_strategy_backtest_section)
    assert callable(render_ai_ml_suite_section)
    assert callable(render_home_tab)
    assert callable(render_market_signals_section)
    assert callable(render_market_scanner_simulation_section)
