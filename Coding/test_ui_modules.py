def test_market_tab_module_exports():
    from trading_terminal.ui import (
        render_market_scanner_simulation_section,
        render_market_signals_section,
    )

    assert callable(render_market_signals_section)
    assert callable(render_market_scanner_simulation_section)

