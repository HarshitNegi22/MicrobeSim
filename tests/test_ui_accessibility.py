from src.ui import theme


def test_result_components_expose_semantic_live_regions(monkeypatch):
    rendered = []
    monkeypatch.setattr(theme.st, "markdown", lambda markup, **kwargs: rendered.append(markup))

    theme.readout([("Growth rate", ("0.2", "1/h"), None)])
    theme.verdict("warn", "Model range", "Review the estimate.")

    assert 'role="group"' in rendered[0]
    assert 'aria-label="Key analysis results"' in rendered[0]
    assert 'role="status"' in rendered[1]
    assert 'aria-live="polite"' in rendered[1]
    assert 'aria-atomic="true"' in rendered[1]
