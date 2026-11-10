from validation.step5.run_validation import _validation_exit_code


def test_validation_harness_returns_nonzero_when_any_paper_fails():
    assert _validation_exit_code([{"status": "success"}]) == 0
    assert _validation_exit_code([{"status": "failed"}]) == 1
    assert _validation_exit_code(
        [{"status": "success"}, {"status": "failed"}]
    ) == 1
