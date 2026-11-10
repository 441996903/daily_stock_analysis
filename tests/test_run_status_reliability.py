# -*- coding: utf-8 -*-
"""V1.0 reliability acceptance tests.

Covers the scenarios the first P0 wave left unproven:

- STRICT_RUN_STATUS=true turns a partial run (8/9) into a non-zero
  process exit code through the real ``main()`` single-run path, while
  the lenient default keeps returning success.
- Failure reasons are recorded for results flagged ``success=False``
  (code + ``error_message``), not only for raised exceptions.
- ``run_status.json`` is persisted for success, partial failure, total
  failure and market-review-only runs.
- For stats-capable markets (A-share), a snapshot whose breadth data is
  missing at runtime is marked ``data_quality="unavailable"`` so every
  downstream consumer (text report, share image, alerts) refuses to
  present the compatibility default score as a real score. Markets
  without breadth by design keep their ``partial`` semantics.
"""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

import main
from src.analyzer import AnalysisResult

NINE_CODES = [
    "600309", "600990", "603259", "600030", "601899",
    "000922", "600406", "002446", "002460",
]


def _dummy_config_class():
    class _DummyConfig(SimpleNamespace):
        def refresh_stock_list(self):
            pass

        def validate(self):
            return []

        def validate_structured(self):
            return []

    return _DummyConfig


def _make_config(temp_dir, **overrides):
    defaults = {
        "log_dir": temp_dir,
        "database_path": str(Path(temp_dir) / "data" / "stock_analysis.db"),
        "stock_list": list(NINE_CODES),
        "market_review_enabled": False,
        "daily_market_context_enabled": False,
        "market_review_region": "cn",
        "trading_day_check_enabled": False,
        "merge_email_notification": False,
        "single_stock_notify": False,
        "analysis_delay": 0,
        "report_type": "brief",
        "report_language": "zh",
        "schedule_enabled": False,
        "schedule_startup_dedup": False,
        "run_immediately": True,
        "webui_enabled": False,
    }
    defaults.update(overrides)
    return _dummy_config_class()(**defaults)


def _make_args(**overrides):
    defaults = {
        "debug": False,
        "stocks": None,
        "portfolio": "",
        "webui": False,
        "webui_only": False,
        "serve": False,
        "serve_only": False,
        "host": None,
        "port": None,
        "backtest": False,
        "etf_rotation": False,
        "market_review": False,
        "schedule": False,
        "no_run_immediately": False,
        "no_notify": True,
        "check_notify": False,
        "no_market_review": False,
        "dry_run": False,
        "workers": 1,
        "force_run": False,
        "single_notify": False,
        "no_context_snapshot": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _analysis_result(code, **kwargs):
    kwargs.setdefault("name", f"stock-{code}")
    kwargs.setdefault("sentiment_score", 60)
    kwargs.setdefault("trend_prediction", "震荡")
    kwargs.setdefault("operation_advice", "观望")
    return AnalysisResult(code=code, **kwargs)


def _make_fake_pipeline_cls(results, report_path, failures):
    class _FakeStockAnalysisPipeline:
        def __init__(self, **kwargs):
            self.notifier = SimpleNamespace(
                is_available=lambda: False,
                send=lambda *a, **k: False,
                generate_aggregate_report=lambda *a, **k: "",
            )
            self.analyzer = SimpleNamespace()
            self.search_service = SimpleNamespace()
            self._last_local_report_path = str(report_path)
            self.last_run_failures = list(failures)

        def run(self, **kwargs):
            return list(results)

    return _FakeStockAnalysisPipeline


class _RunFullAnalysisHarness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self._cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.addCleanup(os.chdir, self._cwd)
        self.report_path = Path(self.temp_dir.name) / "report.md"
        self.config = _make_config(self.temp_dir.name)
        self.args = _make_args()

    def _run(self, *, results, failures, config=None, args=None):
        fake_cls = _make_fake_pipeline_cls(results, self.report_path, failures)
        with ExitStack() as stack:
            stack.enter_context(patch("src.core.pipeline.StockAnalysisPipeline", fake_cls))
            stack.enter_context(patch("main._refresh_stock_index_cache_for_analysis"))
            stack.enter_context(patch("main._run_auto_backtest"))
            stack.enter_context(patch("main._prime_daily_market_context", return_value=("", "")))
            return main.run_full_analysis(config or self.config, args or self.args)

    def _read_status(self):
        path = Path(self.temp_dir.name) / "run_status.json"
        self.assertTrue(path.exists(), "run_status.json must be persisted")
        return json.loads(path.read_text(encoding="utf-8"))


class StrictExitCodeTestCase(_RunFullAnalysisHarness):
    def test_partial_failure_lenient_by_default(self):
        results = [_analysis_result(c) for c in NINE_CODES[:8]]
        failures = [{"code": "002460", "name": "赣锋锂业", "reason": "ambiguous_json"}]
        with patch.dict(os.environ, {"STRICT_RUN_STATUS": "false"}, clear=False):
            result = self._run(results=results, failures=failures)
        # Lenient upstream semantics: partial success still returns True,
        # and the 8 successful results are kept (report path untouched).
        self.assertTrue(result)
        status = self._read_status()
        self.assertEqual(status["status"], "partial_success")
        self.assertEqual(status["succeeded"], 8)
        self.assertEqual(status["total"], 9)

    def test_partial_failure_strict_returns_false_with_reason(self):
        results = [_analysis_result(c) for c in NINE_CODES[:8]]
        failures = [{"code": "002460", "name": "赣锋锂业", "reason": "ambiguous_json"}]
        with patch.dict(os.environ, {"STRICT_RUN_STATUS": "true"}, clear=False):
            result = self._run(results=results, failures=failures)
        self.assertFalse(result)
        status = self._read_status()
        self.assertEqual(status["status"], "partial_success")
        self.assertEqual(status["failed"], [{"code": "002460", "reason": "ambiguous_json"}])

    def test_all_failed_writes_failed_status(self):
        failures = [
            {"code": c, "name": f"stock-{c}", "reason": "provider timeout"}
            for c in NINE_CODES
        ]
        result = self._run(results=[], failures=failures)
        self.assertFalse(result)
        status = self._read_status()
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["succeeded"], 0)
        self.assertEqual(len(status["failed"]), 9)
        self.assertTrue(all(f["reason"] == "provider timeout" for f in status["failed"]))

    def test_full_success_writes_success_status(self):
        results = [_analysis_result(c) for c in NINE_CODES]
        result = self._run(results=results, failures=[])
        self.assertTrue(result)
        status = self._read_status()
        self.assertEqual(status["status"], "success")
        self.assertEqual(status["succeeded"], 9)
        self.assertEqual(status["failed"], [])


class MainExitCodeTestCase(_RunFullAnalysisHarness):
    """The actual process exit code: main() -> sys.exit(main())."""

    def _run_main_single_run(self, *, results, failures):
        fake_cls = _make_fake_pipeline_cls(results, self.report_path, failures)

        def _fake_lock(config, args, stock_codes=None, analysis_targets=None):
            with ExitStack() as stack:
                stack.enter_context(patch("src.core.pipeline.StockAnalysisPipeline", fake_cls))
                stack.enter_context(patch("main._refresh_stock_index_cache_for_analysis"))
                stack.enter_context(patch("main._run_auto_backtest"))
                stack.enter_context(patch("main._prime_daily_market_context", return_value=("", "")))
                return main.run_full_analysis(config, args, stock_codes)

        with ExitStack() as stack:
            stack.enter_context(patch("main.parse_arguments", return_value=self.args))
            stack.enter_context(patch("main.get_config", return_value=self.config))
            stack.enter_context(patch("main._run_analysis_with_runtime_scheduler_lock", _fake_lock))
            return main.main()

    def test_exit_code_1_on_partial_failure_under_strict(self):
        results = [_analysis_result(c) for c in NINE_CODES[:8]]
        failures = [{"code": "002460", "name": "赣锋锂业", "reason": "ambiguous_json"}]
        with patch.dict(
            os.environ,
            {"STRICT_RUN_STATUS": "true", "GITHUB_ACTIONS": "false"},
            clear=False,
        ):
            exit_code = self._run_main_single_run(results=results, failures=failures)
        self.assertEqual(exit_code, 1)

    def test_exit_code_0_on_partial_failure_when_lenient(self):
        results = [_analysis_result(c) for c in NINE_CODES[:8]]
        failures = [{"code": "002460", "name": "赣锋锂业", "reason": "ambiguous_json"}]
        with patch.dict(
            os.environ,
            {"STRICT_RUN_STATUS": "false", "GITHUB_ACTIONS": "false"},
            clear=False,
        ):
            exit_code = self._run_main_single_run(results=results, failures=failures)
        self.assertEqual(exit_code, 0)


class MarketOnlyStatusTestCase(_RunFullAnalysisHarness):
    def _run_main_market_only(self, review_return):
        args = _make_args(market_review=True)
        with ExitStack() as stack:
            stack.enter_context(patch("main.parse_arguments", return_value=args))
            stack.enter_context(patch("main.get_config", return_value=self.config))
            stack.enter_context(
                patch(
                    "src.core.market_review_runtime.build_market_review_runtime",
                    return_value=(SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
                )
            )
            stack.enter_context(
                patch("main._run_market_review_with_shared_lock", return_value=review_return)
            )
            return main.main()

    def test_market_only_success_writes_status(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}, clear=False):
            exit_code = self._run_main_market_only("# 大盘复盘")
        self.assertEqual(exit_code, 0)
        status = self._read_status()
        self.assertEqual(status["status"], "success")
        self.assertEqual(status["market_review"], "ok")
        self.assertEqual(status["total"], 0)

    def test_market_only_failure_writes_status(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}, clear=False):
            exit_code = self._run_main_market_only("")
        self.assertEqual(exit_code, 1)
        status = self._read_status()
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["market_review"], "failed")


class PipelineFailureRecordingTestCase(unittest.TestCase):
    """Item 2: success=False results and worker exceptions are recorded."""

    def _bare_pipeline(self, behavior):
        from src.core.pipeline import StockAnalysisPipeline

        pipe = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipe.config = SimpleNamespace(
            single_stock_notify=False, report_type="simple", analysis_delay=0
        )
        pipe.max_workers = 2
        pipe.fetcher_manager = SimpleNamespace(
            prefetch_daily_klines=lambda *a, **k: 0,
            prefetch_realtime_quotes=lambda *a, **k: 0,
            prefetch_stock_names=lambda *a, **k: None,
        )
        pipe._save_local_report = lambda *a, **k: None
        pipe.process_single_stock = lambda code, **kwargs: behavior[code]()
        return pipe

    def test_success_false_result_records_code_and_error_message(self):
        ok = _analysis_result("600406")
        bad = _analysis_result(
            "002460", success=False, error_message="LLM 返回的 JSON 无法解析"
        )
        pipe = self._bare_pipeline({"600406": lambda: ok, "002460": lambda: bad})

        results = pipe.run(["600406", "002460"], send_notification=False)

        self.assertEqual([r.code for r in results], ["600406"])
        self.assertEqual(
            pipe.last_run_failures,
            [{"code": "002460", "name": "stock-002460", "reason": "LLM 返回的 JSON 无法解析"}],
        )

    def test_worker_exception_records_code_and_reason(self):
        ok = _analysis_result("600406")

        def _boom():
            raise RuntimeError("fetch exploded")

        pipe = self._bare_pipeline({"600406": lambda: ok, "002460": _boom})

        results = pipe.run(["600406", "002460"], send_notification=False)

        self.assertEqual([r.code for r in results], ["600406"])
        self.assertEqual(len(pipe.last_run_failures), 1)
        failure = pipe.last_run_failures[0]
        self.assertEqual(failure["code"], "002460")
        self.assertIn("fetch exploded", failure["reason"])


def _make_market_analyzer(region, profile):
    with patch("src.analyzer.get_config") as mock_cfg, patch(
        "src.market_analyzer.get_config"
    ) as mock_cfg2:
        cfg = MagicMock()
        cfg.litellm_model = "gemini/gemini-2.0-flash"
        cfg.litellm_fallback_models = []
        cfg.gemini_api_keys = ["<redacted>"]
        cfg.anthropic_api_keys = []
        cfg.openai_api_keys = []
        cfg.deepseek_api_keys = []
        cfg.llm_model_list = []
        cfg.openai_base_url = None
        cfg.market_review_region = region
        cfg.market_review_color_scheme = "green_up"
        cfg.report_language = "zh"
        cfg.generation_backend = "litellm"
        cfg.generation_fallback_backend = "litellm"
        mock_cfg.return_value = cfg
        mock_cfg2.return_value = cfg

        from src.analyzer import GeminiAnalyzer
        from src.core.market_strategy import get_market_strategy_blueprint
        from src.market_analyzer import MarketAnalyzer

        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._router = None
        analyzer._litellm_available = True
        analyzer._config_override = cfg
        analyzer.get_generation_backend_identity = MagicMock(
            return_value=("litellm", cfg.litellm_model)
        )
        analyzer.generate_text_with_metadata = MagicMock(
            return_value=SimpleNamespace(
                text="复盘报告", provider="litellm", model=cfg.litellm_model, backend="litellm"
            )
        )

        ma = MarketAnalyzer.__new__(MarketAnalyzer)
        ma.analyzer = analyzer
        ma.config = cfg
        ma.profile = profile
        ma.strategy = get_market_strategy_blueprint(region)
        ma.region = region
        return ma


class MarketLightScoreGateTestCase(unittest.TestCase):
    """Item 4: the compatibility default score must not leak downstream."""

    def test_cn_snapshot_without_breadth_is_unavailable(self):
        from src.core.market_profile import CN_PROFILE
        from src.market_analyzer import MarketIndex, MarketOverview

        ma = _make_market_analyzer("cn", CN_PROFILE)
        overview = MarketOverview(
            date="2026-11-09",
            indices=[
                MarketIndex(code="000001", name="上证指数", current=3660.79, change_pct=0.32),
                MarketIndex(code="399001", name="深证成指", current=11145.49, change_pct=-0.27),
            ],
        )

        snapshot = ma.build_market_light_snapshot(overview)

        self.assertEqual(snapshot["data_quality"], "unavailable")
        self.assertFalse(snapshot["dimensions"]["breadth"]["available"])

    def test_cn_share_image_hides_default_score_when_breadth_missing(self):
        from src.core.market_profile import CN_PROFILE
        from src.market_analyzer import MarketIndex, MarketOverview
        from src.share_image import build_share_image_html

        ma = _make_market_analyzer("cn", CN_PROFILE)
        overview = MarketOverview(
            date="2026-11-09",
            indices=[
                MarketIndex(code="000001", name="上证指数", current=3660.79, change_pct=0.32),
            ],
        )
        snapshot = ma.build_market_light_snapshot(overview)

        html = build_share_image_html(
            "# A股大盘复盘\n\n> 宽度数据缺失，本次不做量化评分。\n",
            generated_on=date(2026, 11, 9),
            structured_payload={
                "kind": "market_review",
                "region": "cn",
                "market_light": snapshot,
            },
        )

        self.assertNotIn('class="market-signal"', html)
        self.assertNotIn("<small>/100</small>", html)

    def test_us_snapshot_without_breadth_stays_partial_and_scores(self):
        from src.core.market_profile import US_PROFILE
        from src.market_analyzer import MarketIndex, MarketOverview
        from src.share_image import build_share_image_html

        ma = _make_market_analyzer("us", US_PROFILE)
        overview = MarketOverview(
            date="2026-11-09",
            indices=[
                MarketIndex(code="SPX", name="S&P 500", current=5000.0, change_pct=0.5),
            ],
        )

        snapshot = ma.build_market_light_snapshot(overview)

        # Other markets keep their original rules: an index-based score
        # remains valid and is still rendered on the share image.
        self.assertEqual(snapshot["data_quality"], "partial")
        html = build_share_image_html(
            "# 美股大盘复盘\n\n> 指数小幅上涨。\n",
            generated_on=date(2026, 11, 9),
            structured_payload={
                "kind": "market_review",
                "region": "us",
                "market_light": snapshot,
            },
        )
        self.assertIn("<small>/100</small>", html)


if __name__ == "__main__":
    unittest.main()

# Wave-2 acceptance coverage (PR #1 follow-up).
