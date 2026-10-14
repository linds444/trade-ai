from copy import deepcopy
import json

import pytest

from local_ai_trader.cli import main
from local_ai_trader.hyperliquid.venue import capture_venue_check
from local_ai_trader.paper.isolated_demo import isolated_demo
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from test_hyperliquid_decisions import scanner


@pytest.fixture
def prepared(scanner, monkeypatch):
    client, config, _ = scanner
    original = client.info
    def info(payload):
        result = deepcopy(original(payload))
        if payload["type"] == "metaAndAssetCtxs":
            for asset in result[0]["universe"]:
                asset.update(marginTableId=25 if asset["name"] == "ETH" else 20,
                             maxLeverage=25 if asset["name"] == "ETH" else 20, szDecimals=2)
        return result
    monkeypatch.setattr(client, "info", info)
    folder, _ = capture_venue_check(config, client=client)
    return config, folder


def test_synthetic_stress_uses_audited_rules_and_reopens_a_reconciled_separate_book(prepared, capsys):
    config, folder = prepared
    path = isolated_demo(config, folder)
    report = PerpetualPaperAccount(path).audit()
    assert report["source_mode"] == "synthetic" and report["fill_count"] == 3
    assert report["backstop_transfers"] == 1 and not report["positions"]
    assert report["missing_funding"] == [] and report["reconciled"]
    text = capsys.readouterr().out
    assert "invented prices" in text and "profitability month has not started" in text
    assert "PASS: isolated collateral" in text
    assert main(["perpetual-paper-status", str(path)]) == 0


def test_synthetic_command_is_self_contained(prepared):
    config, folder = prepared
    assert main(["hyperliquid-isolated-demo", "--config", str(config), "--venue-check", str(folder)]) == 0


def test_different_config_and_corrupted_source_cannot_seed_an_account(prepared):
    config, folder = prepared
    original = config.read_bytes()
    config.write_bytes(original + b"\n# different config\n")
    with pytest.raises(ValueError, match="config disagrees"):
        isolated_demo(config, folder)
    config.write_bytes(original)
    raw = folder/"raw.json"
    changed = json.loads(raw.read_text(encoding="utf-8"))
    changed["responses"]["contexts"][0]["universe"][0]["maxLeverage"] = 99
    raw.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        isolated_demo(config, folder)
    assert not (folder.parent.parent/"paper").exists()
