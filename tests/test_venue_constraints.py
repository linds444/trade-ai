from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import json

import pytest

from local_ai_trader.cli import main
from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid import decisions, venue
from local_ai_trader.hyperliquid.venue import (
    VenueRules, parse_rules, validate_price, round_price, floor_size,
    maintenance_margin, isolated_long_liquidation_price, validate_entry_order,
    capture_venue_check, load_venue_check)
from local_ai_trader.models.train import file_digest

from test_hyperliquid_decisions import scanner, NOW


def rules(size_decimals=3):
    return VenueRules("BTC", size_decimals, 20, 60,
                      ((Decimal(0), 20), (Decimal(1000), 10), (Decimal(5000), 5)))


def metadata(table_id=60):
    return {"collateralToken": 0, "universe": [{"name": "BTC", "szDecimals": 3, "maxLeverage": 20,
            "marginTableId": table_id}], "marginTables": [[60, {"marginTiers": [
            {"lowerBound": "0", "maxLeverage": 20}, {"lowerBound": "1000", "maxLeverage": 10},
            {"lowerBound": "5000", "maxLeverage": 5}]}]]}


def complete_metadata(client, monkeypatch):
    original = client.info
    def info(payload):
        result = original(payload)
        if payload["type"] == "metaAndAssetCtxs":
            result[0]["marginTables"] = [[60, {"marginTiers": [{"lowerBound": "0", "maxLeverage": 20}, {"lowerBound": "1000", "maxLeverage": 10}]}]]
            for asset in result[0]["universe"]:
                asset["marginTableId"] = 60
                if asset["name"] == "xyz:HOOD":
                    asset.update(onlyIsolated=True, marginMode="noCross")
        return result
    monkeypatch.setattr(client, "info", info)


def test_parse_explicit_native_and_implicit_hip3_margin_tables():
    assert parse_rules(metadata(), "BTC") == rules()
    source = metadata(20)
    source["marginTables"] = []
    source["universe"][0].update(onlyIsolated=True, marginMode="noCross")
    actual = parse_rules(source, "BTC")
    assert actual.tiers == ((Decimal(0), 20),) and actual.only_isolated


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(collateralToken=1),
    lambda m: m["universe"].append(deepcopy(m["universe"][0])),
    lambda m: m["universe"][0].update(isDelisted=True),
    lambda m: m["universe"][0].pop("marginTableId"),
    lambda m: m.update(marginTables=[]),
    lambda m: m["marginTables"].append(deepcopy(m["marginTables"][0])),
    lambda m: m["marginTables"][0][1]["marginTiers"][0].update(lowerBound="1"),
    lambda m: m["marginTables"][0][1]["marginTiers"][1].update(lowerBound="0"),
    lambda m: m["marginTables"][0][1]["marginTiers"][1].update(maxLeverage=25),
    lambda m: m["marginTables"][0][1]["marginTiers"][0].update(maxLeverage=10),
    lambda m: m["universe"][0].update(marginMode="unknown"),
    lambda m: m["universe"][0].update(onlyIsolated="yes"),
    lambda m: m["universe"][0].update(maxLeverage=True),
])
def test_missing_inconsistent_or_unsupported_margin_evidence_is_refused(mutation):
    source = metadata()
    mutation(source)
    with pytest.raises((ValueError, KeyError)):
        parse_rules(source, "BTC")


@pytest.mark.parametrize("price,decimals", [("1234.5", 0), ("123456", 5), ("0.001234", 0), ("0.01234", 1), ("100.0100", 3)])
def test_documented_valid_prices_and_integer_exception(price, decimals):
    assert validate_price(price, rules(decimals)) == Decimal(price)


@pytest.mark.parametrize("price,decimals", [("1234.56", 0), ("0.0012345", 0), ("0.012345", 1), ("1.234", 4), ("100000000000000000000.000001", 0)])
def test_price_precision_is_not_silently_rounded_or_lost(price, decimals):
    with pytest.raises(ValueError):
        validate_price(price, rules(decimals))


@pytest.mark.parametrize("value,buy,sell", [("12345.6", "12346", "12345"), ("0.00123456", "0.001235", "0.001234"), ("99999.99", "100000", "99999")])
def test_order_limit_rounding_is_directional_and_valid_across_magnitude_changes(value, buy, sell):
    assert round_price(value, rules(0), side="BUY") == Decimal(buy)
    assert round_price(value, rules(0), side="SELL") == Decimal(sell)


def test_size_rounding_never_increases_allocated_quantity_and_preserves_kpepe_units():
    assert floor_size("1.2349", rules()) == Decimal("1.234")
    pepe = VenueRules("kPEPE", 0, 10, 10, ((Decimal(0), 10),))
    assert floor_size("12345.99", pepe) == 12345
    assert validate_entry_order(2500, ".004", ".004", pepe)["entry_notional"] == "10"
    with pytest.raises(ValueError, match="minimum"):
        validate_entry_order(2499, ".004", ".004", pepe)


def test_maintenance_deductions_keep_every_tier_boundary_continuous():
    actual = rules()
    assert maintenance_margin(999, actual) == Decimal("24.975")
    assert maintenance_margin(1000, actual) == Decimal("25")
    assert maintenance_margin(1001, actual) == Decimal("25.05")
    assert maintenance_margin(5000, actual) == Decimal("225")
    assert maintenance_margin(5001, actual) == Decimal("225.1")
    assert maintenance_margin(0, actual) == 0


def test_native_btc_large_position_uses_maintenance_deduction_not_flat_higher_rate():
    actual = VenueRules("BTC", 5, 40, 56, ((Decimal(0), 40), (Decimal("150000000"), 20)))
    assert maintenance_margin("150000000", actual) == Decimal("1875000")
    assert maintenance_margin("151000000", actual) == Decimal("1900000")


def test_liquidation_solve_uses_tier_at_root_rather_than_entry_notional():
    actual = rules()
    assert isolated_long_liquidation_price(1, 10000, 8075, actual) == 2000
    assert maintenance_margin(2000, actual) == 75
    assert isolated_long_liquidation_price(1, 10000, 9025, actual) == 1000
    assert isolated_long_liquidation_price(1, 100, 100, actual) == 0
    threshold = isolated_long_liquidation_price(1, 100, "99.9", actual)
    assert abs(threshold-Decimal("0.1025641025641025641025641026")) < Decimal("1e-25")


@pytest.mark.parametrize("changes", [{"side": "SELL"}, {"leverage": 2}, {"leverage": True}, {"margin_mode": "cross"}])
def test_configured_entry_restrictions_and_required_isolation(changes):
    actual = replace(rules(), only_isolated=True)
    with pytest.raises(ValueError):
        validate_entry_order(1, 100, 100, actual, **changes)


def test_minimum_opening_notional_and_native_size_precision_boundaries():
    assert validate_entry_order("0.1", 100, 101, rules())["initial_margin_at_mark_1x"] == "10.1"
    with pytest.raises(ValueError, match="minimum"):
        validate_entry_order("0.099", 100, 100, rules())
    with pytest.raises(ValueError, match="precision"):
        validate_entry_order("0.1001", 100, 100, rules())


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "-Infinity", "1e31", "1e-31"])
def test_nonfinite_boolean_or_out_of_range_inputs_cannot_enter_margin_math(value):
    with pytest.raises(ValueError):
        maintenance_margin(value, rules())


def test_mainnet_capture_is_auditable_without_candles_models_or_accounts(scanner, monkeypatch):
    client, config, models = scanner
    complete_metadata(client, monkeypatch)
    original = client.info
    def no_candles(payload):
        assert payload["type"] in ("l2Book", "metaAndAssetCtxs")
        return original(payload)
    monkeypatch.setattr(client, "info", no_candles)
    path, report = capture_venue_check(config, client=client)
    assert load_venue_check(path) == report
    assert all(row["status"] == "READY" for row in report["markets"].values())
    assert report["markets"]["xyz:HOOD"]["isolated_required"] is True
    assert report["entries_enabled"] is False and report["account_risk_authorized"] is False
    assert not (client.settings.observations.parent/"paper").exists()
    assert not (client.settings.observations.parent/"forward").exists()


def test_missing_tier_metadata_remains_unavailable_without_guessing_from_leverage(scanner):
    client, config, models = scanner
    path, report = capture_venue_check(config, client=client)
    assert all(row["status"] == "UNAVAILABLE" for row in report["markets"].values())
    assert load_venue_check(path) == report


def test_bundle_cannot_replace_derived_rules_even_with_unchanged_source_hashes(scanner, monkeypatch):
    client, config, models = scanner
    complete_metadata(client, monkeypatch)
    path, report = capture_venue_check(config, client=client)
    report["markets"]["BTC"]["minimum_entry_notional_usdc"] = "0"
    write_json(path/"venue_check.json", report)
    with pytest.raises(ValueError, match="derived rules changed"):
        load_venue_check(path)


def test_expired_context_before_publication_cannot_publish_ready_rules(scanner, monkeypatch):
    client, config, models = scanner
    complete_metadata(client, monkeypatch)
    original = venue.write_json
    def slow(path, payload):
        original(path, payload)
        if path.name == "venue_check.json":
            client.clock = lambda: NOW+timedelta(seconds=30)
    monkeypatch.setattr(venue, "write_json", slow)
    with pytest.raises(ValueError, match="expired or changed"):
        capture_venue_check(config, client=client)
    assert list((client.settings.observations.parent/"venue_checks").iterdir()) == []


def test_scanner_archives_venue_constraints_without_authorizing_entries(scanner, monkeypatch):
    client, config, models = scanner
    complete_metadata(client, monkeypatch)
    path, report = decisions.create_decision_preview(client, config, models)
    assert report["venue_constraints"]["markets"]["xyz:HOOD"]["isolated_required"] is True
    assert report["venue_constraints"]["account_risk_authorized"] is False
    assert report["paper_fills_created"] is False
    assert all(row["confidence"] is None for row in report["decisions"].values())


def test_cli_and_deep_windows_archival_paths(scanner, monkeypatch):
    client, config, models = scanner
    complete_metadata(client, monkeypatch)
    monkeypatch.setattr(venue, "HyperliquidPublicClient", lambda settings: client)
    assert main(["hyperliquid-venue-check", "--config", str(config)]) == 0
    deep = native_path(config.parent.parent/(("deep_"*20))/(("venue_"*20))/"config"/"hyperliquid.toml")
    deep.parent.mkdir(parents=True)
    deep.write_bytes(config.read_bytes())
    client.settings = venue.load_hyperliquid_settings(deep)
    path, report = capture_venue_check(deep, client=client)
    assert len(str(path)) > 260 and load_venue_check(path) == report
