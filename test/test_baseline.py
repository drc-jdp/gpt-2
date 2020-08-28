import pytest
import requests
import re


def t_expected_output(url: str, text: str, expect: str):
    text = text.strip()
    response = requests.post(f"{url}/autocomplete", json={
        "text": text
    })
    if response.status_code > 200:
        assert False, "http request should success"
        return
    results: dict = response.json()
    assert "result" in results, "response should have key result"
    results = results.get("result")
    for result in results:
        found = re.search(expect, result)
        if found:
            break
    assert found, f"should suggest {expect} ({results})"


def test_reference_obp(url: str):
    t_expected_output(url, """\
    Only one bump pitch can be selected at a time""", "COPY")


def test_reference_ubm(url: str):
    t_expected_output(url, """\
    UBM_PITCH:ERROR5 { Only one bump pitch can be selected at a time""", "COPY")


def test_reference_M1W(url: str):
    t_expected_output(url, "M1.W.1 { @ ", "Width")
    # t_expected_output(url, "M1.W.1 ", "@")
    t_expected_output(url, "M1.W.1", " { @ Width")


def test_reference_M1S(url: str):
    t_expected_output(url, "M1.S.12 { @", "Space")
    t_expected_output(url, "M1.S.12", " { @ Space")


def test_IFO(url: str):
    t_expected_output(url, "IFO_IND.W.", "{ @")


def test_orthogonal(url: str):
    t_expected_output(url, "ORTHOGONAL", "ONLY\n")


def test_notrect(url: str):
    t_expected_output(url, "NOT RECTANGLE ", "_D")


def test_enc(url: str):
    t_expected_output(url, "M1.EN.31.12.1.T", "{ @ Enclosure")


def test_T(url: str):
    t_expected_output(url, "M1.EN", ".T")


def test_abut(url: str):
    t_expected_output(url, "ABUT", " < ")


def test_ar(url: str):
    t_expected_output(url, "A.R.6__A", "A.R")


def test_newline(url: str):
    t_expected_output(url, "}", "<|endof")


def test_netarea(url: str):
    t_expected_output(url, "NET AREA", "RATIO")


def test_cumulative(url: str):
    t_expected_output(url, "cumulative", "area")


def test_dExclamation(url: str):
    t_expected_output(url, "  !!", "A.EA[(]")
    t_expected_output(url, "!!", "A.EA[(]")
    t_expected_output(url, "[\n  !!", "AREA[(]")


def test_riskFloating(url: str):
    t_expected_output(url, "A.R.8.3:M17_M0", "{ @ Risk_Floating_net")


def test_dumVia(url: str):
    t_expected_output(
        url,
        "DENSITY DUMVIA",
        "[0-9]+ DUM[0-9]+ < UBM_R_[0-9]+ INSIDE"
    )


def test_density(url: str):
    t_expected_output(
        url,
        "A10 = DENSITY DUMVIA10 DUM10 < UBM_R_7 INSIDE OF LAYER CHIPx PRINT \
            UBM.R.7_DVIA10DM10.density",
        r"\n[\s]+\[AREA\(DUM"
    )


def test_shape_angle(url: str):
    t_expected_output(
        url,
        "G.3:CO_SRAM51 { @ Shapes must be orthogonal or on a 45 degree angle.",
        r"\n[\s]+ANGLE CO_SRAM51.*[>|<]"
    )
    t_expected_output(
        url,
        "{ @ Shapes must be orthogonal or on a 45 degree angle.",
        r"ANGLE.*[>|<]"
    )


def test_reference_consistent(url: str):
    text = "M1"
    results = []
    for i in range(0, 10):
        response = requests.post(f"{url}/autocomplete", json={
            "text": text
        })
        if response.status_code > 200:
            assert False, "http request should success"
            return
        result: dict = response.json()
        assert "result" in result, "response should have key result"
        results.append(result.get("result"))
    assert len(results) == 10, "should have exactly 10 prediction results"
    batch_size = len(results[0])
    for i in range(1, 10):
        assert len(results[1]) == batch_size, "length of all results must be same"
    wanted = results[0]
    for i in range(1, 10):
        for j in range(len(results[0])):
            assert wanted[j] == results[i][j], "\
                results must be equal to others for the same input"
