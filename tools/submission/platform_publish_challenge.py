#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from platform_submit_selfplay import BASE_URL, CPP_LANG, DEFAULT_ENV, REPO_ROOT, _login, _read_login_key, _redact


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "temp" / "submission_runs" / "platform_challenge"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload one C++ .so as the public challenge model.")
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    so_path = args.so.resolve()
    if not so_path.exists():
        raise FileNotFoundError(so_path)
    if so_path.suffix != ".so":
        raise ValueError(f"expected .so file, got {so_path}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = args.output_root.resolve() / f"challenge_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    token = _login(session, _read_login_key(DEFAULT_ENV))
    upload_resp = _upload_challenge_model(session, token=token, so_path=so_path)
    list_resp = _list_challenge_models(session, token=token)

    summary = {
        "schema": "goldrush2_platform_challenge_upload_v1",
        "so_path": str(so_path),
        "so_bytes": so_path.stat().st_size,
        "run_dir": str(run_dir),
        "upload_response": _redact(upload_resp),
        "own_challenge_models": _own_models(list_resp),
    }
    _write_json(run_dir / "upload_response.json", _redact(upload_resp))
    _write_json(run_dir / "model_list_4.json", _redact(list_resp))
    _write_json(run_dir / "result.json", summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


def _upload_challenge_model(session: requests.Session, *, token: str, so_path: Path) -> dict[str, Any]:
    with so_path.open("rb") as model_file:
        files = {"model_file": (so_path.name, model_file, "application/octet-stream")}
        data = {"model_lang": CPP_LANG}
        resp = session.post(
            f"{BASE_URL}/api/user/add_model_4",
            headers={"Authorization": token},
            data=data,
            files=files,
            timeout=120,
        )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise RuntimeError(f"challenge upload failed: code={payload.get('code')} message={payload.get('message')!r}")
    return payload


def _list_challenge_models(session: requests.Session, *, token: str) -> dict[str, Any]:
    resp = session.get(f"{BASE_URL}/api/user/get_model_list_4", headers={"Authorization": token}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise RuntimeError(f"list challenge models failed: code={payload.get('code')} message={payload.get('message')!r}")
    return payload


def _own_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = (payload.get("data") or {}).get("list") or []
    return [
        {
            "id": item.get("id"),
            "stage": item.get("stage"),
            "user_id": item.get("user_id"),
            "lang": item.get("lang"),
            "name": item.get("name"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "user_name_cn": item.get("user_name_cn"),
        }
        for item in models
        if item.get("user_name_cn") == "gary318" or item.get("name") == "player142" or item.get("user_id") == 142
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
