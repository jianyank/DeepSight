# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_datasets_are_registered():
    registry = json.loads((ROOT / "data/dataset_info.json").read_text(encoding="utf-8"))
    for config in (ROOT / "config").glob("ad_bev_v*.yaml"):
        dataset = next(
            line.split(":", 1)[1].strip()
            for line in config.read_text(encoding="utf-8").splitlines()
            if line.startswith("dataset:")
        )
        assert dataset in registry, f"{config.name}: undefined dataset {dataset}"
    target = registry["bench2drive_bev_target"]
    assert target["file_name"] == "bench2drive_bev_target.jsonl"
    assert target["formatting"] == "sharegpt"
    assert target["columns"] == {"messages": "messages", "images": "images"}
    assert target["tags"]["assistant_tag"] == "assistant"


def test_training_launcher_contract():
    linux = (ROOT / "scripts/train/train.sh").read_text(encoding="utf-8")
    windows = (ROOT / "scripts/train/train.bat").read_text(encoding="utf-8")
    wrapper = (ROOT / "nebula.sh").read_text(encoding="utf-8")
    assert linux.count("exec torchrun ") == 1
    assert 'src/train.py "$CONFIG" "$@"' in linux
    assert 'scripts/train/train.sh" "$@"' in wrapper
    assert "llamafactory-cli train" not in wrapper
    assert "config/ad_bev_v4.yaml" in linux
    assert "config\\ad_bev_v4.yaml" in windows
    assert 'cd /d "%~dp0..\\.."' in windows
    assert "%CD%\\src\\transformers\\src" in windows
    assert "\t" not in windows
    for launcher in (linux, windows):
        assert "--config" not in launcher
        assert "PYTHONPATH" in launcher
        assert "transformers" in launcher


def test_requirements_use_vendored_transformers():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "transformers==4.56.0.dev0" in requirements
    assert '__version__ = "4.56.0.dev0"' in (
        ROOT / "src/transformers/src/transformers/__init__.py"
    ).read_text(encoding="utf-8")
