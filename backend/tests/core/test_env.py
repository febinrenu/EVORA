from evora.core.config import REPO_ROOT, load_env_file


def test_loads_values_without_overriding_the_environment(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# comment\n\nGROQ_KEYS=gsk_a,gsk_b\nexport OLLAMA_HOST='http://x:1'\nQUOTED=\"two words\"\nTRAILING=value # note\n"
        "EXISTING=from_file\nno_equals_line\n=novalue\n",
        encoding="utf-8",
    )
    env = {"EXISTING": "from_shell"}
    assert load_env_file(f, env) == 4
    assert env == {
        "EXISTING": "from_shell", "GROQ_KEYS": "gsk_a,gsk_b", "OLLAMA_HOST": "http://x:1",
        "QUOTED": "two words", "TRAILING": "value",
    }


def test_missing_file_is_fine(tmp_path):
    assert load_env_file(tmp_path / "nope", {}) == 0


def test_values_are_never_printed(tmp_path, capsys, caplog):
    f = tmp_path / ".env"
    f.write_text("SECRET=gsk_supersecret\n", encoding="utf-8")
    load_env_file(f, {})
    out = capsys.readouterr()
    assert "gsk_supersecret" not in out.out + out.err + caplog.text


def test_a_relative_model_cache_path_means_the_repo_folder(tmp_path):
    f = tmp_path / ".env"
    f.write_text("HF_HOME=./models/hf\n", encoding="utf-8")
    env: dict[str, str] = {}
    load_env_file(f, env)
    assert env["HF_HOME"] == str(REPO_ROOT / "models" / "hf")
    absolute = str(tmp_path / "cache")
    f.write_text(f"HF_HOME={absolute}\n", encoding="utf-8")
    env = {}
    load_env_file(f, env)
    assert env["HF_HOME"] == absolute
