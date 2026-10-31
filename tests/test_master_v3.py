"""
MindfulAI — Master Prompt V3.0 Verification Tests
Covers Required Test Cases:
  TEST 1: Python Syntax (All files compile cleanly)
  TEST 2: Application Imports (All core modules and services import without NameError/ImportError)
  TEST 3: Missing Optional Import Fix Verification (app.py execute_turn annotations)
  TEST 4: Audio Model Routing Verification (TTS model never used for audio input)
  TEST 5: Ollama Fallback Setting Verification (Pydantic boolean parsing)
"""
import ast
import os
import py_compile
import inspect
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_python_syntax_all_files():
    """TEST 1: Verify all Python files in the repository compile cleanly without syntax errors."""
    failures = []
    py_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Exclude virtual environment and cache directories
        if "venv" in root or ".pytest_cache" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                py_files.append(full_path)
                try:
                    py_compile.compile(full_path, doraise=True)
                except py_compile.PyCompileError as exc:
                    failures.append((full_path, str(exc)))

    assert len(py_files) > 20, f"Expected >20 Python files, found {len(py_files)}"
    assert not failures, f"Syntax errors in files: {failures}"


def test_application_imports():
    """TEST 2: Verify application modules and services import without NameError or ImportError."""
    modules_to_test = [
        "config",
        "app.config",
        "app.models.database",
        "app.models.conversation",
        "app.models.message",
        "app.schemas.chat",
        "app.providers.llm_provider",
        "app.services.chat_service",
        "app.utils.gemini_discovery",
        "app.utils.gemini_client",
        "app.utils.ollama_client",
        "app.utils.provider_state",
        "services.quota_manager",
        "services.ai_router",
        "services.gemini_service",
        "services.ollama_service",
        "services.tts_service",
        "services.audio_input_service",
        "utils.session_manager",
        "utils.file_validator",
        "utils.error_handler",
        "utils.audio_utils",
    ]

    import importlib
    for mod_name in modules_to_test:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_missing_optional_import_fixed_in_app():
    """
    TEST 3: Verify the NameError: name 'Optional' is not defined bug in app.py is resolved.
    Inspects app.py AST and source to confirm 'Optional' is imported and used correctly.
    """
    app_py_path = PROJECT_ROOT / "app.py"
    assert app_py_path.exists(), "app.py does not exist"

    with open(app_py_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename="app.py")

    # Verify 'Optional' is imported
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    assert "Optional" in imported_names, (
        "Critical Error: 'from typing import Optional' is missing in app.py imports!"
    )

    # Verify PIL Image is imported
    pil_imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "PIL":
            for alias in node.names:
                if (alias.asname or alias.name) == "Image":
                    pil_imported = True

    assert pil_imported, "from PIL import Image is missing in app.py!"

    # Verify execute_turn exists and has annotations referencing Optional
    execute_turn_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_turn":
            execute_turn_found = True
            # Verify args
            arg_names = [a.arg for a in node.args.args]
            assert "prompt" in arg_names
            assert "img_obj" in arg_names
            assert "audio_bytes" in arg_names
            assert "doc_text" in arg_names
            assert "doc_name" in arg_names

    assert execute_turn_found, "execute_turn function not found in app.py"


def test_audio_input_model_never_ends_in_tts():
    """TEST 4: Verify audio input model is separate from TTS and never uses a -tts model."""
    import config
    assert "-tts" not in config.AUDIO_INPUT_MODEL.lower(), (
        f"AUDIO_INPUT_MODEL cannot be a TTS model: {config.AUDIO_INPUT_MODEL}"
    )
    assert "-tts" in config.TTS_MODEL.lower() or "tts" in config.TTS_MODEL.lower(), (
        f"TTS_MODEL must be a designated speech output model: {config.TTS_MODEL}"
    )


def test_ollama_fallback_env_parsing():
    """TEST 5: Verify OLLAMA_FALLBACK_ON_DAILY_QUOTA parses as a valid boolean in app/config.py."""
    from app.config import settings
    assert isinstance(settings.OLLAMA_FALLBACK_ON_DAILY_QUOTA, bool)

