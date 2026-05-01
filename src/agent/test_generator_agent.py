"""
test_generator_agent.py – Test Case Generator Agent
=====================================================
A custom agentic tool that autonomously analyzes source modules and
generates pytest unit test cases for any function in the project.

Public API
----------
TestGeneratorAgent(module_path, output_path)
  .generate()   – Analyze module and write test file
  .preview()    – Return generated test code without writing

CLI usage:
    python -m src.agent.test_generator_agent --module src/trend_detection.py
    python -m src.agent.test_generator_agent --module src/scenario_simulation.py --output tests/
    python -m src.agent.test_generator_agent --all
"""

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── Module registry: source modules to auto-discover ─────────────────────────
_MODULE_REGISTRY = {
    "data_loader":        "src/data_loader.py",
    "expense_classifier": "src/expense_classifier.py",
    "trend_detection":    "src/trend_detection.py",
    "scenario_simulation":"src/scenario_simulation.py",
    "genai_interface":    "src/genai_interface.py",
    "agent_tools":        "src/agent/tools.py",
    "agent_core":         "src/agent/agent_core.py",
}


# ── Dataclass to hold function metadata ──────────────────────────────────────
@dataclass
class FunctionInfo:
    name: str
    docstring: str
    args: List[str] = field(default_factory=list)
    returns: str = ""
    source_file: str = ""


# ── AST-based module analyzer ─────────────────────────────────────────────────
def extract_functions(source_path: str) -> List[FunctionInfo]:
    """
    Parse a Python source file and extract all public function definitions.

    Uses Python's AST module to statically analyze the source without
    importing it, which avoids side effects and import errors.

    Args:
        source_path (str): Absolute or relative path to the .py file.

    Returns:
        List[FunctionInfo]: Metadata for each public function found.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {source_path}")

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions: List[FunctionInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            docstring = ast.get_docstring(node) or ""
            args = [a.arg for a in node.args.args if a.arg != "self"]
            # Try to get return annotation
            returns: str = ""
            if node.returns is not None:
                try:
                    returns = ast.unparse(node.returns)  # type: ignore[arg-type]
                except Exception:
                    returns = ""
            functions.append(FunctionInfo(
                name=node.name,
                docstring=docstring,
                args=args,
                returns=returns,
                source_file=str(path),
            ))
    return functions


# ── LLM-powered test generator ────────────────────────────────────────────────
class TestGeneratorAgent:
    """
    Agentic test case generator that uses an LLM to write pytest test code.

    The agent:
      1. Parses the target module via AST to extract function signatures/docstrings.
      2. Builds a structured prompt describing every function.
      3. Calls the project's configured LLM (Groq) to generate test cases.
      4. Writes the output to the tests/ directory.

    Attributes:
        module_path (str): Path to the source module to test.
        output_dir (str): Directory where the test file will be written.

    Example::

        agent = TestGeneratorAgent("src/trend_detection.py")
        agent.generate()
    """

    def __init__(self, module_path: str, output_dir: str = "tests") -> None:
        self.module_path = module_path
        self.output_dir = output_dir
        self._functions: List[FunctionInfo] = []

    # ── internal helpers ───────────────────────────────────────────────────

    def _analyze(self) -> None:
        """Extract function metadata from the module."""
        self._functions = extract_functions(self.module_path)
        if not self._functions:
            print(f"  ⚠  No public functions found in {self.module_path}")

    def _build_prompt(self) -> str:
        """Build the LLM prompt describing all functions in the module."""
        module_name = Path(self.module_path).stem
        lines = [
            f"You are an expert Python test engineer. Generate comprehensive pytest unit tests "
            f"for the module `{module_name}` located at `{self.module_path}`.\n",
            "## Functions to test:\n",
        ]
        for fn in self._functions:
            lines.append(f"### `{fn.name}({', '.join(fn.args)})`")
            if fn.returns:
                lines.append(f"Returns: `{fn.returns}`")
            if fn.docstring:
                lines.append(f"Docstring:\n{textwrap.indent(fn.docstring, '  ')}")
            lines.append("")

        lines += [
            "## Requirements for the generated tests:",
            "- Use `pytest` with descriptive test function names.",
            "- Include a module-level docstring describing what is being tested.",
            "- Add `@pytest.fixture` where shared test data is needed.",
            "- Cover: happy path, edge cases, invalid inputs, boundary values.",
            "- For DataFrame-dependent functions, create sample DataFrames inline.",
            "- Mock external dependencies (e.g. LLM calls, file I/O) with `unittest.mock`.",
            "- Include at least 3 test cases per function.",
            "- Add inline comments explaining each test's purpose.",
            "- Do NOT import from `__main__`; use the correct module import path.",
            f"- Import path for this module: `from {self._import_path()} import *`\n",
            "Output ONLY valid Python code — no markdown fences, no explanation.",
        ]
        return "\n".join(lines)

    def _import_path(self) -> str:
        """Convert file path to Python import path."""
        p = Path(self.module_path)
        # Remove .py, replace / with .
        parts = list(p.with_suffix("").parts)
        # Drop leading '.' if relative
        if parts and parts[0] in (".", ".."):
            parts = parts[1:]
        return ".".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """Invoke the configured LLM and return the generated test code."""
        try:
            from src.genai_interface import get_llm
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            llm = get_llm()
            chat_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a senior Python test engineer. "
                    "Write complete, runnable pytest test files. "
                    "Output only valid Python code — no markdown, no explanation.",
                ),
                ("human", "{prompt}"),
            ])
            chain = chat_prompt | llm | StrOutputParser()
            return chain.invoke({"prompt": prompt})
        except Exception as exc:
            print(f"  ⚠  LLM call failed: {exc}")
            print("     Falling back to template-based generation.")
            return self._template_fallback()

    def _template_fallback(self) -> str:
        """
        Generate basic test stubs without an LLM when the API is unavailable.

        Returns:
            str: Minimal pytest test file with one stub per function.
        """
        module_name = Path(self.module_path).stem
        import_path = self._import_path()
        lines = [
            f'"""Auto-generated test stubs for {module_name}."""',
            "import pytest",
            f"from {import_path} import {', '.join(f.name for f in self._functions)}\n\n",
        ]
        for fn in self._functions:
            test_name = f"test_{fn.name}_basic"
            args_placeholder = ", ".join(["None"] * len(fn.args))
            lines += [
                f"def {test_name}():",
                f'    """Stub: test {fn.name} with basic inputs."""',
                f"    # TODO: replace None with real test data",
                f"    result = {fn.name}({args_placeholder})",
                f"    assert result is not None\n\n",
            ]
        return "\n".join(lines)

    def _output_path(self) -> Path:
        """Return the output file path for the generated tests."""
        module_name = Path(self.module_path).stem
        return Path(self.output_dir) / f"test_{module_name}_generated.py"

    # ── public API ─────────────────────────────────────────────────────────

    def preview(self) -> str:
        """
        Analyze the module and return generated test code without writing to disk.

        Returns:
            str: Generated pytest test code.
        """
        self._analyze()
        prompt = self._build_prompt()
        return self._call_llm(prompt)

    def generate(self) -> Path:
        """
        Analyze the module, generate tests via LLM, and write the test file.

        Returns:
            Path: Path to the written test file.
        """
        print(f"\n🔍  Analyzing: {self.module_path}")
        self._analyze()
        print(f"   Found {len(self._functions)} public function(s): "
              f"{[f.name for f in self._functions]}")

        print("🤖  Generating tests with LLM ...")
        prompt = self._build_prompt()
        code = self._call_llm(prompt)

        # Strip markdown fences if the LLM wrapped the code
        if code.strip().startswith("```"):
            lines = code.strip().splitlines()
            code = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        out = self._output_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(code, encoding="utf-8")
        print(f"✅  Test file written: {out}")
        return out


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    """Command-line interface for the test generator agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virtual Financial Advisor – Test Case Generator Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python -m src.agent.test_generator_agent --module src/trend_detection.py
          python -m src.agent.test_generator_agent --module src/scenario_simulation.py --output tests/
          python -m src.agent.test_generator_agent --all
          python -m src.agent.test_generator_agent --list
        """),
    )
    parser.add_argument(
        "--module", "-m",
        help="Path to source module to generate tests for.",
    )
    parser.add_argument(
        "--output", "-o",
        default="tests",
        help="Output directory for generated test files (default: tests/).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate tests for all registered modules.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all registered modules available for test generation.",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Print generated code to stdout without writing to disk.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nRegistered modules:")
        for name, path in _MODULE_REGISTRY.items():
            status = "✓" if Path(path).exists() else "✗ (not found)"
            print(f"  {status}  {name:25s} → {path}")
        return

    if args.all:
        generated = []
        for name, path in _MODULE_REGISTRY.items():
            if not Path(path).exists():
                print(f"  ⚠  Skipping {name}: file not found at {path}")
                continue
            agent = TestGeneratorAgent(path, args.output)
            if args.preview:
                print(f"\n{'='*60}")
                print(f"# Preview: {path}")
                print('='*60)
                print(agent.preview())
            else:
                agent.generate()
            generated.append(path)
        print(f"\n🎉  Done. Generated tests for {len(generated)} module(s).")
        return

    if args.module:
        agent = TestGeneratorAgent(args.module, args.output)
        if args.preview:
            print(agent.preview())
        else:
            agent.generate()
        return

    parser.print_help()


if __name__ == "__main__":
    _cli()



