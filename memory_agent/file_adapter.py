from __future__ import annotations

import ast
import copy
import io
import json
import keyword
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import FILE_OPERATION_PREVIEW_CHAR_LIMIT

ALLOWED_FILE_OPERATIONS = {
    "read_text",
    "write_text",
    "append_text",
    "replace_text",
    "replace_python_function",
    "replace_python_class",
    "insert_python_before_symbol",
    "insert_python_after_symbol",
    "delete_python_symbol",
    "rename_python_identifier",
    "rename_python_method",
    "add_python_import",
    "remove_python_import",
    "add_python_function_parameter",
    "add_python_method_parameter",
    "rename_python_export_across_imports",
    "move_python_export_to_module",
}

PYTHON_REFACTOR_IGNORED_NAMES = {
    ".agent",
    ".eval_tmp",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".test_tmp",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
}


@dataclass(slots=True)
class FileOperationResult:
    status: str
    operation: str
    path: str
    changed: bool = False
    bytes_written: int | None = None
    match_count: int | None = None
    preview: str = ""
    reason: str = ""
    changed_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "operation": self.operation,
            "path": self.path,
            "changed": self.changed,
            "bytes_written": self.bytes_written,
            "match_count": self.match_count,
            "preview": self.preview,
            "reason": self.reason,
            "changed_paths": self.changed_paths,
        }


class WorkspaceFileAdapter:
    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        preview_char_limit: int = FILE_OPERATION_PREVIEW_CHAR_LIMIT,
    ):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.preview_char_limit = preview_char_limit

    def execute(
        self,
        operation: str,
        path: str,
        *,
        text: str | None = None,
        find_text: str | None = None,
        replace_all: bool = False,
        symbol_name: str | None = None,
        cwd: str | None = None,
    ) -> FileOperationResult:
        clean_operation = str(operation or "").strip()
        if clean_operation not in ALLOWED_FILE_OPERATIONS:
            return FileOperationResult(
                status="blocked",
                operation=clean_operation,
                path=str(path or ""),
                reason="operation_not_allowed",
            )

        resolved_path = self._resolve_path(path, cwd=cwd)
        if resolved_path is None:
            return FileOperationResult(
                status="blocked",
                operation=clean_operation,
                path=str(path or ""),
                reason="path_outside_workspace",
            )

        if clean_operation == "read_text":
            return self._read_text(resolved_path, clean_operation)
        if clean_operation == "write_text":
            return self._write_text(resolved_path, clean_operation, text=text)
        if clean_operation == "append_text":
            return self._append_text(resolved_path, clean_operation, text=text)
        if clean_operation in {"replace_python_function", "replace_python_class"}:
            return self._replace_python_symbol(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        if clean_operation in {"insert_python_before_symbol", "insert_python_after_symbol"}:
            return self._insert_python_relative_to_symbol(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        if clean_operation == "delete_python_symbol":
            return self._delete_python_symbol(
                resolved_path,
                clean_operation,
                symbol_name=symbol_name,
            )
        if clean_operation == "rename_python_identifier":
            return self._rename_python_identifier(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        if clean_operation == "rename_python_method":
            return self._rename_python_method(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        if clean_operation == "add_python_import":
            return self._add_python_import(
                resolved_path,
                clean_operation,
                text=text,
            )
        if clean_operation == "remove_python_import":
            return self._remove_python_import(
                resolved_path,
                clean_operation,
                text=text,
            )
        if clean_operation in {"add_python_function_parameter", "add_python_method_parameter"}:
            return self._add_python_parameter(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        if clean_operation == "rename_python_export_across_imports":
            return self._rename_python_export_across_imports(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        if clean_operation == "move_python_export_to_module":
            return self._move_python_export_to_module(
                resolved_path,
                clean_operation,
                text=text,
                symbol_name=symbol_name,
            )
        return self._replace_text(
            resolved_path,
            clean_operation,
            text=text,
            find_text=find_text,
            replace_all=replace_all,
        )

    def _read_text(self, path: Path, operation: str) -> FileOperationResult:
        contents, error = self._read_existing_text(path)
        if error is not None:
            return FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                reason=error,
            )
        assert contents is not None
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=False,
            bytes_written=0,
            preview=self._trim_preview(contents),
            reason="ok",
        )

    def _write_text(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> FileOperationResult:
        if text is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            return FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                reason=f"oserror:{exc}",
            )
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(text.encode("utf-8")),
            preview=self._trim_preview(text),
            reason="ok",
        )

    def _append_text(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> FileOperationResult:
        if text is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            return FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                reason=f"oserror:{exc}",
            )
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(text.encode("utf-8")),
            preview=self._trim_preview(text),
            reason="ok",
        )

    def _replace_text(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        find_text: str | None,
        replace_all: bool,
    ) -> FileOperationResult:
        if text is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        if not str(find_text or ""):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_find_text",
            )
        contents, error = self._read_existing_text(path)
        if error is not None:
            return FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                reason=error,
            )
        assert contents is not None
        match_count = contents.count(find_text)
        if match_count == 0:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="find_text_not_found",
            )
        if match_count > 1 and not replace_all:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                match_count=match_count,
                reason="multiple_matches_requires_replace_all",
            )
        updated = contents.replace(find_text, text) if replace_all else contents.replace(find_text, text, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                match_count=match_count,
                reason=f"oserror:{exc}",
            )
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=match_count,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _replace_python_symbol(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        if text is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        symbol = self._locate_python_symbol(
            python_source["tree"],
            clean_symbol_name,
            symbol_kind="function" if operation == "replace_python_function" else "class",
        )
        failed_symbol_lookup = self._failed_symbol_lookup(path, operation, symbol)
        if failed_symbol_lookup is not None:
            return failed_symbol_lookup
        assert isinstance(symbol, tuple)
        start_line, end_line = symbol
        lines = str(python_source["text"]).splitlines(keepends=True)
        existing_segment = "".join(lines[start_line - 1 : end_line])
        replacement = text
        if existing_segment.endswith(("\r\n", "\n")) and not replacement.endswith(("\r\n", "\n")):
            replacement += "\r\n" if "\r\n" in existing_segment else "\n"
        updated = "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=1,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _insert_python_relative_to_symbol(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        if text is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        symbol = self._locate_python_symbol(
            python_source["tree"],
            clean_symbol_name,
            symbol_kind="any",
        )
        failed_symbol_lookup = self._failed_symbol_lookup(path, operation, symbol)
        if failed_symbol_lookup is not None:
            return failed_symbol_lookup
        assert isinstance(symbol, tuple)
        start_line, end_line = symbol
        lines = str(python_source["text"]).splitlines(keepends=True)
        insertion = text
        if insertion and not insertion.endswith(("\r\n", "\n")):
            insertion += "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
        insert_index = start_line - 1 if operation == "insert_python_before_symbol" else end_line
        updated = "".join(lines[:insert_index]) + insertion + "".join(lines[insert_index:])
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=1,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _delete_python_symbol(
        self,
        path: Path,
        operation: str,
        *,
        symbol_name: str | None,
    ) -> FileOperationResult:
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        symbol = self._locate_python_symbol(
            python_source["tree"],
            clean_symbol_name,
            symbol_kind="any",
        )
        failed_symbol_lookup = self._failed_symbol_lookup(path, operation, symbol)
        if failed_symbol_lookup is not None:
            return failed_symbol_lookup
        assert isinstance(symbol, tuple)
        start_line, end_line = symbol
        lines = str(python_source["text"]).splitlines(keepends=True)
        updated = "".join(lines[: start_line - 1]) + "".join(lines[end_line:])
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=1,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _rename_python_identifier(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        clean_new_name = str(text or "").strip()
        if not clean_new_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        if "." in clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="dotted_symbol_names_not_supported_for_rename",
            )
        if not clean_symbol_name.isidentifier() or keyword.iskeyword(clean_symbol_name):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_symbol_name",
            )
        if not clean_new_name.isidentifier() or keyword.iskeyword(clean_new_name):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_new_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        renamed_count = 0
        updated_tokens: list[tokenize.TokenInfo] = []
        for token in tokenize.generate_tokens(io.StringIO(str(python_source["text"])).readline):
            if token.type == tokenize.NAME and token.string == clean_symbol_name:
                renamed_count += 1
                updated_tokens.append(token._replace(string=clean_new_name))
            else:
                updated_tokens.append(token)
        if renamed_count == 0:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="identifier_not_found",
            )
        updated = tokenize.untokenize(updated_tokens)
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=renamed_count,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _rename_python_method(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        clean_new_name = str(text or "").strip()
        if not clean_new_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        if not clean_new_name.isidentifier() or keyword.iskeyword(clean_new_name):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_new_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        method_match = self._locate_python_method(
            python_source["tree"],
            clean_symbol_name,
        )
        if method_match is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="method_not_found",
            )
        if method_match == "ambiguous":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="multiple_methods_found",
            )
        dotted_name, start_line, end_line = method_match
        old_name = dotted_name.rsplit(".", 1)[-1]
        lines = str(python_source["text"]).splitlines(keepends=True)
        method_lines = list(lines[start_line - 1 : end_line])
        header_pattern = re.compile(
            rf"^(\s*(?:async\s+)?def\s+){re.escape(old_name)}(\b)"
        )
        replaced_header = False
        for index, line in enumerate(method_lines):
            updated_line, count = header_pattern.subn(
                rf"\1{clean_new_name}\2",
                line,
                count=1,
            )
            if count:
                method_lines[index] = updated_line
                replaced_header = True
                break
        if not replaced_header:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="method_definition_not_found",
            )
        updated = (
            "".join(lines[: start_line - 1])
            + "".join(method_lines)
            + "".join(lines[end_line:])
        )
        rename_count = 1
        updated_tokens: list[tokenize.TokenInfo] = []
        prior_significant = ""
        for token in tokenize.generate_tokens(io.StringIO(updated).readline):
            if (
                token.type == tokenize.NAME
                and token.string == old_name
                and prior_significant == "."
            ):
                rename_count += 1
                updated_tokens.append(token._replace(string=clean_new_name))
            else:
                updated_tokens.append(token)
            if token.type not in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.ENDMARKER,
            }:
                prior_significant = token.string
        updated = tokenize.untokenize(updated_tokens)
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=rename_count,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _add_python_import(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> FileOperationResult:
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        import_statement, failure = self._parse_python_import_statement(
            path,
            operation,
            text=text,
        )
        if failure is not None:
            return failure
        assert import_statement is not None
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        import_nodes = self._top_level_import_nodes(python_source["tree"])
        normalized_statement = ast.dump(import_statement, include_attributes=False)
        if any(
            ast.dump(node, include_attributes=False) == normalized_statement
            for node in import_nodes
        ):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="import_already_present",
            )
        lines = str(python_source["text"]).splitlines(keepends=True)
        insertion = str(text or "").rstrip("\r\n")
        newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
        insertion += newline
        insert_index = self._python_import_insert_index(python_source["tree"], lines)
        if insert_index < len(lines) and lines[insert_index].strip():
            insertion += newline
        updated = "".join(lines[:insert_index]) + insertion + "".join(lines[insert_index:])
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=1,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _remove_python_import(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> FileOperationResult:
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        import_statement, failure = self._parse_python_import_statement(
            path,
            operation,
            text=text,
        )
        if failure is not None:
            return failure
        assert import_statement is not None
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        normalized_statement = ast.dump(import_statement, include_attributes=False)
        matching_nodes = [
            node
            for node in self._top_level_import_nodes(python_source["tree"])
            if ast.dump(node, include_attributes=False) == normalized_statement
        ]
        if not matching_nodes:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="import_not_found",
            )
        lines = str(python_source["text"]).splitlines(keepends=True)
        skip_lines = {
            line_number
            for node in matching_nodes
            for line_number in range(int(node.lineno), int(node.end_lineno) + 1)
        }
        updated = "".join(
            line
            for index, line in enumerate(lines, start=1)
            if index not in skip_lines
        )
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=len(matching_nodes),
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _add_python_parameter(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        if operation == "add_python_function_parameter" and "." in clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="dotted_function_symbols_not_supported",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        spec, spec_failure = self._parse_python_parameter_refactor_spec(
            path,
            operation,
            text=text,
        )
        if spec_failure is not None:
            return spec_failure
        assert spec is not None
        python_source, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert python_source is not None
        callable_kind = "method" if operation == "add_python_method_parameter" else "function"
        target_node = self._locate_python_callable_node(
            python_source["tree"],
            clean_symbol_name,
            callable_kind=callable_kind,
        )
        if target_node is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"{callable_kind}_not_found",
            )
        if target_node == "ambiguous":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"multiple_{callable_kind}s_found",
            )
        assert isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef))
        parameter_names = self._callable_parameter_names(target_node)
        if spec["parameter_name"] in parameter_names:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="parameter_already_present",
            )
        new_args = self._args_with_added_parameter(
            target_node.args,
            parameter_name=str(spec["parameter_name"]),
            default_value=str(spec["default_value"]) if spec["default_value"] is not None else None,
        )
        header = self._render_callable_header(target_node, new_args)
        updated = self._replace_callable_header(
            str(python_source["text"]),
            target_node,
            header,
        )
        if spec["call_argument"] is not None:
            try:
                updated_tree = ast.parse(updated)
            except SyntaxError as exc:
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(path),
                    reason=f"updated_python_parse_error:{exc.lineno}:{exc.msg}",
                )
            updated, rewrite_failure = self._rewrite_parameter_call_sites(
                updated,
                updated_tree,
                operation=operation,
                path=path,
                target_symbol=clean_symbol_name,
                parameter_name=str(spec["parameter_name"]),
                call_argument=str(spec["call_argument"]),
                callable_kind=callable_kind,
                allow_partial=spec["default_value"] is not None,
            )
            if rewrite_failure is not None:
                return rewrite_failure
        write_failure = self._validate_and_write_python_update(
            path,
            operation,
            updated,
            had_bom=bool(python_source["had_bom"]),
        )
        if write_failure is not None:
            return write_failure
        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=len(updated.encode("utf-8")),
            match_count=1,
            preview=self._trim_preview(updated),
            reason="ok",
        )

    def _rename_python_export_across_imports(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        clean_symbol_name = str(symbol_name or "").strip()
        clean_new_name = str(text or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        if not clean_new_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        if "." in clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="dotted_export_symbols_not_supported",
            )
        if not clean_symbol_name.isidentifier() or keyword.iskeyword(clean_symbol_name):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_symbol_name",
            )
        if not clean_new_name.isidentifier() or keyword.iskeyword(clean_new_name):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_new_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        module_names = self._python_module_name_candidates_for_path(path)
        if not module_names:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="module_path_not_importable",
            )
        source_python, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert source_python is not None
        export_node = self._locate_top_level_export_node(
            source_python["tree"],
            clean_symbol_name,
        )
        if export_node is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="export_not_found",
            )
        if export_node == "ambiguous":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="multiple_exports_found",
            )
        assert isinstance(export_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if self._has_unsafe_name_bindings(
            source_python["tree"],
            clean_symbol_name,
            allowed_def_nodes={export_node},
        ):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="source_binding_conflict",
            )
        changed_paths: list[str] = []
        pending_updates: list[tuple[Path, str, bool]] = []
        source_replacements = self._source_export_rename_replacements(
            str(source_python["text"]),
            source_python["tree"],
            export_node,
            old_name=clean_symbol_name,
            new_name=clean_new_name,
        )
        if source_replacements is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="definition_name_not_found",
            )
        updated_source = self._apply_text_replacements(
            str(source_python["text"]),
            source_replacements,
        )
        validation_failure = self._validate_python_update_text(
            path,
            operation,
            updated_source,
        )
        if validation_failure is not None:
            return validation_failure
        pending_updates.append((path, updated_source, bool(source_python["had_bom"])))
        changed_paths.append(str(path))

        for consumer_path in self._iter_workspace_python_files():
            if consumer_path == path:
                continue
            consumer_python, failure = self._load_python_source(consumer_path, operation)
            if failure is not None:
                return failure
            assert consumer_python is not None
            consumer_updated, consumer_reason = self._rewrite_consumer_imports_for_export_rename(
                str(consumer_python["text"]),
                consumer_python["tree"],
                module_names=module_names,
                old_name=clean_symbol_name,
                new_name=clean_new_name,
            )
            if consumer_reason is not None:
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(consumer_path),
                    reason=consumer_reason,
                    changed_paths=list(changed_paths),
                )
            if consumer_updated is None:
                continue
            validation_failure = self._validate_python_update_text(
                consumer_path,
                operation,
                consumer_updated,
            )
            if validation_failure is not None:
                return validation_failure
            pending_updates.append(
                (consumer_path, consumer_updated, bool(consumer_python["had_bom"]))
            )
            changed_paths.append(str(consumer_path))

        for pending_path, pending_text, had_bom in pending_updates:
            write_failure = self._write_python_update_text(
                pending_path,
                operation,
                pending_text,
                had_bom=had_bom,
            )
            if write_failure is not None:
                return write_failure

        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=sum(len(item.encode("utf-8")) for _, item, _ in pending_updates),
            match_count=len(changed_paths),
            preview=self._trim_preview(updated_source),
            reason="ok",
            changed_paths=changed_paths,
        )

    def _move_python_export_to_module(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
        symbol_name: str | None,
    ) -> FileOperationResult:
        clean_symbol_name = str(symbol_name or "").strip()
        if not clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_symbol_name",
            )
        if "." in clean_symbol_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="dotted_export_symbols_not_supported",
            )
        if not clean_symbol_name.isidentifier() or keyword.iskeyword(clean_symbol_name):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_symbol_name",
            )
        if path.suffix.lower() != ".py":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="not_python_file",
            )
        move_spec, spec_failure = self._parse_python_export_move_spec(
            path,
            operation,
            text=text,
        )
        if spec_failure is not None:
            return spec_failure
        assert move_spec is not None
        destination_path = move_spec["destination_path"]
        assert isinstance(destination_path, Path)
        reexport_from_source = bool(move_spec["reexport_from_source"])
        if destination_path == path:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="destination_matches_source",
            )
        source_module_names = self._python_module_name_candidates_for_path(path)
        destination_module_names = self._python_module_name_candidates_for_path(destination_path)
        source_module_name = self._preferred_module_name(source_module_names)
        destination_module_name = self._preferred_module_name(destination_module_names)
        if not source_module_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="module_path_not_importable",
            )
        if not destination_module_name:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(destination_path),
                reason="destination_module_not_importable",
            )

        source_python, failure = self._load_python_source(path, operation)
        if failure is not None:
            return failure
        assert source_python is not None
        export_node = self._locate_top_level_export_node(
            source_python["tree"],
            clean_symbol_name,
        )
        if export_node is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="export_not_found",
            )
        if export_node == "ambiguous":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="multiple_exports_found",
            )
        assert isinstance(export_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if self._has_unsafe_name_bindings(
            source_python["tree"],
            clean_symbol_name,
            allowed_def_nodes={export_node},
            allowed_import_from_modules=destination_module_names if reexport_from_source else None,
        ):
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="source_binding_conflict",
            )

        required_imports, dependency_conflicts = self._movable_export_import_dependencies(
            str(source_python["text"]),
            source_python["tree"],
            export_node,
            export_name=clean_symbol_name,
        )
        if dependency_conflicts:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="source_local_dependencies_not_supported",
                preview=", ".join(sorted(dependency_conflicts)),
            )

        symbol_lines = self._locate_python_symbol(
            source_python["tree"],
            clean_symbol_name,
            symbol_kind="any",
        )
        symbol_failure = self._failed_symbol_lookup(path, operation, symbol_lines)
        if symbol_failure is not None:
            return symbol_failure
        assert isinstance(symbol_lines, tuple)
        start_line, end_line = symbol_lines
        source_lines = str(source_python["text"]).splitlines(keepends=True)
        export_block = "".join(source_lines[start_line - 1 : end_line])
        updated_source = "".join(source_lines[: start_line - 1]) + "".join(source_lines[end_line:])
        if reexport_from_source:
            updated_source, source_import_failure = self._insert_python_import_statement_text(
                updated_source,
                f"from {destination_module_name} import {clean_symbol_name}",
            )
            if source_import_failure is not None:
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(path),
                    reason=source_import_failure,
                )
        validation_failure = self._validate_python_update_text(path, operation, updated_source)
        if validation_failure is not None:
            return validation_failure

        destination_python: dict[str, Any] | None
        destination_text = ""
        destination_had_bom = False
        if destination_path.exists():
            destination_python, failure = self._load_python_source(destination_path, operation)
            if failure is not None:
                return failure
            assert destination_python is not None
            destination_text = str(destination_python["text"])
            destination_had_bom = bool(destination_python["had_bom"])
            existing_destination_export = self._locate_top_level_export_node(
                destination_python["tree"],
                clean_symbol_name,
            )
            if existing_destination_export is not None:
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(destination_path),
                    reason="destination_export_conflict",
                )
            if self._has_unsafe_name_bindings(destination_python["tree"], clean_symbol_name):
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(destination_path),
                    reason="destination_binding_conflict",
                )
            if reexport_from_source and self._tree_imports_any_module(
                destination_python["tree"],
                source_module_names,
            ):
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(destination_path),
                    reason="destination_imports_source_module",
                )
        else:
            destination_python = None

        updated_destination = destination_text
        if required_imports:
            updated_destination, destination_import_failure = self._ensure_python_import_statements(
                updated_destination,
                required_imports,
            )
            if destination_import_failure is not None:
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(destination_path),
                    reason=destination_import_failure,
                )
        updated_destination = self._append_python_block(updated_destination, export_block)
        validation_failure = self._validate_python_update_text(
            destination_path,
            operation,
            updated_destination,
        )
        if validation_failure is not None:
            return validation_failure

        pending_updates: list[tuple[Path, str, bool]] = [
            (path, updated_source, bool(source_python["had_bom"])),
            (destination_path, updated_destination, destination_had_bom),
        ]
        changed_paths = [str(path), str(destination_path)]

        for consumer_path in self._iter_workspace_python_files():
            if consumer_path in {path, destination_path}:
                continue
            consumer_python, failure = self._load_python_source(consumer_path, operation)
            if failure is not None:
                return failure
            assert consumer_python is not None
            consumer_updated, consumer_reason = self._rewrite_consumer_imports_for_export_move(
                str(consumer_python["text"]),
                consumer_python["tree"],
                source_module_names=source_module_names,
                destination_module_name=destination_module_name,
                moved_name=clean_symbol_name,
            )
            if consumer_reason is not None:
                return FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(consumer_path),
                    reason=consumer_reason,
                    changed_paths=list(changed_paths),
                )
            if consumer_updated is None:
                continue
            validation_failure = self._validate_python_update_text(
                consumer_path,
                operation,
                consumer_updated,
            )
            if validation_failure is not None:
                return validation_failure
            pending_updates.append(
                (consumer_path, consumer_updated, bool(consumer_python["had_bom"]))
            )
            changed_paths.append(str(consumer_path))

        for pending_path, pending_text, had_bom in pending_updates:
            write_failure = self._write_python_update_text(
                pending_path,
                operation,
                pending_text,
                had_bom=had_bom,
            )
            if write_failure is not None:
                return write_failure

        return FileOperationResult(
            status="success",
            operation=operation,
            path=str(path),
            changed=True,
            bytes_written=sum(len(item.encode("utf-8")) for _, item, _ in pending_updates),
            match_count=len(changed_paths),
            preview=self._trim_preview(updated_destination),
            reason="ok",
            changed_paths=changed_paths,
        )

    def _source_export_rename_replacements(
        self,
        text: str,
        tree: ast.AST,
        export_node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        *,
        old_name: str,
        new_name: str,
    ) -> list[tuple[int, int, str]] | None:
        line_offsets = self._line_offsets(text)
        name_span = self._definition_name_span(text, export_node)
        if name_span is None:
            return None
        replacements: list[tuple[int, int, str]] = [(*name_span, new_name)]
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id == old_name
                and isinstance(node.ctx, ast.Load)
            ):
                replacements.append((*self._node_span(node, line_offsets), new_name))
        return replacements

    def _rewrite_consumer_imports_for_export_rename(
        self,
        text: str,
        tree: ast.AST,
        *,
        module_names: set[str],
        old_name: str,
        new_name: str,
    ) -> tuple[str | None, str | None]:
        replacements: list[tuple[int, int, str]] = []
        direct_binding_requested = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or node.module not in module_names:
                continue
            for alias in node.names:
                if alias.name != old_name:
                    continue
                span = self._import_from_name_span(text, node, old_name)
                if span is None:
                    return None, "consumer_import_span_not_found"
                replacements.append((*span, new_name))
                if alias.asname is None:
                    direct_binding_requested = True
        if direct_binding_requested:
            if self._has_unsafe_name_bindings(
                tree,
                old_name,
                allowed_import_from_modules=module_names,
            ):
                return None, "consumer_binding_conflict"
            line_offsets = self._line_offsets(text)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Name)
                    and node.id == old_name
                    and isinstance(node.ctx, ast.Load)
                ):
                    replacements.append((*self._node_span(node, line_offsets), new_name))
        import_prefixes = self._module_import_prefixes(tree, module_names)
        if import_prefixes:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or node.attr != old_name:
                    continue
                chain = self._attribute_chain_names(node.value)
                if chain is None or tuple(chain) not in import_prefixes:
                    continue
                span = self._attribute_name_span(text, node)
                if span is None:
                    return None, "consumer_attribute_span_not_found"
                replacements.append((*span, new_name))
        if not replacements:
            return None, None
        updated = self._apply_text_replacements(text, replacements)
        if updated == text:
            return None, None
        return updated, None

    def _rewrite_consumer_imports_for_export_move(
        self,
        text: str,
        tree: ast.AST,
        *,
        source_module_names: set[str],
        destination_module_name: str,
        moved_name: str,
    ) -> tuple[str | None, str | None]:
        replacements: list[tuple[int, int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or node.module not in source_module_names:
                continue
            moved_aliases = [alias for alias in node.names if alias.name == moved_name]
            if not moved_aliases:
                continue
            if len(node.names) != len(moved_aliases):
                return None, "consumer_import_split_required"
            module_span = self._import_from_module_span(text, node)
            if module_span is None:
                return None, "consumer_import_module_span_not_found"
            replacements.append((*module_span, destination_module_name))
        if not replacements:
            return None, None
        updated = self._apply_text_replacements(text, replacements)
        if updated == text:
            return None, None
        return updated, None

    def _iter_workspace_python_files(self) -> list[Path]:
        return sorted(
            candidate
            for candidate in self.workspace_root.rglob("*.py")
            if candidate.is_file()
            and not any(
                part in PYTHON_REFACTOR_IGNORED_NAMES
                for part in candidate.relative_to(self.workspace_root).parts
            )
        )

    def _package_relative_parts(self, directory: Path) -> list[str]:
        parts: list[str] = []
        current = directory.resolve()
        while current != self.workspace_root:
            if not (current / "__init__.py").exists():
                break
            parts.insert(0, current.name)
            current = current.parent
        return parts

    def _python_module_name_candidates_for_path(self, path: Path) -> set[str]:
        try:
            relative_path = path.resolve().relative_to(self.workspace_root)
        except ValueError:
            return set()
        parts = list(relative_path.parts)
        if not parts:
            return set()
        if parts[-1] == "__init__.py":
            module_parts = parts[:-1]
        elif path.suffix.lower() == ".py":
            module_parts = [*parts[:-1], path.stem]
        else:
            return set()
        candidates: set[str] = set()
        if module_parts and all(part.isidentifier() for part in module_parts):
            candidates.add(".".join(module_parts))
        if path.name == "__init__.py":
            package_parts = self._package_relative_parts(path.parent)
        else:
            package_parts = [*self._package_relative_parts(path.parent), path.stem]
        if package_parts and all(part.isidentifier() for part in package_parts):
            candidates.add(".".join(package_parts))
        return {candidate for candidate in candidates if candidate}

    def _preferred_module_name(self, module_names: set[str]) -> str | None:
        candidates = [candidate for candidate in module_names if candidate]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (candidate.count("."), len(candidate), candidate),
        )

    def _read_existing_text(self, path: Path) -> tuple[str | None, str | None]:
        if not path.exists():
            return None, "file_not_found"
        if not path.is_file():
            return None, "path_is_not_file"
        try:
            return path.read_text(encoding="utf-8"), None
        except UnicodeDecodeError:
            return None, "file_not_utf8_text"
        except OSError as exc:
            return None, f"oserror:{exc}"

    def _resolve_path(self, path: str, *, cwd: str | None) -> Path | None:
        raw_path = str(path or "").strip()
        if not raw_path:
            return None
        base_dir = self.workspace_root
        if cwd:
            resolved_cwd = self._resolve_within_workspace(cwd, base=self.workspace_root)
            if resolved_cwd is None:
                return None
            base_dir = resolved_cwd
        return self._resolve_within_workspace(raw_path, base=base_dir)

    def _resolve_within_workspace(self, raw_path: str, *, base: Path) -> Path | None:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            return None
        return resolved

    def _trim_preview(self, text: str) -> str:
        stripped = text.strip()
        if len(stripped) <= self.preview_char_limit:
            return stripped
        return stripped[: self.preview_char_limit - 3] + "..."

    def _parse_python_parameter_refactor_spec(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> tuple[dict[str, str | None] | None, FileOperationResult | None]:
        clean_text = str(text or "").strip()
        if not clean_text:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        try:
            payload = json.loads(clean_text)
        except json.JSONDecodeError as exc:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"invalid_parameter_refactor_spec:{exc.msg}",
            )
        if not isinstance(payload, dict):
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_parameter_refactor_spec",
            )
        parameter_name = str(payload.get("parameter_name") or "").strip()
        if not parameter_name:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_parameter_name",
            )
        if not parameter_name.isidentifier() or keyword.iskeyword(parameter_name):
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_parameter_name",
            )
        default_value = payload.get("default_value")
        call_argument = payload.get("call_argument")
        clean_default = None if default_value is None else str(default_value).strip()
        clean_call = None if call_argument is None else str(call_argument).strip()
        if not clean_default and not clean_call:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="parameter_refactor_requires_default_or_call_argument",
            )
        for field_name, field_value in (
            ("default_value", clean_default),
            ("call_argument", clean_call),
        ):
            if field_value:
                try:
                    ast.parse(field_value, mode="eval")
                except SyntaxError as exc:
                    return None, FileOperationResult(
                        status="blocked",
                        operation=operation,
                        path=str(path),
                        reason=f"invalid_{field_name}:{exc.msg}",
                    )
        return {
            "parameter_name": parameter_name,
            "default_value": clean_default or None,
            "call_argument": clean_call or None,
        }, None

    def _parse_python_export_move_spec(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> tuple[dict[str, Any] | None, FileOperationResult | None]:
        clean_text = str(text or "").strip()
        if not clean_text:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        try:
            payload = json.loads(clean_text)
        except json.JSONDecodeError as exc:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"invalid_export_move_spec:{exc.msg}",
            )
        if not isinstance(payload, dict):
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_export_move_spec",
            )
        destination_raw = str(payload.get("destination_path") or "").strip()
        if not destination_raw:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_destination_path",
            )
        destination_path = self._resolve_within_workspace(destination_raw, base=self.workspace_root)
        if destination_path is None:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="destination_path_outside_workspace",
            )
        if destination_path.suffix.lower() != ".py":
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(destination_path),
                reason="destination_not_python_file",
            )
        reexport_from_source = payload.get("reexport_from_source", True)
        if not isinstance(reexport_from_source, bool):
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_reexport_flag",
            )
        return {
            "destination_path": destination_path,
            "reexport_from_source": reexport_from_source,
        }, None

    def _parse_python_import_statement(
        self,
        path: Path,
        operation: str,
        *,
        text: str | None,
    ) -> tuple[ast.Import | ast.ImportFrom | None, FileOperationResult | None]:
        clean_text = str(text or "").strip()
        if not clean_text:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="missing_text",
            )
        try:
            import_tree = ast.parse(clean_text)
        except SyntaxError as exc:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"invalid_python_import_statement:{exc.lineno}:{exc.msg}",
            )
        if len(import_tree.body) != 1 or not isinstance(
            import_tree.body[0],
            (ast.Import, ast.ImportFrom),
        ):
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="invalid_python_import_statement",
            )
        return import_tree.body[0], None

    def _insert_python_import_statement_text(
        self,
        text: str,
        import_statement_text: str,
    ) -> tuple[str | None, str | None]:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return None, f"python_parse_error:{exc.lineno}:{exc.msg}"
        try:
            import_tree = ast.parse(import_statement_text)
        except SyntaxError as exc:
            return None, f"invalid_python_import_statement:{exc.lineno}:{exc.msg}"
        if len(import_tree.body) != 1 or not isinstance(
            import_tree.body[0],
            (ast.Import, ast.ImportFrom),
        ):
            return None, "invalid_python_import_statement"
        import_statement = import_tree.body[0]
        normalized_statement = ast.dump(import_statement, include_attributes=False)
        if any(
            ast.dump(node, include_attributes=False) == normalized_statement
            for node in self._top_level_import_nodes(tree)
        ):
            return text, None
        lines = text.splitlines(keepends=True)
        insertion = import_statement_text.rstrip("\r\n")
        newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
        insertion += newline
        insert_index = self._python_import_insert_index(tree, lines)
        if insert_index < len(lines) and lines[insert_index].strip():
            insertion += newline
        updated = "".join(lines[:insert_index]) + insertion + "".join(lines[insert_index:])
        return updated, None

    def _ensure_python_import_statements(
        self,
        text: str,
        import_statements: list[str],
    ) -> tuple[str | None, str | None]:
        updated = text
        for statement in import_statements:
            updated, failure = self._insert_python_import_statement_text(updated, statement)
            if failure is not None:
                return None, failure
            assert updated is not None
        return updated, None

    def _callable_parameter_names(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> set[str]:
        names = {argument.arg for argument in node.args.posonlyargs}
        names.update(argument.arg for argument in node.args.args)
        names.update(argument.arg for argument in node.args.kwonlyargs)
        if node.args.vararg is not None:
            names.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            names.add(node.args.kwarg.arg)
        return names

    def _args_with_added_parameter(
        self,
        args: ast.arguments,
        *,
        parameter_name: str,
        default_value: str | None,
    ) -> ast.arguments:
        updated_args = copy.deepcopy(args)
        parameter = ast.arg(arg=parameter_name, annotation=None)
        default_node = (
            ast.parse(default_value, mode="eval").body
            if default_value is not None
            else None
        )
        if updated_args.vararg is not None or updated_args.kwonlyargs:
            updated_args.kwonlyargs.append(parameter)
            updated_args.kw_defaults.append(default_node)
            return updated_args
        updated_args.args.append(parameter)
        if default_node is not None:
            updated_args.defaults.append(default_node)
        return updated_args

    def _render_callable_header(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        new_args: ast.arguments,
    ) -> str:
        cloned = copy.deepcopy(node)
        cloned.args = new_args
        cloned.body = [ast.Pass()]
        rendered = ast.unparse(cloned)
        marker = "\n    pass"
        if marker in rendered:
            return rendered.split(marker, 1)[0]
        return rendered.splitlines()[0]

    def _replace_callable_header(
        self,
        text: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        new_header: str,
    ) -> str:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        line_offsets = self._line_offsets(text)
        header_start: int | None = None
        header_end: int | None = None
        for index, token in enumerate(tokens):
            if token.start[0] != int(node.lineno):
                continue
            if token.string != "def":
                continue
            start_token = token
            if (
                index > 0
                and tokens[index - 1].start[0] == token.start[0]
                and tokens[index - 1].string == "async"
            ):
                start_token = tokens[index - 1]
            header_start = self._absolute_index(line_offsets, start_token.start)
            paren_depth = 0
            seen_open = False
            for later_token in tokens[index + 1 :]:
                if later_token.string == "(":
                    paren_depth += 1
                    seen_open = True
                elif later_token.string == ")" and seen_open:
                    paren_depth -= 1
                elif later_token.string == ":" and seen_open and paren_depth == 0:
                    header_end = self._absolute_index(line_offsets, later_token.end)
                    break
            break
        if header_start is None or header_end is None:
            return text
        return text[:header_start] + new_header + text[header_end:]

    def _rewrite_parameter_call_sites(
        self,
        text: str,
        tree: ast.AST,
        *,
        operation: str,
        path: Path,
        target_symbol: str,
        parameter_name: str,
        call_argument: str,
        callable_kind: str,
        allow_partial: bool,
    ) -> tuple[str, FileOperationResult | None]:
        insertions: list[tuple[int, str]] = []
        target_leaf = target_symbol.rsplit(".", 1)[-1]
        line_offsets = self._line_offsets(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if callable_kind == "function":
                if not isinstance(node.func, ast.Name) or node.func.id != target_leaf:
                    continue
            else:
                if not isinstance(node.func, ast.Attribute) or node.func.attr != target_leaf:
                    continue
            if any(keyword_item.arg == parameter_name for keyword_item in node.keywords):
                continue
            if int(node.lineno) != int(node.end_lineno):
                if allow_partial:
                    continue
                return text, FileOperationResult(
                    status="blocked",
                    operation=operation,
                    path=str(path),
                    reason="multiline_call_sites_not_supported",
                )
            insertion_index = (
                self._absolute_index(line_offsets, (int(node.end_lineno), int(node.end_col_offset)))
                - 1
            )
            insertion_text = (
                (", " if node.args or node.keywords else "")
                + f"{parameter_name}={call_argument}"
            )
            insertions.append((insertion_index, insertion_text))
        updated = text
        for index, insertion_text in sorted(insertions, key=lambda item: item[0], reverse=True):
            updated = updated[:index] + insertion_text + updated[index:]
        return updated, None

    def _line_offsets(self, text: str) -> list[int]:
        offsets: list[int] = []
        running = 0
        for line in text.splitlines(keepends=True):
            offsets.append(running)
            running += len(line)
        if not offsets:
            offsets.append(0)
        return offsets

    def _absolute_index(
        self,
        line_offsets: list[int],
        position: tuple[int, int],
    ) -> int:
        line_number, column = position
        if line_number - 1 >= len(line_offsets):
            return len(line_offsets) and line_offsets[-1] or 0
        return line_offsets[line_number - 1] + column

    def _apply_text_replacements(
        self,
        text: str,
        replacements: list[tuple[int, int, str]],
    ) -> str:
        deduped: dict[tuple[int, int], str] = {}
        for start, end, replacement in replacements:
            if start >= end:
                continue
            deduped[(start, end)] = replacement
        updated = text
        for (start, end), replacement in sorted(
            deduped.items(),
            key=lambda item: item[0][0],
            reverse=True,
        ):
            updated = updated[:start] + replacement + updated[end:]
        return updated

    def _node_span(
        self,
        node: ast.AST,
        line_offsets: list[int],
    ) -> tuple[int, int]:
        return (
            self._absolute_index(line_offsets, (int(node.lineno), int(node.col_offset))),
            self._absolute_index(
                line_offsets,
                (int(node.end_lineno), int(node.end_col_offset)),
            ),
        )

    def _definition_name_span(
        self,
        text: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> tuple[int, int] | None:
        line_offsets = self._line_offsets(text)
        prior_significant = ""
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.start[0] != int(node.lineno):
                continue
            if token.type == tokenize.NAME and token.string == node.name and prior_significant in {"def", "class"}:
                return (
                    self._absolute_index(line_offsets, token.start),
                    self._absolute_index(line_offsets, token.end),
                )
            if token.type not in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.ENDMARKER,
            }:
                prior_significant = token.string
        return None

    def _import_from_module_span(
        self,
        text: str,
        node: ast.ImportFrom,
    ) -> tuple[int, int] | None:
        line_offsets = self._line_offsets(text)
        node_start = self._absolute_index(line_offsets, (int(node.lineno), int(node.col_offset)))
        node_end = self._absolute_index(
            line_offsets,
            (int(node.end_lineno), int(node.end_col_offset)),
        )
        seen_from = False
        module_start: int | None = None
        module_end: int | None = None
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            start = self._absolute_index(line_offsets, token.start)
            end = self._absolute_index(line_offsets, token.end)
            if end <= node_start or start >= node_end:
                continue
            if token.type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.ENDMARKER,
            }:
                continue
            if not seen_from:
                if token.type == tokenize.NAME and token.string == "from":
                    seen_from = True
                continue
            if token.type == tokenize.NAME and token.string == "import":
                break
            if token.type == tokenize.NAME or (token.type == tokenize.OP and token.string == "."):
                if module_start is None:
                    module_start = start
                module_end = end
        if module_start is None or module_end is None or module_start >= module_end:
            return None
        return module_start, module_end

    def _import_from_name_span(
        self,
        text: str,
        node: ast.ImportFrom,
        target_name: str,
    ) -> tuple[int, int] | None:
        line_offsets = self._line_offsets(text)
        node_start = self._absolute_index(line_offsets, (int(node.lineno), int(node.col_offset)))
        node_end = self._absolute_index(
            line_offsets,
            (int(node.end_lineno), int(node.end_col_offset)),
        )
        seen_import = False
        prior_significant = ""
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            token_start = self._absolute_index(line_offsets, token.start)
            token_end = self._absolute_index(line_offsets, token.end)
            if token_end <= node_start or token_start >= node_end:
                continue
            if token.type in {
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.ENDMARKER,
            }:
                continue
            if token.string == "import":
                seen_import = True
                prior_significant = "import"
                continue
            if (
                seen_import
                and token.type == tokenize.NAME
                and token.string == target_name
                and prior_significant in {"import", ",", "("}
            ):
                return token_start, token_end
            prior_significant = token.string
        return None

    def _attribute_name_span(
        self,
        text: str,
        node: ast.Attribute,
    ) -> tuple[int, int] | None:
        line_offsets = self._line_offsets(text)
        node_start = self._absolute_index(line_offsets, (int(node.lineno), int(node.col_offset)))
        node_end = self._absolute_index(
            line_offsets,
            (int(node.end_lineno), int(node.end_col_offset)),
        )
        name_tokens: list[tuple[int, int]] = []
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            token_start = self._absolute_index(line_offsets, token.start)
            token_end = self._absolute_index(line_offsets, token.end)
            if token_end <= node_start or token_start >= node_end:
                continue
            if token.type == tokenize.NAME:
                name_tokens.append((token_start, token_end))
        return name_tokens[-1] if name_tokens else None

    def _attribute_chain_names(self, node: ast.AST) -> list[str] | None:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            parent = self._attribute_chain_names(node.value)
            if parent is None:
                return None
            return [*parent, node.attr]
        return None

    def _module_import_prefixes(
        self,
        tree: ast.AST,
        module_names: set[str],
    ) -> set[tuple[str, ...]]:
        prefixes: set[tuple[str, ...]] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import):
                continue
            for alias in node.names:
                if alias.name not in module_names:
                    continue
                if alias.asname:
                    prefixes.add((alias.asname,))
                else:
                    prefixes.add(tuple(alias.name.split(".")))
        return prefixes

    def _tree_imports_any_module(self, tree: ast.AST, module_names: set[str]) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_names:
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module in module_names:
                    return True
        return False

    def _validate_python_update_text(
        self,
        path: Path,
        operation: str,
        updated: str,
    ) -> FileOperationResult | None:
        try:
            ast.parse(updated)
        except SyntaxError as exc:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"updated_python_parse_error:{exc.lineno}:{exc.msg}",
            )
        return None

    def _write_python_update_text(
        self,
        path: Path,
        operation: str,
        updated: str,
        *,
        had_bom: bool,
    ) -> FileOperationResult | None:
        if had_bom:
            updated = "\ufeff" + updated
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                reason=f"oserror:{exc}",
            )
        return None

    def _load_python_source(
        self,
        path: Path,
        operation: str,
    ) -> tuple[dict[str, Any] | None, FileOperationResult | None]:
        contents, error = self._read_existing_text(path)
        if error is not None:
            return None, FileOperationResult(
                status="error",
                operation=operation,
                path=str(path),
                reason=error,
            )
        assert contents is not None
        had_bom = contents.startswith("\ufeff")
        normalized_contents = contents[1:] if had_bom else contents
        try:
            tree = ast.parse(normalized_contents)
        except SyntaxError as exc:
            return None, FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason=f"python_parse_error:{exc.lineno}:{exc.msg}",
            )
        return {"text": normalized_contents, "tree": tree, "had_bom": had_bom}, None

    def _validate_and_write_python_update(
        self,
        path: Path,
        operation: str,
        updated: str,
        *,
        had_bom: bool,
    ) -> FileOperationResult | None:
        validation_failure = self._validate_python_update_text(path, operation, updated)
        if validation_failure is not None:
            return validation_failure
        return self._write_python_update_text(
            path,
            operation,
            updated,
            had_bom=had_bom,
        )

    def _failed_symbol_lookup(
        self,
        path: Path,
        operation: str,
        symbol: tuple[int, int] | str | None,
    ) -> FileOperationResult | None:
        if symbol is None:
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="symbol_not_found",
            )
        if symbol == "ambiguous":
            return FileOperationResult(
                status="blocked",
                operation=operation,
                path=str(path),
                reason="multiple_symbols_found",
            )
        return None

    def _locate_python_method(
        self,
        tree: ast.AST,
        symbol_name: str,
    ) -> tuple[str, int, int] | str | None:
        matches: list[tuple[str, int, int]] = []

        def walk_class_bodies(body: list[ast.stmt], prefix: str = "") -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    class_name = f"{prefix}{node.name}"
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            matches.append(
                                (
                                    f"{class_name}.{child.name}",
                                    self._symbol_start_line(child),
                                    int(child.end_lineno),
                                )
                            )
                        elif isinstance(child, ast.ClassDef):
                            walk_class_bodies([child], prefix=f"{class_name}.")

        walk_class_bodies(getattr(tree, "body", []))
        exact_matches = [match for match in matches if match[0] == symbol_name]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return "ambiguous"
        if "." in symbol_name:
            return None
        fallback_matches = [
            match
            for match in matches
            if match[0].rsplit(".", 1)[-1] == symbol_name
        ]
        if len(fallback_matches) == 1:
            return fallback_matches[0]
        if len(fallback_matches) > 1:
            return "ambiguous"
        return None

    def _locate_python_callable_node(
        self,
        tree: ast.AST,
        symbol_name: str,
        *,
        callable_kind: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | str | None:
        matches: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

        def walk(body: list[ast.stmt], prefix: str = "", inside_class: bool = False) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    walk(node.body, prefix=f"{prefix}{node.name}.", inside_class=True)
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dotted_name = f"{prefix}{node.name}"
                    if callable_kind == "function" and not inside_class:
                        matches.append((dotted_name, node))
                    if callable_kind == "method" and inside_class:
                        matches.append((dotted_name, node))
                    walk(node.body, prefix=f"{dotted_name}.", inside_class=inside_class)

        walk(getattr(tree, "body", []))
        exact_matches = [node for name, node in matches if name == symbol_name]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return "ambiguous"
        if "." in symbol_name:
            return None
        fallback_matches = [
            node
            for name, node in matches
            if name.rsplit(".", 1)[-1] == symbol_name
        ]
        if len(fallback_matches) == 1:
            return fallback_matches[0]
        if len(fallback_matches) > 1:
            return "ambiguous"
        return None

    def _locate_top_level_export_node(
        self,
        tree: ast.AST,
        symbol_name: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | str | None:
        matches = [
            node
            for node in getattr(tree, "body", [])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol_name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return "ambiguous"
        return None

    def _has_unsafe_name_bindings(
        self,
        tree: ast.AST,
        target_name: str,
        *,
        allowed_def_nodes: set[ast.AST] | None = None,
        allowed_import_from_modules: set[str] | None = None,
    ) -> bool:
        allowed = allowed_def_nodes or set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == target_name and node not in allowed:
                    return True
            elif isinstance(node, ast.arg):
                if node.arg == target_name:
                    return True
            elif isinstance(node, ast.ExceptHandler):
                if node.name == target_name:
                    return True
            elif isinstance(node, ast.Name):
                if node.id == target_name and isinstance(node.ctx, (ast.Store, ast.Del)):
                    return True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound_name = alias.asname or alias.name.split(".", 1)[0]
                    if bound_name == target_name:
                        return True
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound_name = alias.asname or alias.name
                    if bound_name != target_name:
                        continue
                    if (
                        allowed_import_from_modules is not None
                        and node.level == 0
                        and node.module in allowed_import_from_modules
                        and alias.name == target_name
                    ):
                        continue
                    return True
        return False

    def _top_level_import_nodes(self, tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
        return [
            node
            for node in getattr(tree, "body", [])
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]

    def _movable_export_import_dependencies(
        self,
        text: str,
        tree: ast.AST,
        export_node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        *,
        export_name: str,
    ) -> tuple[list[str], set[str]]:
        import_bindings = self._top_level_import_bindings(text, tree)
        local_bindings = self._top_level_local_bindings(tree, export_name=export_name)
        loaded_names = self._loaded_name_ids(export_node)
        dependency_conflicts = loaded_names & local_bindings
        import_statements: list[str] = []
        seen_statements: set[str] = set()
        for referenced_name in sorted(loaded_names):
            for statement in import_bindings.get(referenced_name, []):
                if statement in seen_statements:
                    continue
                seen_statements.add(statement)
                import_statements.append(statement)
        return import_statements, dependency_conflicts

    def _top_level_import_bindings(
        self,
        text: str,
        tree: ast.AST,
    ) -> dict[str, list[str]]:
        line_offsets = self._line_offsets(text)
        bindings: dict[str, list[str]] = {}
        for node in self._top_level_import_nodes(tree):
            start, end = self._node_span(node, line_offsets)
            statement_text = text[start:end].rstrip("\r\n")
            if not statement_text:
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound_name = alias.asname or alias.name.split(".", 1)[0]
                    bindings.setdefault(bound_name, []).append(statement_text)
            else:
                for alias in node.names:
                    bound_name = alias.asname or alias.name
                    bindings.setdefault(bound_name, []).append(statement_text)
        return bindings

    def _top_level_local_bindings(
        self,
        tree: ast.AST,
        *,
        export_name: str,
    ) -> set[str]:
        bindings: set[str] = set()
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name != export_name:
                    bindings.add(node.name)
                continue
            bindings.update(self._statement_bound_names(node))
        return bindings

    def _statement_bound_names(self, node: ast.stmt) -> set[str]:
        names: set[str] = set()
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(self._target_bound_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(self._target_bound_names(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names.update(self._target_bound_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(self._target_bound_names(item.optional_vars))
        elif isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.name:
                    names.add(str(handler.name))
        elif isinstance(node, ast.Match):
            for case in node.cases:
                names.update(self._pattern_bound_names(case.pattern))
        return names

    def _target_bound_names(self, target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for element in target.elts:
                names.update(self._target_bound_names(element))
            return names
        if isinstance(target, ast.Starred):
            return self._target_bound_names(target.value)
        return set()

    def _pattern_bound_names(self, pattern: ast.AST) -> set[str]:
        if isinstance(pattern, ast.MatchAs):
            return {pattern.name} if pattern.name else set()
        if isinstance(pattern, ast.MatchStar):
            return {pattern.name} if pattern.name else set()
        if isinstance(pattern, ast.MatchSequence):
            names: set[str] = set()
            for element in pattern.patterns:
                names.update(self._pattern_bound_names(element))
            return names
        if isinstance(pattern, ast.MatchMapping):
            names: set[str] = set()
            for child in pattern.patterns:
                names.update(self._pattern_bound_names(child))
            if pattern.rest:
                names.add(pattern.rest)
            return names
        if isinstance(pattern, ast.MatchClass):
            names: set[str] = set()
            for child in pattern.patterns:
                names.update(self._pattern_bound_names(child))
            for child in pattern.kwd_patterns:
                names.update(self._pattern_bound_names(child))
            return names
        if isinstance(pattern, ast.MatchOr):
            names: set[str] = set()
            for child in pattern.patterns:
                names.update(self._pattern_bound_names(child))
            return names
        return set()

    def _loaded_name_ids(
        self,
        node: ast.AST,
    ) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    def _append_python_block(self, text: str, block: str) -> str:
        clean_block = block.strip("\r\n")
        if not clean_block:
            return text
        newline = "\r\n" if "\r\n" in text else "\n"
        if not text.strip():
            return clean_block + newline
        return text.rstrip("\r\n") + newline + newline + clean_block + newline

    def _python_import_insert_index(
        self,
        tree: ast.AST,
        lines: list[str],
    ) -> int:
        body = list(getattr(tree, "body", []))
        if not body:
            return 0
        insertion_line = 1
        body_index = 0
        if (
            isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            insertion_line = int(body[0].end_lineno) + 1
            body_index = 1
        while body_index < len(body):
            node = body[body_index]
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insertion_line = int(node.end_lineno) + 1
                body_index += 1
                continue
            break
        while body_index < len(body):
            node = body[body_index]
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insertion_line = int(node.end_lineno) + 1
                body_index += 1
                continue
            break
        return max(0, min(len(lines), insertion_line - 1))

    def _locate_python_symbol(
        self,
        tree: ast.AST,
        symbol_name: str,
        *,
        symbol_kind: str,
    ) -> tuple[int, int] | str | None:
        matches: list[tuple[str, int, int]] = []

        def walk(body: list[ast.stmt], prefix: str = "") -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    dotted_name = f"{prefix}{node.name}"
                    if symbol_kind in {"any", "class"}:
                        matches.append((dotted_name, self._symbol_start_line(node), int(node.end_lineno)))
                    walk(node.body, prefix=f"{dotted_name}.")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dotted_name = f"{prefix}{node.name}"
                    if symbol_kind in {"any", "function"}:
                        matches.append((dotted_name, self._symbol_start_line(node), int(node.end_lineno)))
                    walk(node.body, prefix=f"{dotted_name}.")

        walk(getattr(tree, "body", []))
        exact_matches = [(start, end) for name, start, end in matches if name == symbol_name]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return "ambiguous"
        if "." in symbol_name:
            return None
        fallback_matches = [
            (start, end)
            for name, start, end in matches
            if name.rsplit(".", 1)[-1] == symbol_name
        ]
        if len(fallback_matches) == 1:
            return fallback_matches[0]
        if len(fallback_matches) > 1:
            return "ambiguous"
        return None

    def _symbol_start_line(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        decorator_lines = [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
        return min([int(node.lineno), *decorator_lines])
