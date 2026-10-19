import ast
import asyncio
import ctypes
import sys
from typing import Any


_WORKER = r'''
import contextlib
import io
import sys

try:
    import resource
    limit = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
except ImportError:
    pass

class CappedOutput(io.StringIO):
    limit = 4000
    def write(self, value):
        remaining = self.limit - self.tell()
        if remaining > 0:
            super().write(str(value)[:remaining])
        return len(value)

def safe_print(*values, sep=" ", end="\n"):
    print(*values, sep=sep, end=end, file=output)

allowed = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "dict": dict, "enumerate": enumerate, "float": float, "int": int,
    "len": len, "list": list, "max": max, "min": min, "pow": pow,
    "print": safe_print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "ValueError": ValueError, "TypeError": TypeError,
    "RuntimeError": RuntimeError, "Exception": Exception,
    "KeyError": KeyError, "IndexError": IndexError, "NameError": NameError,
    "AttributeError": AttributeError, "ZeroDivisionError": ZeroDivisionError,
}
source = sys.stdin.read()
output = CappedOutput()
try:
    with contextlib.redirect_stdout(output):
        exec(compile(source, "<user-code>", "exec"), {"__builtins__": allowed})
    result = output.getvalue()
    if len(result) >= output.limit:
        result += "\n...[sortie tronquee]"
    print("Code sortie: 0")
    if result:
        print("stdout:")
        print(result, end="")
    else:
        print("Succes (aucune sortie console).")
except Exception as exc:
    print(f"Code sortie: 1\n{type(exc).__name__}: {exc}")
'''


class PythonCodeTool:
    """Exécute un sous-ensemble Python dans un processus séparé et limité."""

    name = "python_code"
    description = (
        "Execute un sous-ensemble Python pour les calculs et petits traitements. "
        "Les imports, accès aux attributs, au réseau et aux fichiers sont interdits."
    )
    args_schema = {"code": "Code Python complet a executer."}
    return_direct = True
    trigger_words: tuple[str, ...] = ("python", "execute", "code")
    timeout_seconds = 5
    max_code_chars = 16_000
    max_integer = 100_000

    _allowed_nodes = (
        ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Constant,
        ast.List, ast.Tuple, ast.Dict, ast.Set, ast.BinOp, ast.UnaryOp,
        ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword,
        ast.Subscript, ast.Slice, ast.For, ast.While, ast.If, ast.Break,
        ast.Continue, ast.Pass, ast.Raise, ast.Load, ast.Store,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Not, ast.Invert, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
        ast.NotIn, ast.Is, ast.IsNot,
    )

    def infer_args(self, user_input: str, args: dict[str, Any]) -> dict[str, Any]:
        if args.get("code"):
            return args
        if "```python" in user_input:
            code = user_input.split("```python", 1)[1].split("```", 1)[0].strip()
            return {**args, "code": code}
        return args

    @staticmethod
    def _limit_windows_process(process_id: int) -> tuple[Any, Any, Any]:
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        )
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        job = kernel32.CreateJobObjectW(None, None)
        process_handle = kernel32.OpenProcess(0x0100 | 0x0001, False, process_id)
        if not job or not process_handle:
            if job:
                kernel32.CloseHandle(job)
            if process_handle:
                kernel32.CloseHandle(process_handle)
            raise ctypes.WinError(ctypes.get_last_error())

        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x0100 | 0x2000
        limits.ProcessMemoryLimit = 256 * 1024 * 1024
        ok = kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        )
        ok = ok and kernel32.AssignProcessToJobObject(job, process_handle)
        if not ok:
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(process_handle)
            kernel32.CloseHandle(job)
            raise error
        return kernel32, job, process_handle

    async def execute(self, **kwargs: Any) -> str:
        code = str(kwargs.get("code", "")).strip()
        if not code:
            return "Erreur: Aucun code fourni."
        if len(code) > self.max_code_chars:
            return "Erreur: Code trop volumineux (limite 16000 caracteres)."

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Code sortie: 1\nSyntaxError: {exc.msg}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.names[0].name.split(".")[0] if node.names else "inconnu"
                return f"Import interdit: {module}"
            if isinstance(node, ast.Attribute):
                return "Erreur de sécurité : accès aux attributs interdit."
            if not isinstance(node, self._allowed_nodes):
                return f"Erreur de sécurité : syntaxe interdite '{type(node).__name__}'."
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return "Erreur de sécurité : noms spéciaux interdits."
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                if abs(node.value) > self.max_integer:
                    return "Erreur: Limite numerique depassee (100000)."

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                _WORKER,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            windows_handles = None
            if sys.platform == "win32":
                try:
                    windows_handles = self._limit_windows_process(process.pid)
                except OSError as exc:
                    process.kill()
                    await process.wait()
                    return f"Erreur sandbox Python : impossible de limiter le processus ({exc})."
            stdout, stderr = await asyncio.wait_for(
                process.communicate(code.encode("utf-8")),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return "Code sortie: 1\nTimeoutError: limite de 5 secondes dépassée."
        except OSError as exc:
            return f"Erreur de lancement Python: {exc}"
        finally:
            if "windows_handles" in locals() and windows_handles is not None:
                kernel32, job, process_handle = windows_handles
                kernel32.CloseHandle(process_handle)
                kernel32.CloseHandle(job)

        result = stdout.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
        if process.returncode and not result:
            detail = stderr.decode("utf-8", errors="replace")[:1000].strip()
            return f"Code sortie: 1\n{detail or 'Processus Python interrompu.'}"
        return result or "Code sortie: 0\nSucces (aucune sortie console)."
