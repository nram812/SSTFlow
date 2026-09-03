"""Execute cells in Jupyter_SRDCNN_ResAFNO.20260903.ipynb and save outputs."""

import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr

notebook_path = "/esi/project/niwa03712/rampaln/PUBLICATIONS/2026/SSTDownscaling/SRDN/Jupyter_SRDCNN_ResAFNO.20260903.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

global_scope = {}
execution_count = 1

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        print(f"Executing code cell {execution_count}...")
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, global_scope)
            outputs = []
            stdout_text = stdout_buf.getvalue()
            stderr_text = stderr_buf.getvalue()
            if stdout_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_text.splitlines(keepends=True)
                })
            if stderr_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr_text.splitlines(keepends=True)
                })
            cell["outputs"] = outputs
            cell["execution_count"] = execution_count
        except Exception as e:
            print(f"Error in cell {execution_count}: {e}")
            import traceback
            tb = traceback.format_exc()
            cell["outputs"] = [{
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": tb.splitlines()
            }]
            cell["execution_count"] = execution_count
            break
        execution_count += 1

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=2)

print(f"Executed and saved notebook with pre-rendered outputs at: {notebook_path}")
