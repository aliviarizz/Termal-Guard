"""
Script untuk mengeksekusi notebook 03_model_design_advanced_ml.ipynb
menggunakan nbclient.
"""
import nbformat
from nbclient import NotebookClient
import os

# Path notebook
notebook_path = r"d:\Lloyd\College\Semester5\DataScience\UAS\ProjectTermal\Termal-Guard\notebooks\03_model_design_advanced_ml.ipynb"

# Load notebook
with open(notebook_path, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Execute notebook
print("Mengeksekusi notebook...")
client = NotebookClient(
    nb,
    timeout=180,
    kernel_name="python3",
    resources={"metadata": {"path": os.path.dirname(notebook_path)}}
)
client.execute()

# Save notebook dengan output
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook berhasil dieksekusi dan disimpan!")

# Print summary output dari sel terakhir
for cell in nb.cells:
    if cell.cell_type == "code" and cell.outputs:
        for output in cell.outputs:
            if output.get("output_type") == "stream":
                print(output.get("text", "")[:500])
